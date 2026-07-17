"""Per-slide orchestration: read -> segment -> tile -> embed(s) -> sink -> receipt.

Multi-extractor fan-out is decode-once: each patch is read from storage a single
time per batch and reused across every requested model.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import ClassVar

from raw2features.benchmark.profiler import null_profiler
from raw2features.core import plugins, provenance
from raw2features.core.geometry import Point, Region, Size
from raw2features.core.store import grid_key
from raw2features.core.uris import (
    redact_uri_credentials,
    slide_id_from_source,
    source_uri,
)
from raw2features.embedders.fingerprint import (
    expected_patch_outputs,
    patch_output_fingerprint,
    resolved_patch_amp,
)
from raw2features.embedders.model_registry import build_embedder, resolve_geometry
from raw2features.patcher.grid import GridPatcher, resample_patch
from raw2features.sinks.zarr_sink import (
    ZarrSink,
    _grid_scaffold_is_usable,
    write_patches_geojson,
)

from .receipt import (
    SCHEMA_VERSION,
    Receipt,
    canonical_source_uri,
    config_hash,
    is_complete,
    store_source_bindings,
    validate_model,
    write_receipt,
)

_AMP = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", None: "float32"}


def _positive_float(value, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number greater than zero") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a finite number greater than zero")
    return number


def _positive_int(value, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer greater than zero")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer greater than zero") from exc
    if integer != value or integer <= 0:
        raise ValueError(f"{field} must be an integer greater than zero")
    return integer


@dataclass
class RunConfig:
    """Everything needed to embed one slide. Content-affecting fields feed the
    config hash; runtime knobs (device/batch_size) do not."""

    models: list[str]
    reader: str = "omezarr"
    segmenter: str = "otsu"
    no_seg: bool = False
    target_mpp: float = 1.0
    # Physical level-0 pixel size (µm/px) of the SOURCE, for sources whose OME-Zarr
    # declares no axis unit (uncalibrated -> reader.mpp is None). Distinct from
    # target_mpp (the extraction scale): this only supplies the missing source
    # calibration so extraction can proceed. None = require it from the metadata.
    source_mpp: float | None = None
    patch_px: int = 224
    step_px: int | None = None
    tissue_threshold: float = 0.1
    features_dtype: str = "float16"
    # Stain normalization applied to each patch BEFORE embedding (None | "macenko").
    # Content-affecting: changes the feature values (feeds the content hash), so a
    # with-vs-without experiment writes to separate output stores. Brightfield only.
    stain_norm: str | None = None
    snap_to_level: bool = False
    mpp_tolerance: float = 0.001
    allow_upsample: bool = False
    amp: str = "auto"  # "auto" = each model's card precision (spec.inference_amp)
    # runtime-only:
    device: str = "auto"  # "auto" -> best of cuda/mps/cpu (core.device.resolve_device)
    # Optional comma-separated device list for in-process multi-GPU. ``None`` (the
    # default) means "use the single ``device``" -- the existing single-device path
    # runs unchanged. When set, ``embed`` shards one slide's patches across these
    # devices (patch-parallel) and ``embed-many`` distributes slides across them
    # (slide-parallel). Runtime-only: it changes only *where* work runs, never the
    # patches extracted or the feature values (the gather preserves coord order
    # exactly), so it does not feed the content/grid hash.
    devices: str | None = None
    batch_size: int = 256
    read_workers: int = 8
    # Opt-in read fast path. read_block=N reads patches in N x N blocks (one larger
    # read_region per block, each patch sliced out in memory) instead of one read per
    # patch (N=1, the default). Bit-identical to the per-patch path (each patch's sliced
    # window == its per-patch read, fed to the same resample_patch), so it is
    # runtime-only. Bigger N => fewer, larger reads: a big win on latency-bound stores
    # (remote S3, HDD, slow networks) and on repeat/cached reads, but it costs RAM
    # (~ N^2 * read_px^2 * channels * workers, transient) and over-reads non-tissue
    # gaps -- so it gives little on fast local storage and wastes bandwidth on sparse
    # tissue. Recommended: 8 local, 16 remote/HDD, smaller for big multiplex panels.
    read_block: int = 1
    compile: bool = False  # torch.compile the model once at load (speed only)
    emit_geojson: bool = False
    output_zarr_format: int = 2
    emit_thumbnail: bool = False
    thumbnail_mpp: float = 8.0  # = viz.DEFAULT_THUMBNAIL_MPP (the segmentation MPP)
    thumbnail_max_px: int | None = None
    # Slide-level encoders to run after patch embedding (runtime-only: slide
    # embeddings are a post-processing step on the patch features and do not
    # change which patches are extracted or what patch features look like).
    slide_encoders: list[str] = field(default_factory=list)
    # Per-patch QC producer(s) to run after patch embedding, each writing an optional
    # ``grids/<key>/qc/<tool>/`` layer (e.g. ["grandqc"]). Runtime-only: QC scores are
    # derived auxiliary data; they do not change which patches or features are written.
    qc: list[str] = field(default_factory=list)
    # Stain normalization applied to the QC producer's input (None | "macenko"). Opt-in;
    # helps when a slide's stain is outside the QC model's training domain (e.g. GrandQC
    # otherwise reads eosin-heavy colorectal H&E as out-of-focus).
    qc_stain_norm: str | None = None
    # GrandQC artifact model scale in µm/px ("1.0"/"1.5"/"2.0"): coarser is less
    # focus-sensitive; raise it for a softer-scanned cohort that over-reads as blur.
    qc_artifact_mpp: str = "1.5"

    def __post_init__(self) -> None:
        """Reject invalid geometry before hashing, model loading, or store writes."""

        self.target_mpp = _positive_float(self.target_mpp, field="target_mpp")
        self.patch_px = _positive_int(self.patch_px, field="patch_px")
        if self.step_px is not None:
            self.step_px = _positive_int(self.step_px, field="step_px")
        if self.source_mpp is not None:
            self.source_mpp = _positive_float(self.source_mpp, field="source_mpp")

    # Single source of truth for which fields affect the output (and thus the
    # config hash) vs. runtime-only knobs. ``test_config_integrity`` asserts these
    # partition every RunConfig field, so a new field must be classified.
    _CONTENT_FIELDS: ClassVar[tuple[str, ...]] = (
        "reader",
        "models",
        "segmenter",
        "no_seg",
        "target_mpp",
        "source_mpp",
        "patch_px",
        "step_px",
        "tissue_threshold",
        "features_dtype",
        "stain_norm",
        "snap_to_level",
        "mpp_tolerance",
        "allow_upsample",
        "amp",
    )
    _RUNTIME_FIELDS: ClassVar[tuple[str, ...]] = (
        "device",
        "devices",
        "batch_size",
        "read_workers",
        "read_block",
        "compile",
        "emit_geojson",
        "output_zarr_format",
        "emit_thumbnail",
        "thumbnail_mpp",
        "thumbnail_max_px",
        "slide_encoders",
        "qc",
        "qc_stain_norm",
        "qc_artifact_mpp",
    )

    def _hash_payload(
        self,
        *,
        include_models: bool,
        include_amp: bool = True,
        models: list[str] | None = None,
    ) -> str:
        # Build the hashed payload from _CONTENT_FIELDS, applying the few value
        # transforms inline. config_hash sorts keys, so insertion order is
        # irrelevant -- the bytes are identical to listing the dict by hand.
        payload: dict = {}
        for name in self._CONTENT_FIELDS:
            if name == "amp" and not include_amp:
                continue
            if name == "models":
                if not include_models:
                    continue
                payload[name] = sorted(models if models is not None else self.models)
                continue
            value = getattr(self, name)
            if name == "segmenter":
                value = "none" if self.no_seg else value
            elif name == "step_px":
                value = self.step_px or self.patch_px
            elif name == "source_mpp" and self.source_mpp is None:
                # Absent by default (calibrated sources never set it) -> omit so it does
                # not perturb the hash of every existing run; only an actual override
                # (which changes the grid) contributes to the identity.
                continue
            payload[name] = value
        payload["schema_version"] = SCHEMA_VERSION
        return config_hash(payload)

    def content_hash(self, models: list[str] | None = None) -> str:
        """Identity of a complete run (geometry + model set); pins skip-if-complete.

        ``models`` overrides ``self.models`` (used to hash the union actually
        present in a store after an additive run).
        """
        return self._hash_payload(include_models=True, models=models)

    def grid_hash(self) -> str:
        """Identity of the shared extraction/storage grid, excluding model settings.

        AMP is per-model output identity and lives in the output fingerprint. Keeping
        it out here lets a precision change replace that model in the existing grid.
        """
        return self._hash_payload(include_models=False, include_amp=False)

    def legacy_grid_hash(self) -> str:
        """Pre-fingerprint grid identity, retained solely to append to v0.1 stores."""

        return self._hash_payload(include_models=False, include_amp=True)

    def compatible_legacy_grid_hashes(self) -> tuple[str, ...]:
        """Old grid identities, preferring this request's configured AMP first."""

        amps = (self.amp, "auto", "fp32", "bf16", "fp16")
        return tuple(
            dict.fromkeys(replace(self, amp=value).legacy_grid_hash() for value in amps)
        )

    def device_list(self) -> list[str]:
        """The devices to run on, in order. ``--devices`` overrides the single
        ``--device``; the default (``devices is None``) is ``[self.device]``, so the
        single-device path is unchanged. ``--devices auto`` uses every visible CUDA
        GPU (so a workstation user need not list them), falling back to
        ``[self.device]`` when there are 0 or 1. Order is preserved and duplicates are
        kept (``cpu,cpu`` / ``cuda:0,cuda:0`` is a valid two-worker config used to
        test equivalence on one device)."""
        from raw2features.core.device import resolve_device

        single = resolve_device(self.device)  # "auto" -> cuda/mps/cpu
        if not self.devices:
            return [single]
        if self.devices.strip() == "auto":
            try:
                import torch

                n = int(torch.cuda.device_count())
            except Exception:  # noqa: BLE001 - no torch/cuda visible -> single device
                n = 0
            return [f"cuda:{i}" for i in range(n)] if n > 1 else [single]
        out = [resolve_device(d.strip()) for d in self.devices.split(",") if d.strip()]
        return out or [single]


def slide_id_from_path(path: str) -> str:
    """Backward-compatible public wrapper around the shared source-ID helper."""

    return slide_id_from_source(path)


def _amp_dtype(amp: str):
    import torch

    return getattr(torch, _AMP.get(amp, "float32"))


def run_slide(
    slide_path: str,
    out_dir: str,
    cfg: RunConfig,
    *,
    receipts_dir: str | None = None,
    cli: str | None = None,
    embedders: list | None = None,
    embedder_factory=None,
    force: bool = False,
    profiler=None,
    allow_hashless_legacy: bool = True,
    _expected_model_contracts: dict[str, dict] | None = None,
) -> dict:
    """Embed one slide into exactly one configured grid. Returns a summary dict.

    This is the single-grid primitive. Library callers that want registry-recommended
    per-model geometry (and therefore potentially several grids) should use the
    top-level :func:`raw2features.embed_slide` entry point instead.

    Resume / additive behaviour:
      * A validated 'complete' receipt for this exact request short-circuits
        (the cohort resume path -- unchanged).
      * Otherwise, if a store already exists with the SAME patch geometry
        (``grid_hash``), only the requested models that are missing or invalid are
        embedded and appended in place; existing feature arrays and coords are left
        untouched (mirrors the slide-encoder path). ``force`` rebuilds from scratch.

    ``embedders`` may be passed pre-built (tests use a mock); otherwise they are
    built, for the models that actually need embedding, via the registry.

    Devices: with a single device (``cfg.device_list()`` has one entry -- the
    default) this is the original, unchanged single-device path. With ``--devices``
    listing several, the slide's patches are sharded across them (patch-parallel,
    latency mode): one model copy + reader per device, the per-device features
    gathered back in exact coord order, so the output is byte-identical to the
    single-device run. ``embedder_factory(device) -> list[Embedder]`` supplies the
    per-device model copies for that path (defaults to building from the registry on
    each device; tests inject a factory that replicates a mock). It is only consulted
    when more than one device is requested; the single-device path uses ``embedders``
    exactly as before.
    """
    from raw2features.slide_embedders.model_registry import (
        validate_slide_encoder_names,
    )

    validate_slide_encoder_names(cfg.slide_encoders)

    import torch  # noqa: F401 - imported so a torch-less env fails clearly, here

    from raw2features.core.device import resolve_device

    cfg.device = resolve_device(cfg.device)  # "auto" -> cuda/mps/cpu (idempotent)
    devices = cfg.device_list()
    multi_device = len(devices) > 1
    expected_source = canonical_source_uri(slide_path)
    if expected_source is None:
        raise ValueError("Source URI is malformed and cannot be compared safely.")
    try:
        slide_id = slide_id_from_path(slide_path)
    except Exception:  # noqa: BLE001 - suppress credential-bearing parser errors
        raise ValueError(
            "Source URI is malformed and no safe output ID can be derived."
        ) from None
    content_hash = cfg.content_hash()
    grid_hash = cfg.grid_hash()
    out_path = os.path.join(out_dir, f"{slide_id}.embeddings.zarr")
    expected_output = f"file://{os.path.abspath(out_path)}"
    store_exists = os.path.exists(out_path) and not force

    # The v0.1 basename rule deliberately remains stable for ordinary local paths.
    # Bind the deterministic output to its recorded source instead, before either a
    # receipt or store inspection can skip/append against a same-named slide.
    if store_exists:
        _assert_store_source(out_path, expected_source)

    if _expected_model_contracts is None:
        model_contracts = _expected_contracts_with_factory_probe(
            cfg,
            embedders,
            embedder_factory if multi_device else None,
            devices=devices,
        )
    else:
        model_contracts = dict(_expected_model_contracts)
        if set(model_contracts) != set(cfg.models):
            raise ValueError(
                "Pre-resolved model contracts do not match this grid's models."
            )

    # Fast path: a validated 'complete' receipt for this exact config (incl. the
    # model set), source, and requested output. ``--force`` deliberately bypasses it.
    if (
        not force
        and receipts_dir
        and not _runtime_aux_requested(cfg)
        and is_complete(
            receipts_dir,
            slide_id,
            content_hash,
            expected_source_uri=expected_source,
            expected_output_uri=expected_output,
            expected_model_contracts=model_contracts,
            expected_grid_models={grid_hash: list(cfg.models)},
            compatible_grid_hashes={grid_hash: cfg.compatible_legacy_grid_hashes()},
        )
    ):
        return {"slide_id": slide_id, "status": "skipped", "reason": "already complete"}

    present_valid: list[str] = []
    grid_key_existing: str | None = None
    if store_exists:
        grid_key_existing, _, present_valid = _inspect_store(
            out_path,
            grid_hash,
            cfg.models,
            expected_model_contracts=model_contracts,
            compatible_grid_hashes=cfg.compatible_legacy_grid_hashes(),
            allow_hashless_legacy=allow_hashless_legacy,
            require_mask=not cfg.no_seg,
        )
    # append: a grid of THIS geometry already exists -> add the missing models to it.
    # Otherwise we create a fresh store (no store yet) or ADD a new grid to an existing
    # one (a different geometry, written later) -- never a hard error, never a wipe.
    append = grid_key_existing is not None
    if append:
        models_to_do = [m for m in cfg.models if m not in present_valid]
    else:
        models_to_do = list(cfg.models)

    qc_to_do = list(cfg.qc)
    thumbnail_to_do = bool(cfg.emit_thumbnail)
    thumb_meta = None
    thumbnail_settings = None
    thumbnail_overwrite = not store_exists
    primary_grid_key = _primary_grid_key(out_path) if store_exists else None
    geojson_to_do = bool(cfg.emit_geojson)
    geojson_path = None
    if append:
        qc_to_do = _missing_qc_tools(out_path, grid_key_existing, list(cfg.qc))
        if cfg.emit_thumbnail:
            stored_thumbnail = _stored_thumbnail_metadata(out_path, grid_key_existing)
            if _thumbnail_files_complete(
                stored_thumbnail,
                out_dir,
                require_overlay=_grid_has_patches(out_path, grid_key_existing),
                expected_overlay=_grid_thumbnail_overlay_name(
                    slide_id,
                    grid_key_existing,
                    primary_grid_key or grid_key_existing,
                ),
            ):
                thumb_meta = stored_thumbnail
                thumbnail_to_do = False
            else:
                root_thumbnail = _stored_thumbnail_metadata(out_path)
                thumbnail_settings = stored_thumbnail or root_thumbnail
                thumbnail_overwrite = not _thumbnail_settings(thumbnail_settings)
        if cfg.emit_geojson:
            geojson_path = _grid_geojson_path(
                out_dir,
                slide_id,
                grid_key_existing,
                primary_grid_key or grid_key_existing,
            )
            geojson_to_do = not os.path.isfile(geojson_path)
    elif store_exists and cfg.emit_thumbnail:
        # The plain preview is slide-level, but the overlay is grid-specific. A new
        # grid reuses the primary preview's render settings and writes its own overlay.
        thumbnail_settings = _stored_thumbnail_metadata(out_path)
        thumbnail_overwrite = not _thumbnail_settings(thumbnail_settings)

    if (
        append
        and not models_to_do
        and not cfg.slide_encoders
        and not qc_to_do
        and not thumbnail_to_do
        and not geojson_to_do
    ):
        return {
            "slide_id": slide_id,
            "status": "skipped",
            "reason": "all requested models and auxiliary outputs already present",
            "models_present": present_valid,
            "grid": grid_key_existing,
            "geojson": geojson_path,
            "thumbnail": thumb_meta,
        }

    started = time.time()
    prof = profiler or null_profiler()
    prov = provenance.capture(cli)
    slide_results: dict[str, str] = {}
    sink = ZarrSink(output_zarr_format=cfg.output_zarr_format)
    actual_grid_key = grid_key_existing

    # Per-device model-copy factory for the patch-parallel path (only built when
    # >1 device). Defaults to building from the registry on each device; a test may
    # inject one to replicate a mock. The single-device path ignores it entirely.
    if embedder_factory is None:

        def embedder_factory(device: str) -> list:
            return _build_embedders_on(cfg, models_to_do, device)

    try:
        if models_to_do:
            with prof.stage("model_load"):
                # Single device: build exactly as before (filters injected mocks).
                # Multi-device: build one copy on the first device for the store
                # header/dims; the per-device worker copies are built in
                # _embed_patches_multi. We free this copy before launching workers
                # so peak memory stays at one model copy per device.
                run_embedders = (
                    embedder_factory(devices[0])
                    if multi_device
                    else _select_embedders(
                        embedders, cfg, models_to_do, device=devices[0]
                    )
                )
            run_contracts = {name: model_contracts[name] for name in models_to_do}
            _assert_loaded_model_contracts(run_embedders, run_contracts)
            # Modality of this run: multiplex (marker stacks, e.g. KRONOS) routes the
            # nuclear segmenter + N-channel reads; brightfield is the RGB default.
            multiplex = any(
                getattr(e, "modality", "brightfield") == "multiplex"
                for e in run_embedders
            )
            if multiplex and multi_device:
                # The patch-parallel workers don't bind the marker panel or read
                # native channels, so multiplex on >1 device would embed wrong data.
                # Reject it cleanly rather than produce silently-corrupt features.
                raise ValueError(
                    "multiplex models (e.g. kronos) are not supported on the "
                    "multi-device path; run with a single --device (the panel is "
                    "bound per-slide and channels are read natively only there)."
                )
            reader_cls = plugins.get("readers", cfg.reader)
            with reader_cls(slide_path) as reader:
                _warn_channel_collapse(reader, multiplex)
                # An uncalibrated source (no axis unit -> reader.mpp is None) needs an
                # explicit physical pixel size; apply --source-mpp when given, else the
                # grid build fails loud below with an actionable message.
                if (
                    reader.mpp is None
                    and cfg.source_mpp is not None
                    and hasattr(reader, "apply_source_mpp")
                ):
                    reader.apply_source_mpp(cfg.source_mpp)
                # Bind each multiplex model's marker panel once (no-op for brightfield),
                # capturing the kept/dropped-marker summary for the store provenance -
                # the resolved panel is part of how a multiplex embedding was produced.
                panel_meta: dict[str, dict] = {}
                if multiplex:
                    for e in run_embedders:
                        panel_meta[e.name] = e.set_panel(reader.channel_names)
                if append:
                    (
                        n,
                        coords,
                        read_level,
                        read_px,
                        patch_px,
                        level0_patch,
                    ) = _store_geometry(out_path, grid_key_existing)
                    sink.open_append(
                        out_dir,
                        slide_id,
                        key=grid_key_existing,
                        new_model_dims={e.name: e.embedding_dim for e in run_embedders},
                        new_model_meta=_models_header(run_embedders, run_contracts),
                        replace_models=models_to_do,
                    )
                else:
                    patcher = GridPatcher(
                        target_mpp=cfg.target_mpp,
                        patch_px=cfg.patch_px,
                        step_out_px=cfg.step_px,
                        snap_to_level=cfg.snap_to_level,
                        mpp_tolerance=cfg.mpp_tolerance,
                        allow_upsample=cfg.allow_upsample,
                    )
                    grid = patcher.build_grid(reader)
                    with prof.stage("segment"):
                        tissue, seg_meta = _segment(
                            reader, cfg, "nuclear" if multiplex else None
                        )
                    with prof.stage("tile"):
                        coords, grid_index, grid_tissue = patcher.tile(
                            grid, tissue, cfg.tissue_threshold
                        )
                    n = int(coords.shape[0])
                    read_level, read_px, patch_px = (
                        grid.read_level,
                        grid.read_px,
                        grid.patch_px,
                    )
                    level0_patch = grid.level0_patch
                    header = _build_header(
                        reader,
                        grid,
                        seg_meta,
                        run_embedders,
                        slide_id,
                        n,
                        None,
                        grid_hash,
                        prov,
                        panel_meta,
                        run_contracts,
                    )
                    actual_grid_key = sink.create(
                        out_dir,
                        slide_id,
                        grid=grid_key(grid.target_mpp, grid.patch_px),
                        fresh=not store_exists,  # add a grid to an existing store
                        n_patches=n,
                        coords=coords,
                        grid_index=grid_index,
                        grid_tissue=None if cfg.no_seg else grid_tissue,
                        model_dims={e.name: e.embedding_dim for e in run_embedders},
                        header=header,
                        features_dtype=cfg.features_dtype,
                    )
                    if thumbnail_to_do:
                        with prof.stage("thumbnail"):
                            thumb_meta = _write_grid_thumbnail(
                                reader,
                                sink,
                                out_dir,
                                slide_id,
                                actual_grid_key,
                                primary_grid_key or actual_grid_key,
                                cfg,
                                tissue,
                                coords,
                                level0_patch,
                                settings=thumbnail_settings,
                                overwrite=thumbnail_overwrite,
                            )
                if append and thumbnail_to_do:
                    with prof.stage("thumbnail"):
                        thumbnail_tissue, _ = _segment(
                            reader,
                            cfg,
                            _stored_grid_segmenter(out_path, grid_key_existing),
                        )
                        thumb_meta = _write_grid_thumbnail(
                            reader,
                            sink,
                            out_dir,
                            slide_id,
                            actual_grid_key,
                            primary_grid_key or actual_grid_key,
                            cfg,
                            thumbnail_tissue,
                            coords,
                            level0_patch,
                            settings=thumbnail_settings,
                            overwrite=thumbnail_overwrite,
                        )
                if geojson_to_do:
                    if not append:
                        geojson_path = _grid_geojson_path(
                            out_dir,
                            slide_id,
                            actual_grid_key,
                            primary_grid_key or actual_grid_key,
                        )
                    geojson_path = write_patches_geojson(
                        out_dir,
                        slide_id,
                        coords,
                        level0_patch,
                        filename=os.path.basename(geojson_path),
                    )
                # Stain-norm (--stain-norm): fit the slide's stain once from a
                # thumbnail, then normalize patches (brightfield). Content-affecting.
                normalizer = None
                if cfg.stain_norm and multiplex:
                    import warnings

                    warnings.warn(
                        "--stain-norm is H&E-only; skipped for this multiplex run.",
                        stacklevel=2,
                    )
                elif cfg.stain_norm:
                    from raw2features.core.stain import make_normalizer
                    from raw2features.viz import render_thumbnail

                    normalizer = make_normalizer(
                        cfg.stain_norm, render_thumbnail(reader, max_px=1024).image
                    )
                if multi_device:
                    # Patch-parallel: free the header copy so peak memory is one
                    # model copy per device, then shard coords across devices and
                    # gather features back in coord order (output byte-identical).
                    model_dims = {e.name: e.embedding_dim for e in run_embedders}
                    for e in run_embedders:
                        e.unload()
                    _embed_patches_multi(
                        slide_path,
                        coords,
                        read_level,
                        read_px,
                        patch_px,
                        embedder_factory,
                        model_dims,
                        sink,
                        cfg,
                        devices,
                        normalizer,
                        run_contracts,
                    )
                else:
                    # Marker panels were bound once when the reader opened (above).
                    _embed_patches(
                        reader,
                        coords,
                        read_level,
                        read_px,
                        patch_px,
                        run_embedders,
                        sink,
                        cfg,
                        prof,
                        multichannel=multiplex,
                        normalizer=normalizer,
                        device=devices[0],
                    )
                # The array fingerprint is the completion commit marker. Stamp it
                # only after every row for every model was written successfully.
                sink.finalize_models(run_contracts)
                # Optional per-patch QC (--qc): the producer needs the open reader,
                # so it runs here on a freshly-built grid (its coords + level0_patch). A
                # later model added to an existing grid keeps that grid's qc layer.
                if qc_to_do:
                    with prof.stage("qc"):
                        _run_qc(
                            qc_to_do,
                            reader,
                            sink,
                            coords,
                            level0_patch,
                            cfg.device,
                            cfg.qc_stain_norm,
                            cfg.qc_artifact_mpp,
                        )
        else:
            # Existing complete patch arrays: open the grid for slide encoders and/or
            # produce-if-missing runtime outputs without loading patch embedders.
            n = sink.open_append(
                out_dir,
                slide_id,
                key=grid_key_existing,
                new_model_dims={},
                new_model_meta={},
            )
            actual_grid_key = grid_key_existing
            if qc_to_do or thumbnail_to_do or geojson_to_do:
                (
                    n,
                    coords,
                    _read_level,
                    _read_px,
                    _patch_px,
                    level0_patch,
                ) = _store_geometry(out_path, grid_key_existing)
                if qc_to_do or thumbnail_to_do:
                    reader_cls = plugins.get("readers", cfg.reader)
                    with reader_cls(slide_path) as reader:
                        if (
                            reader.mpp is None
                            and cfg.source_mpp is not None
                            and hasattr(reader, "apply_source_mpp")
                        ):
                            reader.apply_source_mpp(cfg.source_mpp)
                        if thumbnail_to_do:
                            with prof.stage("thumbnail"):
                                thumbnail_tissue, _ = _segment(
                                    reader,
                                    cfg,
                                    _stored_grid_segmenter(out_path, grid_key_existing),
                                )
                                thumb_meta = _write_grid_thumbnail(
                                    reader,
                                    sink,
                                    out_dir,
                                    slide_id,
                                    actual_grid_key,
                                    primary_grid_key or actual_grid_key,
                                    cfg,
                                    thumbnail_tissue,
                                    coords,
                                    level0_patch,
                                    settings=thumbnail_settings,
                                    overwrite=thumbnail_overwrite,
                                )
                        if qc_to_do:
                            with prof.stage("qc"):
                                _run_qc(
                                    qc_to_do,
                                    reader,
                                    sink,
                                    coords,
                                    level0_patch,
                                    cfg.device,
                                    cfg.qc_stain_norm,
                                    cfg.qc_artifact_mpp,
                                )
                if geojson_to_do:
                    geojson_path = write_patches_geojson(
                        out_dir,
                        slide_id,
                        coords,
                        level0_patch,
                        filename=os.path.basename(geojson_path),
                    )

        # Slide-level encoding (inline, -s flag): reads patch features from the
        # store (existing + just-written), so no WSI re-read needed.
        if cfg.slide_encoders:
            available = sorted(set(present_valid) | set(models_to_do))
            # Multi-grid: a slide encoder runs only on the grid with its patch encoder
            # (e.g. titan on the conch_v1_5 grid); on other grids it is skipped here and
            # runs when its grid is processed. embed_slide checks each requested encoder
            # ran on some grid (else a clear error).
            encoders_here = _slide_encoders_for(cfg.slide_encoders, available)
            if encoders_here:
                with prof.stage("slide_encode"):
                    slide_results = _run_slide_encoders(
                        sink, encoders_here, cfg.device, available
                    )

        final_dims = sink.feature_dims()
        with prof.stage("consolidate"):
            sink.close()
        elapsed = time.time() - started

        if receipts_dir:
            write_receipt(
                receipts_dir,
                Receipt(
                    slide_id=slide_id,
                    status="complete",
                    source_uri=source_uri(slide_path),
                    output_uri=sink.uri,
                    reader=cfg.reader,
                    models=sorted(final_dims),
                    config_hash=cfg.content_hash(sorted(final_dims)),
                    n_patches=n,
                    model_dims=final_dims,
                    started_utc=provenance.now_utc_iso(),
                    finished_utc=provenance.now_utc_iso(),
                    elapsed_s=round(elapsed, 2),
                    host=prov.get("host"),
                    raw2features_version=prov.get("raw2features_version"),
                ),
            )
        return {
            "slide_id": slide_id,
            "status": "complete",
            "n_patches": n,
            "models": sorted(final_dims),
            "models_added": models_to_do,
            "models_skipped": present_valid,
            "grid": actual_grid_key,
            "slide_embeddings": slide_results,
            "output_uri": sink.uri,
            "geojson": geojson_path,
            "thumbnail": thumb_meta,
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as exc:  # noqa: BLE001 - record failure in the receipt
        if receipts_dir:
            write_receipt(
                receipts_dir,
                Receipt(
                    slide_id=slide_id,
                    status="failed",
                    source_uri=source_uri(slide_path),
                    output_uri="",
                    reader=cfg.reader,
                    models=cfg.models,
                    config_hash=content_hash,
                    error=redact_uri_credentials(f"{type(exc).__name__}: {exc}"),
                ),
            )
        raise


def _run_content_hash(group_cfgs: list[RunConfig]) -> str:
    """Identity of a whole multi-grid request: the (order-independent) set of per-grid
    content hashes. Two requests with the same grids+models+geometry match, so an
    identical re-run short-circuits via the receipt."""
    return config_hash(
        {
            "groups": sorted(gc.content_hash() for gc in group_cfgs),
            "schema_version": SCHEMA_VERSION,
        }
    )


def _runtime_aux_requested(cfg: RunConfig) -> bool:
    """Whether completion must inspect produce-if-missing runtime outputs."""

    return bool(cfg.qc or cfg.emit_thumbnail or cfg.emit_geojson or cfg.slide_encoders)


def _record_root_attr(out_path: str, key: str, value) -> None:
    """Merge ``key: value`` into the store's root ``raw2features`` header (best-effort).

    Used to record the job's extraction plan so an explicit ``--config`` run replays
    from the store itself. Provenance is non-critical, so any failure is swallowed.
    """
    try:
        import zarr

        root = zarr.open_group(out_path, mode="r+", use_consolidated=False)
        rh = dict(root.attrs.get("raw2features", {}))
        rh[key] = value
        root.attrs["raw2features"] = rh
        zarr.consolidate_metadata(root.store)
    except Exception:  # noqa: BLE001 - provenance recording must never fail a run
        pass


def resolve_run(
    cfg: RunConfig,
    requested_mpp: float | None = None,
    requested_patch_px: int | None = None,
    geometry_config: list[dict] | None = None,
    *,
    model_specs: dict | None = None,
):
    """``(groups, group_cfgs, run_hash)`` for a request.

    ``groups`` are the resolved geometry groups; ``group_cfgs`` are per-grid RunConfigs
    (cfg with that group's models + geometry); ``run_hash`` is the request identity
    used for the receipt. Shared by :func:`embed_slide` and ``verify`` so verify hashes
    identically to the embed that produced the store.
    """
    groups = resolve_geometry(
        cfg.models,
        requested_mpp,
        requested_patch_px,
        geometry_config,
        specs=model_specs,
    )
    if not groups:
        raise ValueError("at least one model/extraction is required")
    group_cfgs = [
        replace(cfg, models=list(g.models), target_mpp=g.mpp, patch_px=g.patch_px)
        for g in groups
    ]
    return groups, group_cfgs, _run_content_hash(group_cfgs)


def embed_slide(
    slide_path: str,
    out_dir: str,
    cfg: RunConfig,
    *,
    requested_mpp: float | None = None,
    requested_patch_px: int | None = None,
    geometry_config: list[dict] | None = None,
    receipts_dir: str | None = None,
    cli: str | None = None,
    embedders: list | None = None,
    embedder_factory=None,
    force: bool = False,
    profiler=None,
) -> dict:
    """Resolve per-model geometry and embed each group as its own grid in one store.

    The high-level public entry point. Without an mpp/patch override, models that
    recommend different geometries are each extracted at their own
    ``(mpp, patch_px)`` into a separate ``grids/<key>/``. An mpp-only override
    retains each model's extraction size; supplying both mpp and patch size
    collapses them onto one grid (see
    :func:`raw2features.embedders.model_registry.resolve_geometry`). It drives the
    single-grid :func:`run_slide` once per group: the first writes the store, each
    later group *finds-or-creates* its grid (never a wipe). ``force`` wipes once, on
    the first group. A single per-slide receipt records the whole request for fast
    skip.

    ``embedders`` (a warm worker's pre-built set for all of ``cfg.models``) is passed
    through to every group; ``run_slide`` selects each group's subset. Their
    :class:`~raw2features.embedders.base.ModelSpec` objects also drive geometry for
    models outside the packaged registry. When only ``embedder_factory`` is supplied,
    it is probed once for the same specifications and output contracts. High-level
    overrides belong in ``requested_mpp`` / ``requested_patch_px`` (or
    ``geometry_config``);
    ``RunConfig.target_mpp`` and ``RunConfig.patch_px`` remain the concrete geometry
    consumed by the single-grid :func:`run_slide` primitive.
    """
    from raw2features.slide_embedders.model_registry import (
        validate_slide_encoder_names,
    )

    validate_slide_encoder_names(cfg.slide_encoders)

    from raw2features.core.device import resolve_device

    cfg.device = resolve_device(cfg.device)
    devices = cfg.device_list()

    expected_source = canonical_source_uri(slide_path)
    if expected_source is None:
        raise ValueError("Source URI is malformed and cannot be compared safely.")
    try:
        slide_id = slide_id_from_path(slide_path)
    except Exception:  # noqa: BLE001 - suppress credential-bearing parser errors
        raise ValueError(
            "Source URI is malformed and no safe output ID can be derived."
        ) from None
    model_specs = {embedder.name: embedder.spec for embedder in (embedders or [])}
    probed_contracts: dict[str, dict] | None = None
    if not model_specs and embedder_factory is not None:
        # A factory can be the only source for a programmatic external model. Probe it
        # once before geometry resolution, then pass the resulting contracts into each
        # run_slide call so the same loaded copy is not probed a second time.
        model_specs, probed_contracts = _probe_factory_contracts(
            cfg, embedder_factory, devices
        )
    groups, group_cfgs, run_hash = resolve_run(
        cfg,
        requested_mpp,
        requested_patch_px,
        geometry_config,
        model_specs=model_specs,
    )
    out_path = os.path.join(out_dir, f"{slide_id}.embeddings.zarr")
    grids = {grid_key(g.mpp, g.patch_px): list(g.models) for g in groups}
    expected_output = f"file://{os.path.abspath(out_path)}"
    model_contracts = probed_contracts or _expected_contracts_with_factory_probe(
        cfg,
        embedders,
        embedder_factory,
        devices=devices,
    )
    expected_grid_models = {
        group_cfg.grid_hash(): list(group_cfg.models) for group_cfg in group_cfgs
    }
    compatible_grid_hashes = {
        group_cfg.grid_hash(): group_cfg.compatible_legacy_grid_hashes()
        for group_cfg in group_cfgs
    }

    if os.path.exists(out_path) and not force:
        _assert_store_source(out_path, expected_source)

    if (
        not force
        and receipts_dir
        and not _runtime_aux_requested(cfg)
        and is_complete(
            receipts_dir,
            slide_id,
            run_hash,
            expected_source_uri=expected_source,
            expected_output_uri=expected_output,
            expected_model_contracts=model_contracts,
            expected_grid_models=expected_grid_models,
            compatible_grid_hashes=compatible_grid_hashes,
        )
    ):
        grids = _stored_grid_summary(out_path, group_cfgs) or grids
        return {
            "slide_id": slide_id,
            "status": "skipped",
            "reason": "already complete",
            "grids": grids,
        }

    started = time.time()
    results = []
    for i, gc in enumerate(group_cfgs):
        results.append(
            run_slide(
                slide_path,
                out_dir,
                gc,
                receipts_dir=None,  # the orchestrator owns the per-slide receipt
                cli=cli,
                embedders=embedders,
                embedder_factory=embedder_factory,
                force=(force and i == 0),  # force wipes once; later groups add grids
                profiler=profiler,
                # A lone hashless legacy grid is backward-compatible for one
                # requested geometry, but cannot identify which of several grids it
                # represents. A multi-grid request therefore leaves it untouched.
                allow_hashless_legacy=(len(group_cfgs) == 1),
                _expected_model_contracts={
                    model: model_contracts[model] for model in gc.models
                },
            )
        )
    grids = {
        (result.get("grid") or grid_key(group.mpp, group.patch_px)): list(group.models)
        for group, result in zip(groups, results, strict=True)
    }
    status = "skipped" if all(r["status"] == "skipped" for r in results) else "complete"

    # Coverage: first use outputs produced on this request's grids. If a specific
    # encoder's patch model lives only on another, already-existing grid, discover it
    # with the same selection path as standalone `slide-embed` and produce it there.
    # An explicit -s request never returns success without a complete output.
    if cfg.slide_encoders:
        ran: set[str] = set()
        for r in results:
            ran |= set(r.get("slide_embeddings") or {})
        missing = [s for s in cfg.slide_encoders if s not in ran]
        if missing:
            ran |= set(_run_slide_encoders_from_store(out_path, missing, cfg.device))
            missing = [s for s in cfg.slide_encoders if s not in ran]
        if missing:
            from raw2features.slide_embedders.model_registry import get_slide_spec

            need = []
            for s in missing:
                try:
                    need.append(f"{s} (needs {get_slide_spec(s).patch_encoder})")
                except KeyError:
                    need.append(s)
            raise ValueError(
                f"{slide_id}: slide encoder(s) {', '.join(need)} found no grid with "
                "their patch encoder -- embed that patch model (at its geometry) first."
            )

    elapsed = round(time.time() - started, 2)

    # Record the explicit job knobs (extraction plan, stain norm) so the run replays
    # from one artifact and the store is self-describing about how it was made.
    if status != "skipped" and (geometry_config is not None or cfg.stain_norm):
        job: dict = {}
        if geometry_config is not None:
            job["geometry_config"] = geometry_config
        if cfg.stain_norm:
            job["stain_norm"] = cfg.stain_norm
        _record_root_attr(out_path, "job", job)

    if receipts_dir:
        all_models = sorted({m for ms in grids.values() for m in ms})
        write_receipt(
            receipts_dir,
            Receipt(
                slide_id=slide_id,
                status="complete",
                source_uri=source_uri(slide_path),
                output_uri=f"file://{os.path.abspath(out_path)}",
                reader=cfg.reader,
                models=all_models,
                config_hash=run_hash,
                started_utc=provenance.now_utc_iso(),
                finished_utc=provenance.now_utc_iso(),
                elapsed_s=elapsed,
                host=provenance.capture(cli).get("host"),
                raw2features_version=provenance.capture(cli).get(
                    "raw2features_version"
                ),
            ),
        )
    return {
        "slide_id": slide_id,
        "status": status,
        "grids": grids,
        "output_uri": f"file://{os.path.abspath(out_path)}",
        "per_grid": results,
        "elapsed_s": elapsed,
    }


def _resolve_amp(cfg: RunConfig, spec, device: str | None = None) -> str:
    """The precision the forward path actually uses on the concrete device."""

    return resolved_patch_amp(spec, cfg.amp, device or cfg.device)


def expected_model_contracts(
    cfg: RunConfig,
    embedders: list | None = None,
    *,
    device: str | None = None,
) -> dict:
    """Current output dimension/fingerprint for every model requested by *cfg*.

    Injected embedders override registry specs by name, preserving the public plugin
    and test seam while normal registry runs stay load-free for receipt/store checks.
    """

    specs = {e.name: e.spec for e in (embedders or [])}
    return expected_patch_outputs(
        cfg.models,
        cfg.amp,
        cfg.device if device is None else device,
        specs=specs,
    )


def _expected_contracts_for_devices(
    cfg: RunConfig,
    embedders: list | None,
    devices: list[str],
) -> dict[str, dict]:
    """Return one output contract shared by every configured patch worker.

    A feature array has one fingerprint. Mixed devices are therefore safe only when
    they resolve every model to the same complete contract (for example explicit
    fp32 on CPU and CUDA). Device-dependent precision differences fail before any
    receipt or store can be used or mutated.
    """

    if not devices:
        raise ValueError("At least one patch worker device is required.")
    by_device = {
        device: expected_model_contracts(cfg, embedders, device=device)
        for device in dict.fromkeys(devices)
    }
    first_device = next(iter(by_device))
    expected = by_device[first_device]
    mismatches: list[str] = []
    for name in cfg.models:
        contracts = {device: values[name] for device, values in by_device.items()}
        if any(value != contracts[first_device] for value in contracts.values()):
            amps = ", ".join(
                f"{device}="
                f"{value['output_fingerprint']['payload']['output']['resolved_amp']}"
                for device, value in contracts.items()
            )
            mismatches.append(f"{name} ({amps})")
    if mismatches:
        raise ValueError(
            "Configured patch worker devices resolve to different model output "
            f"contracts: {', '.join(mismatches)}. Use devices with matching "
            "effective precision or pass --amp fp32."
        )
    return expected


def _expected_contracts_with_factory_probe(
    cfg: RunConfig,
    embedders: list | None,
    embedder_factory,
    *,
    devices: list[str] | None = None,
) -> dict[str, dict]:
    """Resolve custom-only factory specs without leaving a model copy resident."""

    worker_devices = list(devices or cfg.device_list())
    try:
        return _expected_contracts_for_devices(cfg, embedders, worker_devices)
    except KeyError:
        factory = embedder_factory or (
            lambda device: _build_embedders_on(cfg, cfg.models, device)
        )
    _specs, contracts = _probe_factory_contracts(cfg, factory, worker_devices)
    return contracts


def _probe_factory_contracts(
    cfg: RunConfig,
    factory,
    worker_devices: list[str],
) -> tuple[dict[str, object], dict[str, dict]]:
    """Probe a factory once for external specs and the shared worker contract."""

    if not worker_devices:
        raise ValueError("At least one patch worker device is required.")
    probe = factory(worker_devices[0])
    try:
        contracts = _expected_contracts_for_devices(cfg, probe, worker_devices)
        _assert_loaded_model_contracts(probe, contracts)
        specs = {embedder.name: embedder.spec for embedder in probe}
        return specs, contracts
    finally:
        for embedder in probe:
            embedder.unload()


def _amp_label(dtype) -> str:
    """torch dtype -> amp string, for recording the effective precision used."""
    if dtype is None:
        return "fp32"  # no dtype set (e.g. a not-yet-loaded model) - don't import torch
    import torch

    return {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
    }.get(dtype, "fp32")


def _loaded_model_contracts(embedders: list) -> dict[str, dict]:
    """Derive contracts from loaded specs and their effective execution precision."""

    contracts: dict[str, dict] = {}
    for embedder in embedders:
        selected = _amp_label(getattr(embedder, "_dtype", None))
        effective = resolved_patch_amp(
            embedder.spec,
            selected,
            getattr(embedder, "_device", "cpu"),
        )
        contracts[embedder.name] = {
            "embedding_dim": int(embedder.embedding_dim),
            "output_fingerprint": patch_output_fingerprint(embedder.spec, effective),
        }
    return contracts


def _assert_loaded_model_contracts(embedders: list, expected: dict[str, dict]) -> None:
    """Fail before store mutation if loaded model copies differ from provenance."""

    actual = _loaded_model_contracts(embedders)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            "Loaded model set does not match the output contract "
            f"(missing={missing}, unexpected={extra})."
        )
    mismatched = [name for name in expected if actual[name] != expected[name]]
    if mismatched:
        raise ValueError(
            "Loaded model contract differs from the requested/persisted contract for "
            f"{mismatched}. Check effective AMP, preprocessing, dimensions, and "
            "weights."
        )


def load_embedders(cfg: RunConfig, device: str | None = None) -> list:
    """Build + load every model in ``cfg.models`` once, at its resolved precision.

    Used by the warm worker (``embed-many``) to pay the model-load cost a single
    time and reuse the embedders across a shard of slides. The returned list can be
    passed straight to ``run_slide(..., embedders=...)``. ``device`` overrides
    ``cfg.device`` so the slide-parallel path can build one model copy per device.
    """
    return _build_embedders_on(cfg, cfg.models, device or cfg.device)


def _build_embedders_on(cfg: RunConfig, names: list[str], device: str) -> list:
    """Build the embedders for *names* on *device*, each at its resolved precision.

    The single seam for replicating models onto a device: the single-device build,
    the warm worker, and each multi-GPU worker all go through here, so a per-device
    copy is loaded exactly as the single-device run loads its one copy (same models,
    same ``--amp`` resolution) -- only the target device differs.
    """
    from raw2features.core.device import resolve_device

    device = resolve_device(device)  # "auto" -> cuda/mps/cpu before any .to(device)
    built = []
    for m in names:
        emb = build_embedder(m)
        _warn_scale_mismatch(cfg, emb.spec)
        dtype = _amp_dtype(_resolve_amp(cfg, emb.spec, device))
        built.append(emb.load(device=device, dtype=dtype, compile=cfg.compile))
    return built


def _warn_scale_mismatch(cfg: RunConfig, spec) -> None:
    """Note when the extraction scale differs from the model's conventional one.

    ``recommended_mpp`` is the scale the model is *commonly* run at - usually the
    pathology-FM convention of ~0.5 µm/px (20x), and for most models inferred from the
    paper rather than stated on the card. Scale does affect FM embeddings, but the best
    extraction MPP is task-dependent and not prescriptive (e.g. UNI is used at both 20x
    and 10x), so this is an informational note, not a correction - the run proceeds.
    """
    import warnings

    rec = spec.recommended_mpp
    if rec is None or rec <= 0:
        return
    if abs(cfg.target_mpp - rec) / rec > 0.1:
        warnings.warn(
            f"{spec.name}: extracting at target_mpp={cfg.target_mpp} "
            f"(~{round(10 / cfg.target_mpp)}x); this model is commonly run at "
            f"~{rec} µm/px (~{round(10 / rec)}x). Scale affects FM embeddings - pick "
            f"deliberately for your task; pass --mpp to set it explicitly.",
            stacklevel=3,
        )


def _warn_channel_collapse(reader, multiplex: bool) -> None:
    """Warn when a brightfield run silently drops channels of a multi-channel source.

    The RGB read path keeps the first three channels, so feeding a brightfield model a
    source whose ``omero`` lists more than three channels (a multiplex/CODEX stack)
    discards the rest with no diagnostic - exactly the kind of garbage-in the embeddings
    would hide. The reverse (a multiplex model on a source with no marker panel) already
    fails loudly when the panel is bound, so it needs no guard here.
    """
    if multiplex:
        return
    names = getattr(reader, "channel_names", None) or []
    if len(names) > 3:
        import warnings

        warnings.warn(
            f"source has {len(names)} channels ({', '.join(names[:4])}…) but a "
            "brightfield (RGB) model keeps only the first three; the rest are dropped. "
            "If this is a multiplex slide, run a multiplex model (e.g. kronos).",
            stacklevel=2,
        )


def _select_embedders(
    embedders: list | None,
    cfg: RunConfig,
    names: list[str],
    *,
    device: str | None = None,
) -> list:
    """The embedders to run for *names*: filter injected ones (tests), else build.

    Each model is loaded at its resolved precision (``--amp auto`` -> the model's
    card precision; an explicit ``--amp`` overrides all), so a single decode-once run
    can mix precisions across models.
    """
    if embedders is not None:
        return [e for e in embedders if e.name in names]
    return _build_embedders_on(cfg, names, device or cfg.device)


def _segment(reader, cfg: RunConfig, segmenter_name: str | None = None):
    """Return (tissue, seg_meta); (None, none-meta) when segmentation is off.

    ``segmenter_name`` overrides ``cfg.segmenter`` (multiplex slides route to the
    ``nuclear`` segmenter, which thresholds the DAPI/Hoechst channel).
    """
    if cfg.no_seg:
        return None, {"segmenter": "none"}
    name = segmenter_name or cfg.segmenter
    seg = plugins.get("segmenters", name)()
    tissue = seg.segment(reader)
    return tissue, {
        "segmenter": name,
        "tissue_threshold": cfg.tissue_threshold,
        "seg_level": tissue.level,
        "seg_downsample": tissue.downsample,
    }


def _decode_one(reader, x, y, read_level, read_px, patch_px, multichannel=False):
    """Read + resample a single patch at the store geometry (one grid cell).

    ``multichannel`` (multiplex models) reads every channel natively
    (``read_region_channels`` -> ``[H,W,C]``); otherwise the RGB ``read_region`` path.
    """
    region = Region(read_level, Point(int(x), int(y)), Size(read_px, read_px))
    raw = (
        reader.read_region_channels(region)
        if multichannel
        else reader.read_region(region)
    )
    return resample_patch(raw, patch_px)


# Warn (never clamp) above this transient block RAM, so a large read_block / multiplex
# misconfig is flagged before it OOMs. Generous: only clearly-risky configs trip it.
_READ_BLOCK_RAM_WARN_BYTES = 4 * 1024**3  # 4 GiB


def _warn_read_block_memory(reader, cfg, read_px, read_workers, multichannel) -> None:
    """One-line transient-RAM heads-up for read_block > 1 (warn-only, never clamps).

    A read_block=N block holds ~ ``(N*read_px)**2 * bytes_per_px`` in host RAM
    transiently, times ``read_workers`` concurrent blocks. We estimate and warn only
    above a generous threshold -- the user chooses the size; we just surface the cost so
    a big read_block (especially on a many-marker multiplex panel) cannot OOM by
    surprise. It is host RAM, not GPU VRAM (batch_size still governs VRAM, unchanged).
    """
    import warnings

    side = int(cfg.read_block)
    if multichannel:
        nch = len(reader.channel_names or []) or 1
        bytes_per_px = nch * 2  # marker stack, ~uint16
    else:
        bytes_per_px = 3  # RGB uint8
    per_block = (side * read_px) ** 2 * bytes_per_px
    total = per_block * read_workers
    if total > _READ_BLOCK_RAM_WARN_BYTES:
        warnings.warn(
            f"read_block={side}: ~{per_block / 1024**2:.0f} MiB per block x "
            f"{read_workers} workers ~= {total / 1024**3:.1f} GiB transient host RAM "
            f"(read_px={read_px}, {bytes_per_px} B/px). Lower --read-block or "
            "--read-workers if memory-constrained.",
            stacklevel=2,
        )


def _decode_batch(
    reader,
    batch,
    read_level,
    read_px,
    patch_px,
    executor=None,
    multichannel=False,
    read_block=1,
):
    """Decode one batch's patch list once at the store geometry (read + resample).

    Read read_px px then resample to exactly patch_px -> the stored patch is at the
    target MPP, independent of each model's input_size (resized per-model). The list
    is decode-once: it is reused across every embedder by the consumer.

    With an *executor* the per-patch reads run concurrently: zarr/blosc decompress
    releases the GIL, so threads give real parallelism and concurrent chunk fetches
    hide per-read filesystem latency (the bottleneck on a network FS). Equivalence is
    exact regardless of worker count -- ``executor.map`` yields results in argument
    order, so the returned list is identical to the serial comprehension below,
    patch-for-patch. ``read_region`` calls touch disjoint slices and return fresh
    arrays, so they are independent and thread-safe.

    With ``read_block`` > 1 the batch's reads are grouped into ``read_block`` x
    ``read_block`` blocks (one larger ``read_region`` per block, patches sliced out in
    memory) -- fewer, larger reads on latency-bound stores. Bit-identical to the
    per-patch path: see :func:`_decode_batch_blocked`.
    """
    if read_block > 1:
        return _decode_batch_blocked(
            reader,
            batch,
            read_level,
            read_px,
            patch_px,
            executor,
            multichannel,
            side=read_block,
        )
    if executor is None:
        return [
            _decode_one(reader, x, y, read_level, read_px, patch_px, multichannel)
            for x, y in batch
        ]
    return list(
        executor.map(
            lambda xy: _decode_one(
                reader, xy[0], xy[1], read_level, read_px, patch_px, multichannel
            ),
            batch,
        )
    )


def _decode_batch_blocked(
    reader, batch, read_level, read_px, patch_px, executor, multichannel, side
):
    """Block read path: one ``read_region`` per ``side`` x ``side`` block, sliced out.

    Bit-identical to the per-patch path. ``read_region`` maps a level-0 point to the
    read-level array index via the reader's per-axis mapping (``read_level_mapping`` ->
    ``round(x/ds_x + off_x)`` / ``round(y/ds_y + off_y)``) and slices a ``read_px``
    window (zero-padded at borders). This path uses the *same* mapping for the block
    anchor and
    each patch, so a block anchored at the group's bounding-box top-left ``(xa, ya)``
    reads the contiguous read-level region from ``map(xa, ya)``; patch ``j`` sits at
    integer offset ``map(xj, yj) - map(xa, ya)`` within it, giving the exact same native
    pixels the per-patch read would return (borders included). The resampled patch is
    therefore identical, and results are returned in ``batch`` order. ``side`` (patches
    per block edge) bounds the bounding box -- and thus transient RAM and the over-read
    of non-tissue gaps.
    """
    import numpy as np

    coords = np.asarray(batch, dtype=np.int64).reshape(-1, 2)
    n = coords.shape[0]
    if n == 0:
        return []
    # One mapping, shared with read_region, so the block anchor and per-patch offsets
    # agree exactly (per-axis downsample + translation offset; isotropic 0-offset for
    # the common pyramid). Fall back to the scalar downsample for a reader without it.
    if hasattr(reader, "read_level_mapping"):
        dsx, dsy, ox, oy = reader.read_level_mapping(read_level)
    else:  # pragma: no cover - all built-in readers provide read_level_mapping
        ds = float(reader.level_downsamples()[read_level])
        dsx, dsy, ox, oy = ds, ds, 0.0, 0.0
    xr = np.round(coords[:, 0] / dsx + ox).astype(np.int64)  # read-level top-left (x)
    yr = np.round(coords[:, 1] / dsy + oy).astype(np.int64)  # read-level top-left (y)

    tile_px = max(1, side * read_px)
    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        groups.setdefault((int(xr[i] // tile_px), int(yr[i] // tile_px)), []).append(i)

    read_one = reader.read_region_channels if multichannel else reader.read_region
    out: list = [None] * n

    def _do_group(idxs: list[int]) -> None:
        xa = int(coords[idxs, 0].min())
        ya = int(coords[idxs, 1].min())
        xa_r, ya_r = int(round(xa / dsx + ox)), int(round(ya / dsy + oy))
        w = int(xr[idxs].max()) - xa_r + read_px
        h = int(yr[idxs].max()) - ya_r + read_px
        block = read_one(Region(read_level, Point(xa, ya), Size(w, h)))
        for i in idxs:
            cy, cx = int(yr[i]) - ya_r, int(xr[i]) - xa_r
            sub = block[cy : cy + read_px, cx : cx + read_px]
            out[i] = resample_patch(sub, patch_px)

    items = list(groups.values())
    if executor is None:
        for idxs in items:
            _do_group(idxs)
    else:
        list(executor.map(_do_group, items))
    return out


def _group_by_transform(embedders: list) -> list[list]:
    """Group embedders by ``transform_signature``, preserving first-seen order.

    Members of a group share preprocessing, so the transformed tensor computed for
    the first member is reused (bit-for-bit) by the rest. Order is preserved within
    and across groups, so the work is identical to the per-model loop apart from not
    recomputing the shared transform.
    """
    groups: dict[object, list] = {}
    for emb in embedders:
        groups.setdefault(emb.transform_signature, []).append(emb)
    return list(groups.values())


def _embed_patches(
    reader,
    coords,
    read_level,
    read_px,
    patch_px,
    embedders,
    sink,
    cfg,
    prof=None,
    multichannel=False,
    normalizer=None,
    device=None,
):
    """Decode each patch once at the store geometry and embed it with every
    extractor in *embedders*, writing ``features/<model>`` in batches.

    Decode-once: a patch is read and resampled a single time per batch and reused
    across every model -- so adding one model to a store re-reads the WSI once, not
    once per existing model. ``prof`` (a benchmark Profiler) attributes time to the
    read / transform / gpu / write sub-stages; it is a no-op in production.

    Transform-once: models that share a preprocessing signature
    (``Embedder.transform_signature`` -- input_size, mean, std, interpolation)
    produce a bit-identical transformed tensor, so the H2D copy + normalise is run
    *once per signature* and the resulting device tensor is fed to every model in
    the group via ``embed_batch``. ``embed_batch`` only reads the tensor, so sharing
    it is safe. A single-model run is one group of one (behaviour unchanged), and
    each model still writes ``features/<name>`` for ``coords[start:...]`` 1:1.

    Prefetch: a single background worker decodes upcoming batches (the read-bound,
    GIL-releasing zarr work) into a bounded queue while the main thread runs the
    transform/GPU/write of the current batch -- so reads overlap GPU compute instead
    of stalling it. Within a batch, ``cfg.read_workers`` decode threads fetch the
    patches concurrently (a single reusable pool for the whole slide), hiding
    per-read latency. Equivalence is preserved exactly: one producer enqueues batches
    in coord order and one consumer dequeues them in the same order, ``executor.map``
    keeps each batch's patches in coord order, each patch is read once and reused
    across models, and the rows written for ``start`` are the features of
    ``coords[start:start+batch_size]`` 1:1, as before. Independent ``read_region``
    calls touch disjoint slices of per-level arrays, so the concurrent reads are
    thread-safe alongside the main thread's compute.

    This is the single-device default path: it runs ``coords`` on ``cfg.device`` and
    writes straight to ``sink``. The patch-parallel multi-GPU path
    (``_embed_patches_multi``) is a thin wrapper that drives the very same per-batch
    loop (``_run_batches``) once per device over a coord shard.
    """
    _run_batches(
        reader,
        coords,
        read_level,
        read_px,
        patch_px,
        embedders,
        device or cfg.device,
        sink.write_block,
        cfg,
        prof,
        multichannel,
        normalizer,
    )


def _run_batches(
    reader,
    coords,
    read_level,
    read_px,
    patch_px,
    embedders,
    device,
    write_block,
    cfg,
    prof=None,
    multichannel=False,
    normalizer=None,
):
    """Core decode -> transform -> embed -> write loop over ``coords`` on ``device``.

    Factored out of :func:`_embed_patches` so it can be reused verbatim by each
    patch-parallel worker (its own reader / embedders / device / coord shard). The
    only generalisation over the original inline loop is that the transform runs on
    the passed ``device`` and features are handed to ``write_block(model, start,
    feats)`` (the single-device path passes ``sink.write_block``; a worker passes a
    collector that stores into its shard buffer). Behaviour for the single-device
    case is identical -- ``device == cfg.device`` and the writes are 1:1 the same.
    """
    prof = prof or null_profiler()
    n = int(coords.shape[0])
    starts = list(range(0, n, cfg.batch_size))

    # Group models by preprocessing signature once (it is static across batches):
    # the transform is computed once per group and shared across its members.
    transform_groups = _group_by_transform(embedders)

    # One reusable decode pool for the whole slide (avoids per-batch thread churn).
    # read_workers <= 1 keeps the serial path (no pool, no threads).
    read_workers = max(1, int(cfg.read_workers))
    if cfg.read_block > 1:
        _warn_read_block_memory(reader, cfg, read_px, read_workers, multichannel)
    pool_ctx = (
        ThreadPoolExecutor(max_workers=read_workers, thread_name_prefix="r2f-read")
        if read_workers > 1
        else nullcontext(None)
    )

    # (start, patches) in batch order; maxsize bounds memory to ~2 batches in flight.
    work: queue.Queue = queue.Queue(maxsize=2)
    _DONE = object()  # sentinel: producer finished (cleanly or via error)
    stop = threading.Event()  # set by the consumer to unblock the producer on error
    error: list[BaseException] = []

    with pool_ctx as executor:

        def _producer() -> None:
            try:
                for start in starts:
                    if stop.is_set():
                        break
                    batch = coords[start : start + cfg.batch_size]
                    with prof.stage("read"):
                        patches = _decode_batch(
                            reader,
                            batch,
                            read_level,
                            read_px,
                            patch_px,
                            executor,
                            multichannel,
                            read_block=cfg.read_block,
                        )
                        if normalizer is not None:  # per-patch stain norm (brightfield)
                            patches = [normalizer(p) for p in patches]
                        nch = patches[0].shape[2] if patches else 3
                        prof.add_bytes(len(patches) * read_px * read_px * nch)
                    # Poll for a consumer abort so a full queue can't deadlock join().
                    while not stop.is_set():
                        try:
                            work.put((start, patches), timeout=0.5)
                            break
                        except queue.Full:
                            continue
            except BaseException as exc:  # noqa: BLE001 - surfaced to the main thread
                error.append(exc)
            finally:
                work.put(_DONE)

        worker = threading.Thread(target=_producer, name="r2f-prefetch", daemon=True)
        worker.start()
        drained = False  # True once the single _DONE sentinel has been consumed
        try:
            while True:
                item = work.get()
                if item is _DONE:
                    drained = True
                    break
                start, patches = item
                for group in transform_groups:
                    # Batched on-device transform: one H2D copy + vectorised normalise
                    # on the GPU, instead of a per-patch CPU transform that competes
                    # with the (bottleneck) reads. embed_batch's .to(device) is a no-op.
                    # Computed once per group (shared signature -> identical tensor) and
                    # reused by every model in the group; embed_batch only reads it.
                    with prof.stage("transform"):
                        batch_tensor = group[0].transform_batch(patches, device)
                    for emb in group:
                        with prof.stage("gpu"):
                            feats = (
                                emb.embed_batch(batch_tensor)
                                .numpy()
                                .astype(cfg.features_dtype)
                            )
                        with prof.stage("write"):
                            write_block(emb.name, start, feats)
        finally:
            # On a consumer-side error we leave the loop before the sentinel: signal
            # the producer to stop and drain until its (guaranteed) _DONE so a full
            # queue can't deadlock join(). Normally the sentinel is already consumed.
            stop.set()
            while not drained:
                if work.get() is _DONE:
                    drained = True
            worker.join()
    if error:
        raise error[0]


class _FeatureCollector:
    """In-memory write target with the sink's ``write_block`` signature.

    A patch-parallel worker writes its shard's features here (into a contiguous,
    pre-allocated ``(shard_n, dim)`` buffer per model) instead of to the zarr store,
    so workers never touch the same array concurrently. The main thread then
    concatenates the per-shard buffers in shard order -- which is coord order, since
    shards are contiguous -- and writes them once, making the on-disk result
    byte-identical to the single-device run regardless of how many devices ran.
    """

    def __init__(self, shard_n: int, dims: dict[str, int], dtype: str) -> None:
        import numpy as np

        self.arrays = {m: np.empty((shard_n, d), dtype=dtype) for m, d in dims.items()}

    def write_block(self, model: str, start: int, feats) -> None:
        self.arrays[model][start : start + feats.shape[0]] = feats


def _contiguous_shards(n: int, k: int) -> list[tuple[int, int]]:
    """Split ``range(n)`` into ``k`` contiguous ``(lo, hi)`` blocks, as even as
    possible. Concatenating the blocks in order reproduces ``range(n)`` exactly, so
    a per-shard gather preserves coord order. Empty blocks (when ``k > n``) are
    dropped so we never spawn a worker with no work."""
    base, extra = divmod(n, k)
    shards: list[tuple[int, int]] = []
    lo = 0
    for i in range(k):
        hi = lo + base + (1 if i < extra else 0)
        if hi > lo:
            shards.append((lo, hi))
        lo = hi
    return shards


def _embed_patches_multi(
    slide_path,
    coords,
    read_level,
    read_px,
    patch_px,
    embedder_factory,
    model_dims,
    sink,
    cfg,
    devices,
    normalizer=None,
    expected_contracts: dict[str, dict] | None = None,
):
    """Patch-parallel embedding: shard ``coords`` across ``devices`` and gather.

    One contiguous coord block per device; each device gets its own reader (the
    chunk cache is per-reader; ``read_region`` touches disjoint slices, so the
    readers are independent) and its own embedder copies built on that device (via
    ``embedder_factory(device)``). Each worker runs the exact single-device per-batch
    loop (:func:`_run_batches`) over its block, writing into an in-memory
    :class:`_FeatureCollector`. When all workers finish, the per-block buffers are
    concatenated in block order -- i.e. coord order, since the blocks are contiguous
    and ordered -- and written to ``sink`` once. The decode/transform/embed of a
    given patch is identical to the single-device path (same code, same device-kind
    arithmetic when the devices are the same), and the gather restores exact coord
    order, so the on-disk features are byte-identical to the single-device run.

    This is the latency mode for a single slide on a multi-GPU box. It is only used
    when ``len(devices) > 1``; one device falls back to :func:`_embed_patches`.
    """
    import numpy as np

    reader_cls = plugins.get("readers", cfg.reader)
    shards = _contiguous_shards(int(coords.shape[0]), len(devices))
    if not shards:
        return  # no patches: feature arrays were created empty, nothing to embed
    collectors: list[_FeatureCollector] = [None] * len(shards)  # type: ignore[list-item]
    errors: list[BaseException] = []
    err_lock = threading.Lock()

    def _worker(idx: int, device: str, lo: int, hi: int) -> None:
        try:
            embedders = embedder_factory(device)
            if expected_contracts is not None:
                _assert_loaded_model_contracts(embedders, expected_contracts)
            collector = _FeatureCollector(hi - lo, model_dims, cfg.features_dtype)
            # Own reader per worker: per-reader chunk cache, thread-safe zarr reads.
            with reader_cls(slide_path) as reader:
                # prof=None: aggregate stage timings across concurrent workers aren't
                # meaningful, so the patch-parallel path is not profiled per stage.
                _run_batches(
                    reader,
                    coords[lo:hi],
                    read_level,
                    read_px,
                    patch_px,
                    embedders,
                    device,
                    collector.write_block,
                    cfg,
                    None,
                    False,
                    normalizer,
                )
            for emb in embedders:
                emb.unload()
            collectors[idx] = collector
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller below
            with err_lock:
                errors.append(exc)

    threads = [
        threading.Thread(
            target=_worker,
            args=(idx, devices[idx], lo, hi),
            name=f"r2f-dev{idx}",
        )
        for idx, (lo, hi) in enumerate(shards)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]

    # Gather in shard order == coord order (shards are contiguous, ascending) and
    # write each model's full column once. ``write_block`` at 0 writes rows 0..n.
    for model in model_dims:
        full = np.concatenate([c.arrays[model] for c in collectors], axis=0)
        sink.write_block(model, 0, full)


def _inspect_store(
    out_path: str,
    expected_grid_hash: str,
    requested_models: list[str],
    *,
    expected_model_contracts: dict[str, dict] | None = None,
    compatible_grid_hashes: tuple[str, ...] = (),
    allow_hashless_legacy: bool = True,
    require_mask: bool = False,
):
    """Find the grid matching ``expected_grid_hash``: ``(key | None, n, valid_models)``.

    Returns the key of the grid whose geometry matches (so the caller appends missing
    models to it), or ``None`` when no grid matches (the caller adds a NEW grid). A grid
    with no recorded ``grid_hash`` (a legacy/hand-built store) is trusted only when it
    is the store's sole grid; several hashless grids are not geometrically
    distinguishable. ``valid_models`` is the subset of *requested_models* already
    present + output-valid in that grid, so the caller embeds only the rest.
    """
    import zarr

    from raw2features.core.store import GRIDS, grid_keys

    try:
        # Completion is a live-metadata decision. Consolidated metadata can retain a
        # pre-crash fingerprint after replacement removed the array commit marker.
        root = zarr.open_group(out_path, mode="r", use_consolidated=False)
        keys = grid_keys(root)
    except Exception:  # noqa: BLE001 - an unreadable/absent store is "not appendable"
        return (None, 0, [])

    def _result_for(k: str):
        g = root[GRIDS][k]
        try:
            header = dict(g.attrs.get("raw2features", {}))
            patching = header.get("patching", {})
            patching = patching if isinstance(patching, dict) else {}
            declared_n = patching.get("n_patches")
            expected_n = declared_n if isinstance(declared_n, int) else None
            grid_shape = patching.get("grid_shape")
            expected_mask_shape = (
                tuple(int(v) for v in grid_shape)
                if require_mask
                and isinstance(grid_shape, (list, tuple))
                and len(grid_shape) == 2
                else None
            )
            if not _grid_scaffold_is_usable(
                g,
                expected_n=expected_n,
                require_mask=require_mask,
                expected_mask_shape=expected_mask_shape,
            ):
                return None
            n = int(g["coords"].shape[0])
        except Exception:  # noqa: BLE001 - skip an unreadable grid
            return None
        valid = []
        for model in requested_models:
            contract = (expected_model_contracts or {}).get(model, {})
            if validate_model(
                g,
                model,
                n,
                expected_dim=contract.get("embedding_dim"),
                expected_fingerprint=contract.get("output_fingerprint"),
            ):
                valid.append(model)
        return (k, n, valid)

    # Prefer the new identity, then the exact requested legacy AMP, then other legal
    # legacy AMP values. A hashless hand-built/legacy grid is the last-resort match.
    candidates = tuple(dict.fromkeys((expected_grid_hash, *compatible_grid_hashes)))
    for candidate in candidates:
        for k in keys:
            stored = dict(root[GRIDS][k].attrs.get("raw2features", {})).get("grid_hash")
            if stored == candidate and (result := _result_for(k)) is not None:
                return result
    if allow_hashless_legacy and len(keys) == 1:
        k = keys[0]
        stored = dict(root[GRIDS][k].attrs.get("raw2features", {})).get("grid_hash")
        if stored is None and (result := _result_for(k)) is not None:
            return result
    return (None, 0, [])


def _assert_store_source(out_path: str, expected_source_uri: str) -> None:
    """Refuse to reuse a store whose live root/grid provenance names another slide."""

    expected = canonical_source_uri(expected_source_uri)
    if expected is None:
        raise ValueError("Refusing to reuse existing store: invalid source binding.")
    try:
        bindings = store_source_bindings(out_path)
    except Exception as exc:  # noqa: BLE001 - fail safe before any store mutation
        raise ValueError(
            f"Refusing to reuse existing store {out_path!r}: its source provenance "
            "could not be read. Use --force to replace it deliberately, or choose "
            "a different output directory."
        ) from exc

    missing = [label for label, recorded in bindings if recorded is None]
    invalid = [
        label
        for label, recorded in bindings
        if recorded is not None and canonical_source_uri(recorded) is None
    ]
    mismatched = [
        (label, canonical)
        for label, recorded in bindings
        if recorded is not None
        and (canonical := canonical_source_uri(recorded)) is not None
        and canonical != expected
    ]
    if not bindings or missing or invalid or mismatched:
        details: list[str] = []
        if missing or not bindings:
            labels = ", ".join(missing) if missing else "root/grids"
            details.append(f"missing source.uri at {labels}")
        if invalid:
            details.append(f"invalid source.uri at {', '.join(invalid)}")
        if mismatched:
            seen = ", ".join(f"{label}={uri}" for label, uri in mismatched)
            details.append(f"recorded {seen}; requested {expected}")
        raise ValueError(
            f"Refusing to reuse existing store {out_path!r}: "
            f"{'; '.join(details)}. Ordinary local outputs use basename IDs, so "
            "same-named slides must not share an output directory. Use --force to "
            "replace this store deliberately, or choose a different output directory."
        )


def _store_geometry(out_path: str, key: str | None = None):
    """Read coords + read-geometry of an existing grid (for an additive append)."""
    import numpy as np
    import zarr

    from raw2features.core.store import open_grid

    root = zarr.open_group(out_path, mode="r", use_consolidated=False)
    g = open_grid(root, key)  # the grid being appended to (sole if key is None)
    p = dict(g.attrs.get("raw2features", {}))["patching"]
    coords = np.asarray(g["coords"][:])
    return (
        int(coords.shape[0]),
        coords,
        int(p["read_level"]),
        int(p["read_px"]),
        int(p["patch_px"]),
        int(p["level0_patch"]),
    )


def _missing_qc_tools(
    out_path: str, key: str | None, requested: list[str]
) -> list[str]:
    """Requested QC producers whose active-grid layer is absent or incomplete."""

    if not requested:
        return []
    try:
        import zarr

        from raw2features.core.store import open_grid

        root = zarr.open_group(out_path, mode="r", use_consolidated=False)
        group = open_grid(root, key)
        n_patches = int(group["coords"].shape[0])
        present = {
            tool for tool in requested if _qc_tool_complete(group, tool, n_patches)
        }
    except Exception:  # noqa: BLE001 - unreadable auxiliary state is missing
        return list(requested)
    return [tool for tool in requested if tool not in present]


def _qc_tool_complete(group, tool: str, n_patches: int) -> bool:
    """Whether a QC producer left its complete generic per-patch array contract."""

    try:
        if "qc" not in group or tool not in group["qc"]:
            return False
        tool_group = group["qc"][tool]
        if dict(tool_group.attrs).get("complete") is not True:
            return False
        if "scores" not in tool_group:
            return False
        scores = tool_group["scores"]
        classes = dict(scores.attrs).get("classes")
        if (
            scores.ndim != 2
            or int(scores.shape[0]) != n_patches
            or dict(scores.attrs).get("role") != "qc"
            or not isinstance(classes, (list, tuple))
            or len(classes) != int(scores.shape[1])
        ):
            return False
        for name in tool_group.keys():
            array = tool_group[name]
            if not hasattr(array, "shape"):  # a nested group, not a QC array
                continue
            if (
                array.ndim < 1
                or int(array.shape[0]) != n_patches
                or dict(array.attrs).get("role") != "qc"
            ):
                return False
        return True
    except Exception:  # noqa: BLE001 - an unreadable layer is incomplete
        return False


def _thumbnail_files_complete(
    metadata: object,
    out_dir: str,
    *,
    require_overlay: bool,
    expected_overlay: str | None = None,
) -> bool:
    if not isinstance(metadata, dict):
        return False
    plain = metadata.get("plain")
    overlay = metadata.get("overlay")
    if (
        not isinstance(plain, str)
        or os.path.basename(plain) != plain
        or not os.path.isfile(os.path.join(out_dir, plain))
    ):
        return False
    if require_overlay and not isinstance(overlay, str):
        return False
    if isinstance(overlay, str) and os.path.basename(overlay) != overlay:
        return False
    if require_overlay and expected_overlay is not None and overlay != expected_overlay:
        return False
    return not isinstance(overlay, str) or os.path.isfile(
        os.path.join(out_dir, overlay)
    )


def _stored_thumbnail_metadata(out_path: str, key: str | None = None) -> dict | None:
    """Read live root or per-grid thumbnail metadata, even when an asset is missing."""

    try:
        import zarr

        from raw2features.core.store import open_grid

        root = zarr.open_group(out_path, mode="r", use_consolidated=False)
        owner = root if key is None else open_grid(root, key)
        metadata = dict(owner.attrs.get("raw2features", {})).get("thumbnail")
        if isinstance(metadata, dict):
            return dict(metadata)
    except Exception:  # noqa: BLE001 - unreadable metadata is absent
        pass
    return None


def _thumbnail_settings(metadata: object) -> tuple[float | None, int | None] | None:
    """Recover one coherent render setting from stored thumbnail metadata."""

    try:
        if not isinstance(metadata, dict):
            return None
        max_px = metadata.get("max_px")
        if max_px is not None:
            max_px = int(max_px)
            return (None, max_px) if max_px > 0 else None
        mpp = float(metadata.get("mpp"))
        return (mpp, None) if mpp > 0 else None
    except (TypeError, ValueError):
        pass
    return None


def _primary_grid_key(out_path: str) -> str | None:
    """The first-created grid, which owns backward-compatible root sidecars."""

    try:
        import zarr

        from raw2features.core.store import grid_keys

        root = zarr.open_group(out_path, mode="r", use_consolidated=False)
        indexed = dict(root.attrs.get("raw2features", {})).get("grids")
        if isinstance(indexed, dict) and indexed:
            return next(iter(indexed))
        keys = grid_keys(root)
        return keys[0] if keys else None
    except Exception:  # noqa: BLE001 - unreadable store has no primary grid
        return None


def _grid_has_patches(out_path: str, key: str | None) -> bool:
    try:
        import zarr

        from raw2features.core.store import open_grid

        root = zarr.open_group(out_path, mode="r", use_consolidated=False)
        return int(open_grid(root, key)["coords"].shape[0]) > 0
    except Exception:  # noqa: BLE001 - fail closed and require the overlay
        return True


def _stored_grid_segmenter(out_path: str, key: str | None) -> str | None:
    """The effective segmenter recorded by the grid (not merely the CLI default)."""

    try:
        import zarr

        from raw2features.core.store import open_grid

        root = zarr.open_group(out_path, mode="r", use_consolidated=False)
        header = dict(open_grid(root, key).attrs.get("raw2features", {}))
        name = (header.get("segmentation") or {}).get("segmenter")
        return name if isinstance(name, str) and name != "none" else None
    except Exception:  # noqa: BLE001 - the request's segmenter remains the fallback
        return None


def _grid_geojson_path(out_dir: str, slide_id: str, key: str, primary_key: str) -> str:
    """Backward-compatible primary path; namespaced path for every other grid."""

    filename = (
        f"{slide_id}.patches.geojson"
        if key == primary_key
        else f"{slide_id}.{key}.patches.geojson"
    )
    return os.path.join(out_dir, filename)


def _grid_thumbnail_overlay_name(slide_id: str, key: str, primary_key: str) -> str:
    return (
        f"{slide_id}.thumbnail.overlay.png"
        if key == primary_key
        else f"{slide_id}.{key}.thumbnail.overlay.png"
    )


def _write_grid_thumbnail(
    reader,
    sink,
    out_dir: str,
    slide_id: str,
    key: str,
    primary_key: str,
    cfg: RunConfig,
    tissue,
    coords,
    level0_patch: int,
    *,
    settings: dict | None,
    overwrite: bool,
) -> dict:
    """Write a coherent preview plus this grid's own overlay and bind metadata."""

    from raw2features.viz import write_thumbnails

    stored = _thumbnail_settings(settings)
    mpp = cfg.thumbnail_mpp if stored is None or stored[0] is None else stored[0]
    max_px = cfg.thumbnail_max_px if stored is None else stored[1]
    overlay_name = _grid_thumbnail_overlay_name(slide_id, key, primary_key)
    metadata = write_thumbnails(
        reader,
        out_dir,
        slide_id,
        mpp=mpp,
        max_px=max_px,
        tissue=tissue,
        coords=coords,
        level0_patch=level0_patch,
        overlay=True,
        overwrite=overwrite,
        overlay_name=overlay_name,
    )
    metadata["grid"] = key
    sink.update_thumbnail(metadata, update_root=(key == primary_key))
    return metadata


def _stored_grid_summary(
    out_path: str, group_cfgs: list[RunConfig]
) -> dict[str, list[str]]:
    """Resolve requested grid hashes to their actual (possibly suffixed) labels."""

    try:
        import zarr

        from raw2features.core.store import GRIDS, grid_keys

        root = zarr.open_group(out_path, mode="r", use_consolidated=False)
        keys = grid_keys(root)
        stored = {
            key: dict(root[GRIDS][key].attrs.get("raw2features", {})).get("grid_hash")
            for key in keys
        }
        result: dict[str, list[str]] = {}
        used: set[str] = set()
        for cfg in group_cfgs:
            candidates = (cfg.grid_hash(), *cfg.compatible_legacy_grid_hashes())
            match = None
            for candidate in dict.fromkeys(candidates):
                hits = [
                    key for key in keys if key not in used and stored[key] == candidate
                ]
                if len(hits) == 1:
                    match = hits[0]
                    break
                if len(hits) > 1:
                    return {}
            if (
                match is None
                and len(group_cfgs) == 1
                and len(keys) == 1
                and stored[keys[0]] is None
            ):
                match = keys[0]
            if match is None:
                return {}
            used.add(match)
            result[match] = list(cfg.models)
        return result
    except Exception:  # noqa: BLE001 - summaries fall back to their planned labels
        return {}


def _slide_encoders_for(names: list[str], available: list[str]) -> list[str]:
    """Slide encoders whose patch encoder is among *available* (this grid's models).

    A specific-patch-encoder slide model (e.g. titan -> conch_v1_5) runs only on the
    grid that holds its patch encoder; on other grids it is skipped. A model-agnostic
    pooling baseline (``patch_encoder == "any"``) runs on any grid with at least one
    model. Unknown names are dropped (the registry build errors elsewhere).
    """
    from raw2features.slide_embedders.model_registry import get_slide_spec

    out: list[str] = []
    for name in names:
        try:
            required = get_slide_spec(name).patch_encoder
        except KeyError:
            continue
        if (required == "any" and available) or required in available:
            out.append(name)
    return out


def _run_qc(
    qc_tools,
    reader,
    sink,
    coords,
    level0_patch,
    device,
    stain_norm=None,
    artifact_mpp="1.5",
) -> None:
    """Run requested QC producers on the active grid -> write ``qc/<tool>/`` scores.

    A producer turns the WSI into a per-pixel raster and projects it to per-patch
    coverage fractions (:func:`raw2features.core.qc.patch_qc_scores`), written via
    ``sink.write_qc``. ``stain_norm`` (e.g. ``"macenko"``) normalizes the producer input
    first; ``artifact_mpp`` picks GrandQC's artifact model scale. Producers are
    optional/external; an unknown name warns and skips.
    """
    import warnings

    from raw2features.core.device import resolve_device

    dev = resolve_device(device)
    for tool in qc_tools:
        if tool == "grandqc":
            from raw2features.qc.grandqc import QC_CLASSES, GrandQC

            gq = GrandQC(device=dev, stain_norm=stain_norm, artifact_mpp=artifact_mpp)
            scores, classes = gq.qc_for_grid(reader, coords, level0_patch)
            legend = {
                "version": "0.5",
                "properties": [
                    {"label-value": v, "name": n} for v, n in QC_CLASSES.items()
                ],
            }
            sink.write_qc(
                "grandqc", scores, classes, legend=legend, provenance=gq.provenance()
            )
        else:
            warnings.warn(f"unknown --qc producer {tool!r}; skipping", stacklevel=2)


def _run_slide_encoders(
    sink,
    slide_encoder_names: list[str],
    device: str,
    available_patch_models: list[str],
) -> dict[str, str]:
    """Run slide-level encoders on patch features already written to *sink*.

    Reads ``features/<patch_encoder>`` directly from the open sink group -
    no WSI access, no re-embedding. Returns a mapping of slide model name
    to the slide embedding array path (``slide/<model>``).
    """
    from raw2features.slide_embedders.encoding import (
        encode_slide_embedding,
        resolve_slide_patch_model,
        slide_embedding_is_complete,
    )

    results: dict[str, str] = {}

    for slide_model_name in slide_encoder_names:
        patch_model = resolve_slide_patch_model(
            sink._group,
            slide_model_name,
            available_patch_models=available_patch_models,
        )
        if slide_embedding_is_complete(
            sink._group,
            slide_model_name,
            patch_model=patch_model,
            device=device,
        ):
            results[slide_model_name] = f"slide/{slide_model_name}"
            continue

        encoding = encode_slide_embedding(
            sink._group,
            slide_model_name,
            device,
            patch_model=patch_model,
            available_patch_models=available_patch_models,
        )
        if encoding is None:
            continue
        sink.write_slide_embedding(
            slide_model_name,
            encoding.vector,
            encoding.provenance,
        )
        results[slide_model_name] = f"slide/{slide_model_name}"

    return results


def _run_slide_encoders_from_store(
    out_path: str,
    slide_encoder_names: list[str],
    device: str,
) -> dict[str, str]:
    """Produce missing slide outputs by discovering compatible grids in a store.

    This is the high-level inline fallback for patch features written by an earlier
    request. Grid and patch-model selection is shared with standalone ``slide-embed``;
    no WSI is reopened and complete outputs are left untouched.
    """
    import zarr

    from raw2features.slide_embedders.encoding import (
        encode_slide_embedding,
        resolve_slide_grid,
        slide_embedding_is_complete,
        write_slide_embedding,
    )

    root = zarr.open_group(out_path, mode="r+", use_consolidated=False)
    results: dict[str, str] = {}
    wrote = False
    for slide_model_name in slide_encoder_names:
        try:
            selected_grid, group, patch_model = resolve_slide_grid(
                root,
                slide_model_name,
            )
        except (KeyError, ValueError):
            # Coverage is checked by embed_slide after every requested grid and this
            # fallback have been tried. Selection failure here means only that no
            # compatible pre-existing grid was found; model-load/encode errors below
            # still propagate rather than being mistaken for a missing grid.
            continue
        if slide_embedding_is_complete(
            group,
            slide_model_name,
            patch_model=patch_model,
            device=device,
        ):
            results[slide_model_name] = (
                f"grids/{selected_grid}/slide/{slide_model_name}"
            )
            continue
        encoding = encode_slide_embedding(
            group,
            slide_model_name,
            device,
            patch_model=patch_model,
        )
        if encoding is None:
            continue
        write_slide_embedding(
            group,
            slide_model_name,
            encoding.vector,
            encoding.provenance,
        )
        wrote = True
        results[slide_model_name] = f"grids/{selected_grid}/slide/{slide_model_name}"

    if wrote:
        try:
            zarr.consolidate_metadata(root.store)
        except Exception:  # noqa: BLE001 - consolidation is rebuildable metadata
            pass
    return results


def _models_header(
    embedders: list,
    model_contracts: dict[str, dict] | None = None,
) -> dict:
    """The per-model provenance block stored in the zarr header's ``models`` key."""
    contracts = model_contracts or {
        e.name: {
            "embedding_dim": e.spec.embedding_dim,
            "output_fingerprint": patch_output_fingerprint(
                e.spec, _amp_label(getattr(e, "_dtype", None))
            ),
        }
        for e in embedders
    }
    return {
        e.name: {
            "source": e.spec.source,
            "embedding_dim": e.spec.embedding_dim,
            "input_size": e.spec.input_size,
            "pooling": e.spec.pooling,
            "mean": list(e.spec.mean),
            "std": list(e.spec.std),
            "interpolation": e.spec.interpolation,
            "transform_source_url": e.spec.transform_source_url,
            "doi": e.spec.doi,
            "license": e.spec.license,
            "gated": e.spec.gated,
            "weights_sha256": e.spec.weights_sha256,
            "weights_revision": e.spec.weights_revision,
            "weights_filename": e.spec.weights_filename,
            "experimental": e.spec.experimental,
            "output_fingerprint": contracts[e.name]["output_fingerprint"],
            # The precision the embedding was actually computed at (provenance).
            "inference_amp": _amp_label(getattr(e, "_dtype", None)),
        }
        for e in embedders
    }


def _build_header(
    reader,
    grid,
    seg_meta,
    embedders,
    slide_id,
    n,
    thumbnail,
    grid_hash,
    prov,
    panel_meta=None,
    model_contracts=None,
) -> dict:
    header = {
        "schema_version": SCHEMA_VERSION,
        "grid_hash": grid_hash,
        "provenance": prov,
        "thumbnail": thumbnail,
        "source": {
            "uri": source_uri(reader.path),
            "ngff_version": getattr(reader, "ngff_version", None),
            "reader": reader.name,
            "slide_id": slide_id,
            "mpp_level0": reader.mpp,
            "level_dimensions": [[d.width, d.height] for d in reader.level_dimensions],
            "level_downsamples": reader.level_downsamples(),
            # Self-description of the source coordinate frame (optional VALUES, not an
            # RFC-5 transform object): the axis order, per-axis units + level-0 scale
            # in µm, and the source level-0 translation/origin (µm) when it carries one.
            # coords stay level-0 px (origin top-left); these let a consumer re-express
            # them in the source's physical frame. See docs/SPEC.md "Coordinates".
            "axes": list(getattr(reader, "axes", ()) or []),
            "axis_units": getattr(reader, "axis_units", {}),
            "scale_um": getattr(reader, "scale_um", {}),
            "level0_translation_um": getattr(reader, "level0_translation_um", None),
        },
        "patching": {
            "target_mpp": grid.target_mpp,
            "achieved_mpp": grid.achieved_mpp,
            "patch_px": grid.patch_px,
            "step_out_px": grid.step_out_px,
            "read_level": grid.read_level,
            "read_px": grid.read_px,
            "resample": grid.resample,
            "needs_resample": grid.needs_resample,
            "level0_patch": grid.level0_patch,
            "level0_step": grid.level0_step,
            "n_patches": n,
            "grid_shape": [grid.n_rows, grid.n_cols],
            "coords_convention": "level0_xy",
        },
        "segmentation": seg_meta,
        "models": _models_header(embedders, model_contracts),
    }
    # Multiplex marker-panel resolution (per model): which of the slide's channels
    # matched the model's marker vocabulary and which were dropped. Absent for
    # brightfield runs. Part of the reproducibility record for a multiplex embedding.
    if panel_meta:
        header["panel"] = panel_meta
    return header
