"""Content identities for persisted patch- and slide-model outputs.

``grid_hash`` deliberately identifies patch geometry only.  These fingerprints
identify the model contract that produced one concrete feature array, so changing
weights, preprocessing, precision, or loader construction invalidates that model
without renaming the grid.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from urllib.parse import unquote, urlsplit

from raw2features.core.uris import (
    is_qualified_uri,
    redact_uri_credentials,
    source_uri,
)

from .base import ModelSpec

PATCH_LOADER_CONTRACT_VERSION = 1
SLIDE_LOADER_CONTRACT_VERSION = 1
OUTPUT_FINGERPRINT_VERSION = 1
OUTPUT_FINGERPRINT_ALGORITHM = "sha256"

# Construction inputs in SEAL's pinned fork that affect the image encoder.  The
# loader consumes this object directly as well as fingerprinting it, preventing a
# CWD-local conf/config.yaml or conf/user/<name>.yaml from silently changing output.
SEAL_CONSTRUCTOR_CONTRACT: dict[str, Any] = {
    "model": "panST",
    "partial_blocks": 1,
    "use_adapter": False,
    "adapter_bottleneck": 256,
    "use_lora": True,
    "lora": {
        "r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.15,
        "use_rslora": False,
    },
    "lambda_recon_img": 0.0,
    "organ_token": False,
    "projection_head": None,
    "dec_batch_norm": False,
    "dec_dropout": 0.0,
    # Required by PatchRecEncoder even though its reconstruction decoder is disabled.
    "n_train_genes": 2000,
}

SEAL_FORK_REVISION = "5334490645e8410e7d8ef6978cebc4fd98f9cf9a"
CONCH_PACKAGE_REVISION = "141cc09c7d4ff33d8eda562bd75169b457f71a62"
KRONOS_PACKAGE_REVISION = "48979362386c8440c934954be3d88ccfa74d6f36"
MUSK_PACKAGE_REVISION = "714b666969c1911e5efe70d991140a21030f4ef3"
GIGAPATH_PACKAGE_REVISION = "3505f87e197d167522be491bb3f18fb5a08ca584"
MADELEINE_PACKAGE_REVISION = "419287dc60a57296d959840b893481019c4f0d21"
BIOMEDCLIP_TEXT_REPO = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
BIOMEDCLIP_TEXT_REVISION = "d673b8835373c6fa116d6d8006b33d48734e305d"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _credential_free(value: Any) -> Any:
    """Recursively remove URI credentials before hashing or persistence."""

    if isinstance(value, Mapping):
        return {str(key): _credential_free(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_credential_free(item) for item in value]
    if isinstance(value, str):
        if is_qualified_uri(value):
            try:
                return source_uri(value)
            except Exception:  # noqa: BLE001 - malformed URIs must still fail closed
                return redact_uri_credentials(value)
        return redact_uri_credentials(value)
    return value


def make_output_fingerprint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a self-validating, JSON-safe fingerprint record for *payload*."""

    # Round-trip once so tuples and other JSON-compatible containers have the same
    # representation before and after persistence through zarr attributes. URI
    # credentials are removed first, both to keep the record safe to persist and to
    # keep rotating signed URLs from changing model-output identity.
    normalised = json.loads(_canonical_json(_credential_free(dict(payload))))
    digest = hashlib.sha256(_canonical_json(normalised).encode("utf-8")).hexdigest()
    return {
        "version": OUTPUT_FINGERPRINT_VERSION,
        "algorithm": OUTPUT_FINGERPRINT_ALGORITHM,
        "digest": digest,
        "payload": normalised,
    }


def valid_output_fingerprint(value: Any) -> bool:
    """Whether *value* is a well-formed record whose digest matches its payload."""

    if not isinstance(value, Mapping):
        return False
    if value.get("version") != OUTPUT_FINGERPRINT_VERSION:
        return False
    if value.get("algorithm") != OUTPUT_FINGERPRINT_ALGORITHM:
        return False
    digest = value.get("digest")
    payload = value.get("payload")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or not isinstance(payload, Mapping)
    ):
        return False
    try:
        normalised = json.loads(_canonical_json(dict(payload)))
    except (TypeError, ValueError):
        return False
    # Validation hashes the exact persisted payload. Calling make_output_fingerprint
    # here would sanitize it first and could accidentally bless a credential-bearing
    # payload whose digest was calculated from the redacted form.
    if normalised != _credential_free(normalised):
        return False
    expected = hashlib.sha256(_canonical_json(normalised).encode("utf-8")).hexdigest()
    return digest == expected


def output_fingerprints_equal(left: Any, right: Any) -> bool:
    """Compare two complete records, rejecting malformed records on either side."""

    return (
        valid_output_fingerprint(left)
        and valid_output_fingerprint(right)
        and dict(left) == dict(right)
    )


def fingerprint_digest(value: Any) -> str | None:
    """Return a validated record's digest, else ``None``."""

    return str(value["digest"]) if valid_output_fingerprint(value) else None


def _hf_repo(source: str) -> str:
    return source.removeprefix("hf-hub:").removeprefix("hf_hub:")


def _effective_patch_checkpoint(spec: ModelSpec) -> dict[str, Any]:
    if spec.checkpoint:
        checkpoint = deepcopy(spec.checkpoint)
        url = checkpoint.get("url")
        filename = spec.weights_filename or checkpoint.get("filename")
        if not filename and isinstance(url, str):
            filename = unquote(posixpath.basename(urlsplit(url).path)) or None
        return {
            "repo": checkpoint.get("repo"),
            "filename": filename,
            "url": url,
            "mechanism": "explicit_checkpoint",
        }

    repo = _hf_repo(spec.source)
    fixed: dict[str, tuple[str, str]] = {
        "conch": (repo, "pinned_local_file"),
        "conch_v1_5": (
            "MahmoodLab/TITAN",
            "transformers_snapshot_return_conch",
        ),
        "kronos": (repo, "pinned_local_file"),
        "musk": (repo, "pinned_local_file"),
        "open_clip": (repo, "pinned_local_snapshot"),
        "seal": (
            "MahmoodLab/SEAL",
            "pinned_adapter_file",
        ),
    }
    if spec.family in fixed:
        resolved_repo, mechanism = fixed[spec.family]
        return {
            "repo": resolved_repo,
            "filename": spec.weights_filename,
            "mechanism": mechanism,
        }
    if spec.family == "torchvision":
        return {
            "repo": "torchvision",
            "filename": spec.weights_filename or spec.source,
            "mechanism": "weights_enum",
        }
    return {
        "repo": repo,
        "filename": spec.weights_filename,
        "mechanism": "loader_managed_snapshot",
    }


def _seal_composite(spec: ModelSpec) -> dict[str, Any]:
    bases = {
        "conch": {
            "repo": "MahmoodLab/conch",
            "architecture": "conch_ViT-B-16",
            "loader_package_revision": CONCH_PACKAGE_REVISION,
        },
        "univ2": {
            "repo": "MahmoodLab/UNI2-h",
            "architecture": "vit_huge_patch14_224",
            "constructor": {
                "img_size": 224,
                "patch_size": 14,
                "depth": 24,
                "num_heads": 24,
                "init_values": 1e-5,
                "embed_dim": 1536,
                "mlp_ratio": 5.33334,
                "num_classes": 0,
                "no_embed_class": True,
                "mlp_layer": "timm.layers.SwiGLUPacked",
                "act_layer": "torch.nn.SiLU",
                "reg_tokens": 8,
                "dynamic_img_size": True,
            },
        },
    }
    base = deepcopy(bases.get(spec.source, {"repo": spec.source}))
    base.update(
        {
            "revision": None,
            "sha256": None,
            "integrity": "upstream_factory_unpinned",
        }
    )
    return {
        "experimental": True,
        "pinning_policy": "adapter_verified_base_unpinned",
        "adapter": {
            "repo": "MahmoodLab/SEAL",
            "filename": spec.weights_filename,
            "revision": spec.weights_revision,
            "sha256": spec.weights_sha256,
        },
        "base": base,
        "construction_dependencies": {
            "seal_fork_revision": SEAL_FORK_REVISION,
        },
    }


def _patch_constructor(spec: ModelSpec) -> dict[str, Any]:
    constructor: dict[str, Any] = {
        "timm_kwargs": deepcopy(spec.timm_kwargs),
        "checkpoint_load": deepcopy(spec.checkpoint),
    }
    family_contracts: dict[str, dict[str, Any]] = {
        "timm": {"entrypoint": "timm.create_model"},
        "torchvision": {"classifier": "identity"},
        "transformers": {
            "entrypoint": "transformers.AutoModel.from_pretrained",
            "trust_remote_code": bool(spec.timm_kwargs.get("trust_remote_code", False)),
        },
        "clip_hf": {
            "entrypoint": "transformers.AutoModel.from_pretrained",
            "trust_remote_code": True,
            "image_method": "encode_image_or_get_image_features",
        },
        "conch": {
            "architecture": "conch_ViT-B-16",
            "proj_contrast": False,
            "normalize": False,
            "construction_package_revision": CONCH_PACKAGE_REVISION,
        },
        "conch_v1_5": {"parent_model": "MahmoodLab/TITAN", "selector": "return_conch"},
        "kronos": {
            "model_type": "vits16",
            "token_overlap": False,
            "sdpa": bool(spec.timm_kwargs.get("sdpa", True)),
            "marker_metadata": "marker_metadata.csv",
            "construction_package_revision": KRONOS_PACKAGE_REVISION,
        },
        "musk": {
            "architecture": "musk_large_patch16_384",
            "with_head": False,
            "out_norm": True,
            "ms_aug": False,
            "construction_package_revision": MUSK_PACKAGE_REVISION,
        },
        "open_clip": {"entrypoint": "open_clip.create_model_from_pretrained"},
        "seal": {
            "entrypoint": "seal.models.load_model.ModelMixin.get_img_model",
            "parameters": deepcopy(SEAL_CONSTRUCTOR_CONTRACT),
        },
    }
    constructor.update(family_contracts.get(spec.family, {"entrypoint": spec.family}))
    if spec.name == "biomedclip":
        constructor["nested_text_config"] = {
            "repo": BIOMEDCLIP_TEXT_REPO,
            "revision": BIOMEDCLIP_TEXT_REVISION,
        }
    return constructor


def patch_output_fingerprint(spec: ModelSpec, resolved_amp: str) -> dict[str, Any]:
    """Fingerprint the complete persisted-output contract for one patch model."""

    payload: dict[str, Any] = {
        "kind": "patch_features",
        "model": spec.name,
        "loader": {
            "family": spec.family,
            "contract_version": PATCH_LOADER_CONTRACT_VERSION,
            "source": spec.source,
            "constructor": _patch_constructor(spec),
        },
        "checkpoint": {
            "effective": _effective_patch_checkpoint(spec),
            "weights_revision": spec.weights_revision,
            "weights_sha256": spec.weights_sha256,
        },
        "preprocessing": {
            "input_size": int(spec.input_size),
            "mean": list(spec.mean),
            "std": list(spec.std),
            "interpolation": spec.interpolation,
            "transform_source": spec.transform_source,
        },
        "output": {
            "pooling": spec.pooling,
            "embedding_dim": int(spec.embedding_dim),
            "reg_tokens": int(spec.reg_tokens),
            "modality": spec.modality,
            "resolved_amp": resolved_amp,
        },
        "composite": _seal_composite(spec) if spec.family == "seal" else None,
    }
    return make_output_fingerprint(payload)


def resolved_patch_amp(spec: ModelSpec, requested_amp: str, device: str) -> str:
    """Resolve requested/card AMP to the precision the forward path really uses."""

    selected = spec.inference_amp if requested_amp == "auto" else requested_amp
    # Embedder._forward_ctx enables fp16/bf16 autocast only on CUDA. Passing either
    # dtype on CPU/MPS still executes the model in fp32, so provenance must say fp32.
    device_type = str(device).split(":", 1)[0]
    if device_type in {"cpu", "mps"} and selected in {"bf16", "fp16"}:
        return "fp32"
    return selected


def expected_patch_outputs(
    models: list[str],
    requested_amp: str,
    device: str = "cuda",
    *,
    specs: Mapping[str, ModelSpec] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return current ``embedding_dim`` + fingerprint contracts by model name."""

    from .model_registry import get_spec

    overrides = specs or {}
    contracts: dict[str, dict[str, Any]] = {}
    for name in models:
        spec = overrides.get(name) or get_spec(name)
        resolved_amp = resolved_patch_amp(spec, requested_amp, device)
        contracts[name] = {
            "embedding_dim": int(spec.embedding_dim),
            "output_fingerprint": patch_output_fingerprint(spec, resolved_amp),
        }
    return contracts


def _effective_slide_checkpoint(spec) -> dict[str, Any]:
    repo = _hf_repo(spec.source)
    if spec.family == "gigapath_slide":
        return {
            "repo": repo,
            "filename": spec.weights_filename,
            "mechanism": "pinned_local_file",
        }
    if spec.family == "madeleine":
        return {
            "repo": repo,
            "filename": spec.weights_filename,
            "auxiliary_files": ["model_config.json"],
            "mechanism": "pinned_local_snapshot",
        }
    if str(spec.source).startswith("gdrive:"):
        return {
            "repo": spec.source,
            "filename": spec.weights_filename,
            "mechanism": "sha256_verified_file",
        }
    if spec.family == "pool":
        return {"repo": "builtin", "filename": None, "mechanism": "weight_free"}
    return {
        "repo": repo,
        "filename": spec.weights_filename,
        "mechanism": "loader_managed_snapshot",
    }


def _slide_constructor(spec) -> dict[str, Any]:
    contracts: dict[str, dict[str, Any]] = {
        "pool": {
            "operation": spec.name,
            "input_cast": "float32",
            "reduction_axis": 0,
            "meanmax_concat_order": ["mean", "max"] if spec.name == "meanmax" else None,
        },
        "titan": {
            "entrypoint": "transformers.AutoModel.from_pretrained",
            "trust_remote_code": True,
            "forward": "encode_slide_from_patch_features",
            "features_dtype": "float32",
            "coords_dtype": "int64",
            "patch_size_lv0_cast": "int",
            "batched": False,
            "uses_coords": True,
            "uses_patch_size_lv0": True,
        },
        "prism": {
            "entrypoint": "transformers.AutoModel.from_pretrained",
            "trust_remote_code": True,
            "forward": "slide_representations",
            "output_key": "image_embedding",
            "features_dtype": "float32",
            "batched": True,
        },
        "feather": {
            "entrypoint": "transformers.AutoModel.from_pretrained",
            "input": "pinned_local_snapshot",
            "trust_remote_code": True,
            "forward": "forward_features",
            "result_index": 0,
            "features_dtype": "float32",
            "batched": True,
        },
        "madeleine": {
            "entrypoint": "madeleine.models.Model.create_model",
            "construction_package_revision": MADELEINE_PACKAGE_REVISION,
            "config": {
                "repo": _hf_repo(spec.source),
                "filename": "model_config.json",
                "revision": spec.weights_revision,
            },
            "arguments": ["parsed_model_config", "device", "checkpoint_path"],
            "forward": "encode_he",
            "features_dtype": "float32",
            "batched": True,
        },
        "gigapath_slide": {
            "entrypoint": "gigapath.slide_encoder.create_model",
            "architecture": "gigapath_slide_enc12l768d",
            "patch_dim": 1536,
            "global_pool": True,
            "tile_size_source": "patching.level0_patch",
            "coords_dtype": "float32",
            "features_dtype": "float32",
            "all_layer_embed": True,
            "result_index": -1,
            "construction_package_revision": GIGAPATH_PACKAGE_REVISION,
        },
        "chief": {
            "implementation": "raw2features",
            "pooling": "gated_attention",
            "input_dim": 768,
            "projection_dims": [768, 512, 256],
            "projection_activation": "relu",
            "projection_dropout": 0.25,
            "attention": {
                "tanh_dim": 256,
                "sigmoid_dim": 256,
                "dropout": 0.25,
                "projection": [256, 1],
                "normalization": "softmax_over_patches",
            },
            "output": "weighted_sum_of_original_input",
            "checkpoint": {
                "include_prefix": "attention_net.",
                "drop": ["organ_embedding"],
                "strict": False,
                "reject_missing_keys": True,
            },
        },
        "tangle": {
            "implementation": "raw2features",
            "pooling": "multi_head_abmil",
            "patch_dim": 1024,
            "hidden_dim": 512,
            "heads": 4,
            "pre_mlp": {
                "dims": [1024, 512, 512, 2048],
                "normalization": "layer_norm",
                "activation": "gelu",
                "dropout": 0.1,
            },
            "attention": {
                "type": "gated",
                "dropout": 0.25,
                "reshape": "heads_fastest",
            },
            "output_projection": [2048, 512],
            "checkpoint": {
                "strip_prefixes": ["module.", "wsi_embedder."],
            },
        },
    }
    return deepcopy(contracts.get(spec.family, {"entrypoint": spec.family}))


def slide_output_dim(spec, patch_dim: int) -> int:
    """Resolve fixed or pooling-dependent slide output width."""

    if int(spec.embedding_dim) > 0:
        return int(spec.embedding_dim)
    return int(patch_dim) * (2 if spec.name == "meanmax" else 1)


def resolved_slide_amp(spec, device: str) -> str:
    """Return the effective slide-forward precision for the resolved device."""

    # These loaders follow their model-card examples with fp16 CUDA autocast and
    # deliberately run fp32 on CPU/MPS. Other slide encoders currently run fp32.
    if str(device).startswith("cuda") and spec.family in {
        "gigapath_slide",
        "prism",
        "titan",
    }:
        return "fp16"
    return "fp32"


def slide_output_fingerprint(
    spec,
    *,
    patch_model: str,
    patch_output_fingerprint: Mapping[str, Any],
    patch_dim: int,
    resolved_amp: str,
) -> dict[str, Any]:
    """Fingerprint a slide model together with the concrete patch output it uses."""

    patch_digest = fingerprint_digest(patch_output_fingerprint)
    if patch_digest is None:
        raise ValueError(
            f"Patch model {patch_model!r} has no valid output fingerprint; rerun "
            "raw2features embed for that patch model before slide encoding."
        )
    output_dim = slide_output_dim(spec, patch_dim)
    payload = {
        "kind": "slide_embedding",
        "model": spec.name,
        "loader": {
            "family": spec.family,
            "contract_version": SLIDE_LOADER_CONTRACT_VERSION,
            "source": spec.source,
            "constructor": _slide_constructor(spec),
        },
        "checkpoint": {
            "effective": _effective_slide_checkpoint(spec),
            "weights_revision": spec.weights_revision,
            "weights_sha256": spec.weights_sha256,
        },
        "input": {
            "patch_model": patch_model,
            "patch_dim": int(patch_dim),
            "patch_output_fingerprint": patch_digest,
        },
        "output": {"embedding_dim": output_dim, "resolved_amp": resolved_amp},
    }
    return make_output_fingerprint(payload)
