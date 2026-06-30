"""Lightweight per-stage profiler for the benchmark harness.

The pipeline takes an optional profiler. In production it is a ``NullProfiler``
(every method a no-op), so instrumentation costs nothing when benchmarking is off.
A real ``Profiler`` accumulates wall-time per named stage plus the volume of pixel
data decoded, which is enough to attribute time to read / transform / GPU / write
and answer "where is the bottleneck" without guessing.
"""

from __future__ import annotations

import contextlib
import threading
import time


class NullProfiler:
    """No-op profiler: the production default."""

    @contextlib.contextmanager
    def stage(self, name: str):
        yield

    def add_bytes(self, n: int) -> None:
        pass


class Profiler:
    """Accumulates wall-time per named stage (repeats sum) plus bytes decoded.

    ``cuda=True`` synchronises around the ``"gpu"`` stage so its timing reflects
    actual kernel completion rather than just queue submission.

    Thread-safe: the prefetch hot path times the ``"read"`` stage on a background
    worker while the main thread times ``transform``/``gpu``/``write`` concurrently,
    so the accumulators are guarded by a lock. With prefetch, stage times overlap
    and may sum to more than the wall clock -- that overlap is the speedup.
    """

    def __init__(self, *, cuda: bool = False) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.bytes_read: int = 0
        self._cuda = cuda
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def stage(self, name: str):
        sync = self._cuda and name == "gpu"
        if sync:
            import torch

            torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                import torch

                torch.cuda.synchronize()
            dt = time.perf_counter() - t
            with self._lock:
                self.totals[name] = self.totals.get(name, 0.0) + dt
                self.counts[name] = self.counts.get(name, 0) + 1

    def add_bytes(self, n: int) -> None:
        with self._lock:
            self.bytes_read += n

    def summary(self, *, n_patches: int, wall_s: float) -> dict:
        """A JSON-able breakdown: per-stage seconds + %, throughput, what's left."""
        staged = sum(self.totals.values())
        stages = {
            name: {
                "seconds": round(t, 4),
                "pct_wall": round(100 * t / wall_s, 1) if wall_s else 0.0,
                "calls": self.counts.get(name, 0),
            }
            for name, t in sorted(self.totals.items(), key=lambda kv: -kv[1])
        }
        read_s = self.totals.get("read", 0.0)
        return {
            "n_patches": n_patches,
            "wall_s": round(wall_s, 3),
            "staged_s": round(staged, 3),
            "unaccounted_s": round(wall_s - staged, 3),
            "patches_per_s": round(n_patches / wall_s, 1) if wall_s else 0.0,
            "decoded_MB": round(self.bytes_read / 1e6, 1),
            "decode_MB_per_s": (
                round(self.bytes_read / 1e6 / read_s, 1) if read_s else 0.0
            ),
            "stages": stages,
        }


_NULL = NullProfiler()


def null_profiler() -> NullProfiler:
    """Shared no-op profiler instance (the production default)."""
    return _NULL
