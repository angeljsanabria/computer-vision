"""Profiler de latencia por etapa para main_track (Fase 0 mejora_fps)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

_NS_TO_MS = 1.0e-6

_STAGE_ORDER = (
    "capture",
    "mog2",
    "rf",
    "bt",
    "embed",
    "match",
    "mesh",
    "nozzle",
    "disp",
    "total",
)

_STAGE_LABELS = {
    "capture": "capture",
    "mog2": "mog2",
    "rf": "rf",
    "bt": "bt",
    "embed": "embed",
    "match": "match",
    "mesh": "mesh",
    "nozzle": "nozzle",
    "disp": "disp",
    "total": "total",
}


@dataclass
class _StageWindow:
    total_ns: int = 0
    max_ns: int = 0
    n: int = 0

    def add(self, dt_ns: int) -> None:
        self.total_ns += dt_ns
        if dt_ns > self.max_ns:
            self.max_ns = dt_ns
        self.n += 1

    def avg_ms(self) -> float:
        return (self.total_ns / self.n * _NS_TO_MS) if self.n else 0.0

    def max_ms(self) -> float:
        return self.max_ns * _NS_TO_MS


class PipelineProfiler:
    """Acumula tiempos; emite log INFO cada log_every_n frames procesados."""

    def __init__(self, *, enabled: bool, log_every_n: int = 30) -> None:
        self.enabled = enabled
        self.log_every_n = max(1, int(log_every_n))
        self._frame_count = 0
        self._stages: dict[str, _StageWindow] = {
            key: _StageWindow() for key in _STAGE_ORDER
        }

    def mark(self) -> int:
        """Marca de tiempo; retorna 0 si el profiler esta desactivado."""
        return time.monotonic_ns() if self.enabled else 0

    def lap(self, stage: str, t0: int) -> None:
        """Registra dt desde t0=mark() hasta ahora."""
        if not self.enabled or t0 <= 0:
            return
        self.add(stage, time.monotonic_ns() - t0)

    def begin_frame(self) -> None:
        if not self.enabled:
            return
        self._frame_count += 1

    def add(self, stage: str, dt_ns: int) -> None:
        if not self.enabled or dt_ns < 0:
            return
        acc = self._stages.get(stage)
        if acc is None:
            acc = _StageWindow()
            self._stages[stage] = acc
        acc.add(dt_ns)

    def maybe_log(self) -> None:
        if not self.enabled or self._frame_count == 0:
            return
        if self._frame_count % self.log_every_n != 0:
            return
        parts: list[str] = []
        for key in _STAGE_ORDER:
            acc = self._stages[key]
            if acc.n == 0:
                continue
            label = _STAGE_LABELS[key]
            parts.append(f"{label}={acc.avg_ms():.1f}/{acc.max_ms():.1f}")
        total_acc = self._stages["total"]
        fps = (
            1000.0 / total_acc.avg_ms()
            if total_acc.n and total_acc.avg_ms() > 0
            else 0.0
        )
        logging.info(
            "[PROF] n=%d %s ms(avg/max) fps=%.1f",
            self._frame_count,
            " ".join(parts),
            fps,
        )
        for acc in self._stages.values():
            acc.total_ns = 0
            acc.max_ns = 0
            acc.n = 0
