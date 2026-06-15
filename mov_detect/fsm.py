"""Maquina de estados movimiento (MOG2) + fase facial (RetinaFace)."""
from __future__ import annotations

from mov_detect.types import FlowState, FsmConfig, FsmTickResult


class MotionFaceFsm:
    """
    FSM pura: sin OpenCV ni modelos.

    Uso por frame (desde main):
      1. ``tick_motion(hay_mov, now)`` tras MOG2
      2. Si ``run_face_detector``: inferir RetinaFace
      3. ``tick_face(hay_cara, now)`` solo si se inferio
    """

    _FACE_INFER_STATES = frozenset(
        {
            FlowState.MOV_DETECTED,
            FlowState.MOV_OUT,
            FlowState.FACE_PROCESSED,
            FlowState.FACE_OUT,
        }
    )

    def __init__(self, config: FsmConfig | None = None) -> None:
        cfg = config or FsmConfig()
        self._t_mov = float(cfg.timeout_mov_s)
        self._t_face = float(cfg.timeout_face_s)
        self.state = FlowState.IDLE
        self._t_ultimo_mov: float | None = None
        self._t_ultima_cara: float | None = None

    def tick_motion(self, hay_mov: bool, now: float) -> FsmTickResult:
        """Transiciones y timeouts por sensor MOG2."""
        events: list[str] = []

        if hay_mov:
            self._t_ultimo_mov = now

        sin_mov_mog2 = (
            self._t_ultimo_mov is not None
            and (now - self._t_ultimo_mov) >= self._t_mov
        )

        if sin_mov_mog2 and self.state in (
            FlowState.MOV_DETECTED,
            FlowState.MOV_OUT,
        ):
            s_antes = self.state
            self.state = FlowState.IDLE
            self._t_ultima_cara = None
            events.append(
                "[FSM] {} -> IDLE (timeout {:.1f} s sin movimiento MOG2)".format(
                    s_antes.value, self._t_mov
                )
            )

        s_antes_mog2 = self.state
        if self.state == FlowState.IDLE:
            if hay_mov:
                self.state = FlowState.MOV_DETECTED
        elif self.state == FlowState.MOV_DETECTED:
            if not hay_mov:
                self.state = FlowState.MOV_OUT
        elif self.state == FlowState.MOV_OUT:
            if hay_mov:
                self.state = FlowState.MOV_DETECTED

        if self.state != s_antes_mog2:
            events.append(
                "[FSM] {} -> {} (MOG2)".format(s_antes_mog2.value, self.state.value)
            )

        return self._snapshot(events)

    def tick_face(self, hay_cara: bool, now: float) -> FsmTickResult:
        """Transiciones tras RetinaFace (llamar solo si se inferio este frame)."""
        events: list[str] = []
        s_antes = self.state

        if hay_cara:
            self._t_ultima_cara = now
            self._t_ultimo_mov = now
            self.state = FlowState.FACE_PROCESSED
        elif self.state == FlowState.FACE_PROCESSED:
            self.state = FlowState.FACE_OUT
        elif self.state == FlowState.FACE_OUT:
            if (
                self._t_ultima_cara is not None
                and (now - self._t_ultima_cara) >= self._t_face
            ):
                self.state = FlowState.FACE_PROCESSED_TIMEOUT

        if self.state != s_antes:
            events.append(
                "[FSM] {} -> {} (RetinaFace)".format(s_antes.value, self.state.value)
            )

        if self.state == FlowState.FACE_PROCESSED_TIMEOUT:
            events.append(
                "[FSM] {} -> IDLE (sin cara durante {:.1f} s)".format(
                    FlowState.FACE_PROCESSED_TIMEOUT.value, self._t_face
                )
            )
            self.state = FlowState.IDLE
            self._t_ultima_cara = None

        return self._snapshot(events)

    def _snapshot(self, events: list[str]) -> FsmTickResult:
        return FsmTickResult(
            state=self.state,
            run_face_detector=self.state in self._FACE_INFER_STATES,
            run_embedding=self.state == FlowState.FACE_PROCESSED,
            transitions=tuple(events),
        )
