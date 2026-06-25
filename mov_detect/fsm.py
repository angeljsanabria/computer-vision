"""Maquina de estados movimiento (MOG2) + fase facial (RetinaFace)."""
from __future__ import annotations

from mov_detect.types import FlowState, FsmConfig, FsmTickResult


class MotionFaceFsm:
    """
    FSM pura: sin OpenCV ni modelos.

    FACE_RECOGNIZED (sesion de identidad confirmada):
      - Entrada: unico MATCH en FACE_PROCESSED (notify_embed_match).
      - Salida: timer de identidad vencido -> FACE_PROCESSED (refresca cara/mov).
      - No reacciona a hay_cara / MOG2 / FSM_TIMEOUT_FACE_S.
      - Embed igual que FACE_PROCESSED (cooldown EMBED_AND_FACEDETEC_COOLDOWN_S
        en main_mov); cada MATCH renueva el timer a recognized_refresh_s; NO_MATCH
        no expulsa mientras el timer siga activo.

    FACE_PROCESSED / FACE_OUT siguen las reglas de deteccion de cara (RetinaFace).
    """

    _FACE_INFER_STATES = frozenset(
        {
            FlowState.MOV_DETECTED,
            FlowState.MOV_OUT,
            FlowState.FACE_PROCESSED,
            FlowState.FACE_RECOGNIZED,
            FlowState.FACE_OUT,
        }
    )

    def __init__(self, config: FsmConfig | None = None) -> None:
        cfg = config or FsmConfig()
        self._t_mov = float(cfg.timeout_mov_s)
        self._t_face = float(cfg.timeout_face_s)
        self._t_refresh = float(cfg.recognized_refresh_s)
        self.state = FlowState.IDLE
        self._t_ultimo_mov: float | None = None
        self._t_ultima_cara: float | None = None
        self._recognized_id: str | None = None
        self._t_timer_hasta: float | None = None

    def _clear_recognition(self) -> None:
        self._recognized_id = None
        self._t_timer_hasta = None

    def _sin_cara_timeout(self, now: float) -> bool:
        return (
            self._t_ultima_cara is not None
            and (now - self._t_ultima_cara) >= self._t_face
        )

    def _timer_activo(self, now: float) -> bool:
        return self._t_timer_hasta is not None and now < self._t_timer_hasta

    def _renovar_timer(self, now: float) -> None:
        self._t_timer_hasta = now + self._t_refresh

    def _volver_a_face_processed(
        self, now: float, motivo: str, events: list[str]
    ) -> None:
        self._clear_recognition()
        self.state = FlowState.FACE_PROCESSED
        self._t_ultima_cara = now
        self._t_ultimo_mov = now
        events.append(
            "[FSM] FACE_RECOGNIZED -> FACE_PROCESSED ({})".format(motivo)
        )

    def _salir_recognized_si_timer_vencido(
        self, now: float, events: list[str]
    ) -> None:
        if self.state != FlowState.FACE_RECOGNIZED or self._timer_activo(now):
            return
        self._volver_a_face_processed(now, "timer identidad vencido", events)

    def tick_motion(self, hay_mov: bool, now: float) -> FsmTickResult:
        """Transiciones MOG2. Tambien vence timer de identidad (cada frame)."""
        events: list[str] = []

        self._salir_recognized_si_timer_vencido(now, events)

        if hay_mov:
            self._t_ultimo_mov = now

        sin_mov = (
            self._t_ultimo_mov is not None
            and (now - self._t_ultimo_mov) >= self._t_mov
        )
        if sin_mov and self.state in (FlowState.MOV_DETECTED, FlowState.MOV_OUT):
            s_antes = self.state
            self.state = FlowState.IDLE
            self._t_ultima_cara = None
            self._clear_recognition()
            events.append(
                "[FSM] {} -> IDLE (timeout {:.1f}s MOG2)".format(
                    s_antes.value, self._t_mov
                )
            )

        s_antes = self.state
        if self.state == FlowState.IDLE and hay_mov:
            self.state = FlowState.MOV_DETECTED
        elif self.state == FlowState.MOV_DETECTED and not hay_mov:
            self.state = FlowState.MOV_OUT
        elif self.state == FlowState.MOV_OUT and hay_mov:
            self.state = FlowState.MOV_DETECTED

        if self.state != s_antes:
            events.append(
                "[FSM] {} -> {} (MOG2)".format(s_antes.value, self.state.value)
            )
        return self._snapshot(events)

    def tick_face(self, hay_cara: bool, now: float) -> FsmTickResult:
        """Transiciones RetinaFace. FACE_RECOGNIZED no usa hay_cara para salir."""
        events: list[str] = []
        s_antes = self.state

        self._salir_recognized_si_timer_vencido(now, events)

        if self.state == FlowState.FACE_RECOGNIZED:
            if hay_cara:
                self._t_ultima_cara = now
                self._t_ultimo_mov = now
            return self._snapshot(events)

        if hay_cara:
            self._t_ultima_cara = now
            self._t_ultimo_mov = now
            if self.state in (FlowState.MOV_DETECTED, FlowState.MOV_OUT):
                self._clear_recognition()
                self.state = FlowState.FACE_PROCESSED
            elif self.state == FlowState.FACE_OUT:
                self.state = FlowState.FACE_PROCESSED
        elif self.state == FlowState.FACE_PROCESSED:
            self.state = FlowState.FACE_OUT
        elif self.state == FlowState.FACE_OUT and self._sin_cara_timeout(now):
            self.state = FlowState.FACE_PROCESSED_TIMEOUT

        if self.state != s_antes:
            events.append(
                "[FSM] {} -> {} (RetinaFace)".format(s_antes.value, self.state.value)
            )

        if self.state == FlowState.FACE_PROCESSED_TIMEOUT:
            events.append(
                "[FSM] {} -> IDLE (sin cara {:.1f}s)".format(
                    FlowState.FACE_PROCESSED_TIMEOUT.value, self._t_face
                )
            )
            self.state = FlowState.IDLE
            self._t_ultima_cara = None
            self._clear_recognition()

        return self._snapshot(events)

    def notify_embed_match(
        self,
        person_id: str,
        is_match: bool,
        now: float,
    ) -> tuple[str, ...]:
        """
        Entrada a FACE_RECOGNIZED: solo MATCH desde FACE_PROCESSED.
        En FACE_RECOGNIZED: MATCH renueva timer; NO_MATCH no expulsa si timer activo.
        """
        events: list[str] = []

        if is_match:
            s_antes = self.state
            self._recognized_id = person_id
            self._renovar_timer(now)
            if s_antes == FlowState.FACE_PROCESSED:
                self.state = FlowState.FACE_RECOGNIZED
                events.append(
                    "[FSM] {} -> FACE_RECOGNIZED (id={})".format(
                        s_antes.value, person_id
                    )
                )
            elif s_antes == FlowState.FACE_RECOGNIZED:
                events.append(
                    "[FSM] FACE_RECOGNIZED refresh MATCH id={} (timer renovado)".format(
                        person_id
                    )
                )
            return tuple(events)

        if self.state != FlowState.FACE_RECOGNIZED:
            return tuple(events)

        if self._timer_activo(now):
            events.append(
                "[FSM] FACE_RECOGNIZED refresh NO_MATCH (timer activo, id={})".format(
                    self._recognized_id or "?"
                )
            )
            return tuple(events)

        self._volver_a_face_processed(now, "NO_MATCH y timer vencido", events)
        return tuple(events)

    def refresh_outputs(self, now: float) -> FsmTickResult:
        """Snapshot tras notify_embed_match (mismo frame)."""
        del now  # firma estable para main_mov; el snapshot ya no depende de now
        return self._snapshot(())

    def tick_identity_timer(self, now: float) -> FsmTickResult:
        """Solo evalua la expiracion del timer de FACE_RECOGNIZED (sin RetinaFace)."""
        events: list[str] = []
        self._salir_recognized_si_timer_vencido(now, events)
        return self._snapshot(events)

    def _run_embedding(self) -> bool:
        return self.state in (
            FlowState.FACE_PROCESSED,
            FlowState.FACE_RECOGNIZED,
        )

    def _snapshot(self, events: list[str]) -> FsmTickResult:
        return FsmTickResult(
            state=self.state,
            run_face_detector=self.state in self._FACE_INFER_STATES,
            run_embedding=self._run_embedding(),
            transitions=tuple(events),
        )
