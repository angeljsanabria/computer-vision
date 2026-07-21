"""
Pipeline edge WIP: captura + MOG2 + FSM.

Responsabilidad de este archivo: orquestar el bucle por frame. La logica de
movimiento y estados vive en ``mov_detect/``; captura en ``utils/``;
inferencia (RetinaFace + MobileFaceNet via ``inference/``); UI en ``ui/``.

Flujo por frame:
  1. ``CaptureCameras.get_frame()`` — frame BGR canonico.
  2. ``Mog2MotionSensor.evaluate()`` — movimiento sobre frame reducido.
  3. ``MotionFaceFsm.tick_motion()`` — transiciones IDLE / MOV_*.
  4. ``_sync_umbral_mog2()`` — histéresis de umbral MOG2 segun estado FSM.
  5. Si ``run_face_detector``: RetinaFace + ``tick_face()``.
  6. Si ``run_embedding``: preprocess + MobileFaceNet (cooldown ``EMBED_AND_FACEDETEC_COOLDOWN_S``).
  7. Tras embed: ``gallery @ live`` vs ``gallery.npy`` + meta ``gallery_meta.json``.

Estados FSM (resumen):
  FACE_LOOKING     — vigilancia facial sin MOG2 (ENABLE_MOV_DETECTION=false)
  IDLE / MOV_*      — gate MOG2 (ENABLE_MOV_DETECTION=true)
  IDLE          — sin inferencia facial.
  MOV_DETECTED  — MOG2 supero umbral.
  MOV_OUT       — MOG2 bajo umbral dentro de sesion activa.
  FACE_*        — RetinaFace activo (cuando INFERENCE_BACKEND != none).

Variables de entorno utiles (ver ``configs/settings.py``):
  CONFIG_MODO          — USB | RTSP | RTMP | SNAP
  IP_CAM_RTMP_STREAM — MAIN | EXT | SUB (solo CONFIG_MODO=RTMP; default EXT)
  DISPLAY_IS_ENABLE    — true/false (overlay OpenCV)
  DISPLAY_FORCE_FULL_SCREEN    — true/false (overlay OpenCV) set WND_PROP_FULLSCREEN
  DISPLAY_WIDTH / DISPLAY_HEIGHT — tamano ventana (0 = sin resizeWindow)
  MOG2_* / FSM_TIMEOUT_* — umbrales MOG2 y timeouts mov/cara
  FSM_RECOGNIZED_REFRESH_S — retencion identidad MATCH en FACE_RECOGNIZED (s)
  INFERENCE_BACKEND    — none | pc | rk3568 (factory en ``inference/``)
  FACE_PROCESS_TOP_N     — cuantas caras mostrar (bbox/track), mejores primero
  FACE_EMBED_TOP_N       — de esas, cuantas reciben embed + reconocimiento
  ENABLE_FACEMESH        — UX landmarks 468 en tracks sin MATCH (requiere display)
  FACE_MESH_TOP_N        — max desconocidos con mesh por frame (defecto 2)
  FACE_MESH_EVERY_N_FRAMES — inferir cada N frames; hold anti-parpadeo (defecto 2)
  EMBED_MIN_SCORE        — score minimo RetinaFace para embed
  EMBED_AND_FACEDETEC_COOLDOWN_S — segundos entre embeds/deteccion (0 = cada tick con cara)
  FACE_ALIGNMENT_ENABLE              — true=align ArcFace siempre (refs alineadas)
  FACE_ROT_ALIGNMENT_SIMPLE_ENABLE   — true=hibrido crop/roll-fix
  FACE_ROLL_MAX_DEG                  — umbral roll-fix simple
  EMBED_SIM_MIN_MATCH      — umbral coseno identidad (defecto 0.57)
  EMBED_REF_GALLERY_DIR    — carpeta con gallery.npy + gallery_meta.json (defecto ../data/)
  LOG_MODE                 — PROD (INFO, default) | DEV (DEBUG, telemetria)
  ENABLE_ENDPOINT          — true/false (GET /api/v1/vision-status via vision_http)
  HTTP_API_HOST / HTTP_API_PORT — bind del servidor (default 0.0.0.0:8008)
  IMG_QUALITY_CHECK_ENABLE — true/false snapshot headless (default false)
  IMG_QUALITY_CHECK_INTERVAL_S — segundos entre guardados (pisa img_snap.jpg)

Ejemplos:
  cd src && python main.py
  python src/main.py   # desde la raiz del repo (main fija cwd en src/)

  INFERENCE_BACKEND=pc DISPLAY_IS_ENABLE=true CONFIG_MODO=USB python main.py
  INFERENCE_BACKEND=none python main.py

Despliegue RK3568: registrar manejadores SIGINT/SIGTERM para cierre limpio
con systemd (ver ``main.py`` / comentarios historicos en el repo).
"""
from __future__ import annotations

import logging
import os
import sys
import time

# Rutas relativas en settings (models/, ../data) asumen cwd = src/.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from configs import settings_track as s  # noqa: E402
from mov_detect import (  # noqa: E402
    FlowState,
    FsmConfig,
    Mog2Config,
    MotionFaceFsm,
    Mog2MotionSensor,
    MotionResult,
)
from mov_detect.types import FsmTickResult  # noqa: E402
import numpy as np  # noqa: E402
from inference import (  # noqa: E402
    FaceDetector,
    FaceEmbedder,
    FaceMeshEstimator,
    build_embedder,
    build_face_detector,
    build_face_mesh,
    build_identity_matcher,
)
from inference.identity.matcher import (  # noqa: E402
    GALLERY_ALIGN_META_NAME,
    GALLERY_ALIGN_NPY_NAME,
    GALLERY_META_NAME,
    GALLERY_NPY_NAME,
)
from inference.identity.types import IdentityMatch  # noqa: E402
from inference.types import FaceDetections, FaceEmbedding, FaceMeshLandmarks  # noqa: E402
from inference.face_preprocess import prepare_face_patch  # noqa: E402
from inference.facemesh.from_detection import estimate_from_det  # noqa: E402
from inference.retinaface.select_best import mejores_caras  # noqa: E402
from bytetrack import ByteTrackConfig, FaceTracker, TrackResult, build_face_tracker  # noqa: E402
from ui import DisplayBanner, FrameView, PipelineDisplay  # noqa: E402
from utils.capture_cameras import CaptureCameras  # noqa: E402
from utils.ip_cam_urls import build_rtmp_url, build_rtsp_url, build_snap_url  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers locales (orquestacion del pipeline por etapas).
#
# MOG2, FSM e inferencia viven en mov_detect/ e inference/. La UI vive en ui/.
# Reloj del bucle: time.monotonic_ns() (int). FSM aun trabaja en segundos:
# se convierte en la frontera (now_ns / _NS_PER_S). Cooldown embed/deteccion en ns.
# ---------------------------------------------------------------------------

_NS_PER_S = 1_000_000_000


def _log_transitions(transitions: tuple[str, ...]) -> None:
    """Escribe en log las transiciones FSM que devolvio tick_motion/tick_face."""
    for msg in transitions:
        logging.debug(msg)


def _sync_umbral_mog2(
    motion: Mog2MotionSensor,
    estado_antes: FlowState,
    estado_despues: FlowState,
) -> None:
    """
    Sincroniza umbral MOG2 con el estado FSM (histéresis).

    En IDLE usa el umbral base (menos sensible). Al salir de IDLE baja el
    umbral (mas sensible) para no perder actividad durante MOV_* / FACE_*.
    Debe llamarse despues de cada tick que cambie ``fsm.state``.
    """
    if estado_despues == estado_antes:
        return
    if estado_despues == FlowState.IDLE:
        motion.set_umbral_idle()
    elif estado_antes == FlowState.IDLE:
        motion.set_umbral_activo()


def _log_mog2(mov: MotionResult, umbral: int) -> None:
    """Log de una lectura MOG2 (pixeles en mascara vs umbral activo)."""
    tag = "MOV_DETECTED" if mov.hay_mov else "NOT_MOV"
    logging.debug("[MOG2] %s pixels=%d umbral=%d", tag, mov.pixel_count, umbral)


def _tick_mog2_fsm(
    motion: Mog2MotionSensor,
    fsm: MotionFaceFsm,
    frame,
    now_ns: int,
) -> tuple[MotionResult, FsmTickResult]:
    """
    Un ciclo MOG2 + FSM por frame (fase actual del pipeline).

    Encapsula evaluate -> tick_motion -> sync umbral -> log. RetinaFace y
    embed iran despues de este bloque, usando ``fsm_out.run_face_detector`` y
    ``fsm_out.run_embedding`` (no dentro de esta funcion).

    Ahorro CPU: en FACE_LOOKING, FACE_PROCESSED y FACE_RECOGNIZED (y siempre si
    ENABLE_MOV_DETECTION=false) no se corre MOG2 evaluate. Con la cara presente
    el movimiento no decide transiciones (las maneja RetinaFace) y no actualizar
    el fondo evita absorber a la persona; el fondo se recalibra luego en FACE_OUT
    antes de volver a IDLE/FACE_LOOKING. Se fuerza hay_mov=True (pixel_count=-1
    = no medido) solo para el overlay.
    """
    now_s = now_ns / _NS_PER_S
    estado_antes = fsm.state

    skip_mog2 = not s.ENABLE_MOV_DETECTION or estado_antes in (
        FlowState.FACE_LOOKING,
        FlowState.FACE_PROCESSED,
        FlowState.FACE_RECOGNIZED,
    )
    if skip_mog2:
        if estado_antes == FlowState.FACE_RECOGNIZED:
            fsm_out = fsm.tick_identity_timer(now_s)
        elif estado_antes == FlowState.FACE_PROCESSED:
            fsm_out = fsm.refresh_outputs(now_s)
        else:
            # IDLE -> FACE_LOOKING (MOG2 off) o permanecer en FACE_LOOKING
            fsm_out = fsm.tick_motion(hay_mov=False, now=now_s)
        _log_transitions(fsm_out.transitions)
        return MotionResult(hay_mov=True, pixel_count=-1), fsm_out

    mov = motion.evaluate(frame)
    motion.log_motion_if_changed(mov)
    #_log_mog2(mov, motion.umbral_pixeles)  # ver los de movimiento en cada frame

    fsm_out = fsm.tick_motion(hay_mov=mov.hay_mov, now=now_s)
    _sync_umbral_mog2(motion, estado_antes, fsm_out.state)
    _log_transitions(fsm_out.transitions)
    return mov, fsm_out


def _debe_saltar_deteccion(
    state: FlowState, now_ns: int, t_ultimo_embed_ns: int | None
) -> bool:
    """En FACE_RECOGNIZED sin full-rate, RetinaFace sigue el cooldown de embed."""
    if s.FACE_DETECT_FULLRATE or state != FlowState.FACE_RECOGNIZED:
        return False
    cooldown_ns = int(s.EMBED_AND_FACEDETEC_COOLDOWN_S * _NS_PER_S)
    if cooldown_ns <= 0:
        return False
    return (
        t_ultimo_embed_ns is not None
        and (now_ns - t_ultimo_embed_ns) < cooldown_ns
    )


def _tick_retinaface_if_needed(
    face: FaceDetector | None,
    fsm: MotionFaceFsm,
    motion: Mog2MotionSensor,
    frame,
    now_ns: int,
    fsm_out: FsmTickResult,
    t_ultimo_embed_ns: int | None,
) -> tuple[FaceDetections | None, FsmTickResult]:
    """
    RetinaFace + ranking + tick_face cuando la FSM lo indica.

    Devuelve las mejores ``FACE_PROCESS_TOP_N`` caras (bbox/track/overlay).
    La FSM usa ``hay_cara`` sobre todas las detecciones del modelo, no solo las filtradas.
    """
    if not fsm_out.run_face_detector or face is None:
        return None, fsm_out
    if _debe_saltar_deteccion(fsm_out.state, now_ns, t_ultimo_embed_ns):
        return None, fsm_out

    raw = face.detect(frame)
    dets = mejores_caras(raw, top_n=s.FACE_PROCESS_TOP_N)
    estado_antes = fsm.state
    fsm_out = fsm.tick_face(hay_cara=raw.has_faces, now=now_ns / _NS_PER_S)
    _sync_umbral_mog2(motion, estado_antes, fsm_out.state)
    _log_transitions(fsm_out.transitions)
    if estado_antes == FlowState.FACE_OUT:
        if fsm_out.state == FlowState.IDLE:
            logging.info("Ya no hay detecciones de rostros.")
        elif fsm_out.state == FlowState.FACE_LOOKING:
            logging.info("Sin caras; vigilancia facial activa (FACE_LOOKING).")
    return dets, fsm_out


def _tick_bytetrack_if_needed(
    tracker: FaceTracker | None, dets: FaceDetections | None
) -> TrackResult | None:
    """
    Tracking visual sobre dets ya filtradas (top-N); no altera dets ni FSM.

    Aislado con try/except: una falla de tracking nunca debe tapar el resto
    del frame (embed/matcher/FSM/display). Degrada a None (overlay cae a
    _draw_faces) en vez de propagar la excepcion.
    """
    if tracker is None:
        return None
    try:
        return tracker.update(dets)
    except Exception as exc:
        logging.warning("ByteTrack: fallo update(): %s", exc)
        return None


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU entre dos cajas (x1, y1, x2, y2). Solo para correlacionar display."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _track_id_for_bbox(
    bbox: np.ndarray | None, tracks: TrackResult | None, *, min_iou: float = 0.3
) -> int | None:
    """
    Correlaciona geometricamente el bbox embedeado con el track de ByteTrack
    mas solapado (solo para etiquetar el overlay; no influye en el matching).
    """
    if bbox is None or tracks is None or not tracks.tracks:
        return None
    best_id, best_iou = None, 0.0
    for track in tracks.tracks:
        iou = _iou_xyxy(bbox, track.tlbr)
        if iou > best_iou:
            best_iou, best_id = iou, track.track_id
    return best_id if best_iou >= min_iou else None


def _det_row_for_track(
    track_tlbr: np.ndarray,
    dets: FaceDetections | None,
    *,
    min_iou: float = 0.3,
) -> np.ndarray | None:
    """Fila RetinaFace mas solapada con el track (para crop FaceMesh)."""
    if dets is None or not dets.has_faces:
        return None
    best_row: np.ndarray | None = None
    best_iou = 0.0
    for row in dets.dets:
        iou = _iou_xyxy(track_tlbr, row[:4])
        if iou > best_iou:
            best_iou = iou
            best_row = row
    return best_row if best_iou >= min_iou and best_row is not None else None


def _tick_facemesh_if_needed(
    mesh: FaceMeshEstimator | None,
    frame,
    dets: FaceDetections | None,
    tracks: TrackResult | None,
    identity_by_track: dict[int, IdentityMatch],
    hold: dict[int, FaceMeshLandmarks],
    frame_idx: int,
) -> dict[int, FaceMeshLandmarks]:
    """
    FaceMesh UX: landmarks solo en tracks sin MATCH, hasta FACE_MESH_TOP_N.

    ``hold`` persiste entre frames (mismo patron que identity_by_track):
    - throttle: solo infiere cuando ``frame_idx % FACE_MESH_EVERY_N_FRAMES == 0``
    - anti-parpadeo: si no hay det correlacionada, reusa landmarks previos
    - MATCH / track muerto: se elimina del hold

    Corre despues de actualizar ``identity_by_track``. Sin display o sin
    estimator limpia hold y no hace trabajo.
    """
    if not s.ENABLE_FACEMESH or not s.DISPLAY_IS_ENABLE or mesh is None:
        hold.clear()
        return {}
    if tracks is None or not tracks.tracks:
        hold.clear()
        return {}

    live_ids = {track.track_id for track in tracks.tracks}
    for tid in list(hold.keys()):
        if tid not in live_ids:
            del hold[tid]
            continue
        idm = identity_by_track.get(tid)
        if idm is not None and idm.is_match:
            del hold[tid]

    candidates = []
    for track in tracks.tracks:
        idm = identity_by_track.get(track.track_id)
        if idm is not None and idm.is_match:
            continue
        candidates.append(track)
    candidates.sort(key=lambda t: t.score, reverse=True)
    top_n = min(s.FACE_MESH_TOP_N, len(candidates))
    top = candidates[:top_n]
    top_ids = {track.track_id for track in top}

    for tid in list(hold.keys()):
        if tid not in top_ids:
            del hold[tid]

    every_n = max(1, int(s.FACE_MESH_EVERY_N_FRAMES))
    do_infer = (frame_idx % every_n) == 0

    if do_infer:
        for track in top:
            det_row = _det_row_for_track(track.tlbr, dets)
            if det_row is None:
                # Sin det fresca: conservar hold[track_id] si existe.
                continue
            try:
                landmarks = estimate_from_det(
                    frame,
                    det_row,
                    mesh,
                    margin_frac=s.FACE_CROP_MARGIN_FRAC,
                )
            except Exception as exc:
                logging.warning("[FaceMesh] fallo estimate: %s", exc)
                continue
            if landmarks is not None:
                hold[track.track_id] = landmarks

    return {tid: hold[tid] for tid in top_ids if tid in hold}


def _tick_embed_if_needed(
    embedder: FaceEmbedder | None,
    frame,
    dets: FaceDetections | None,
    fsm_out: FsmTickResult,
    now_ns: int,
    t_ultimo_embed_ns: int | None,
) -> tuple[list[tuple[FaceEmbedding, np.ndarray]], int | None]:
    """
    Preprocess + MobileFaceNet en FACE_PROCESSED y FACE_RECOGNIZED; en ambos
    estados se respeta el cooldown EMBED_AND_FACEDETEC_COOLDOWN_S. En FACE_RECOGNIZED cada MATCH
    renueva el timer de sesion (FSM_RECOGNIZED_REFRESH_S).

    ``dets`` ya viene rankeado (``mejores_caras``, ``FACE_PROCESS_TOP_N``).
    Solo las primeras ``FACE_EMBED_TOP_N`` filas son candidatas a embed; cada
    una debe superar ``EMBED_MIN_SCORE``. Cada par (embedding, bbox) correlaciona
    display con ByteTrack.
    """
    if not fsm_out.run_embedding or embedder is None:
        return [], t_ultimo_embed_ns
    if dets is None or not dets.has_faces:
        return [], t_ultimo_embed_ns

    cooldown_ns = int(s.EMBED_AND_FACEDETEC_COOLDOWN_S * _NS_PER_S)
    if (
        cooldown_ns > 0
        and t_ultimo_embed_ns is not None
        and (now_ns - t_ultimo_embed_ns) < cooldown_ns
    ):
        return [], t_ultimo_embed_ns

    embed_top_n = min(s.FACE_EMBED_TOP_N, int(dets.dets.shape[0]))
    results: list[tuple[FaceEmbedding, np.ndarray]] = []
    for row in dets.dets[:embed_top_n]:
        if float(row[4]) < s.EMBED_MIN_SCORE:
            continue
        try:
            patch = prepare_face_patch(
                frame,
                row,
                arcface_align_enable=s.FACE_ALIGNMENT_ENABLE,
                rot_align_simple_enable=s.FACE_ROT_ALIGNMENT_SIMPLE_ENABLE,
                max_abs_roll_deg=s.FACE_ROLL_MAX_DEG,
                crop_margin_frac=s.FACE_CROP_MARGIN_FRAC,
            )
            vector = embedder.embed(patch.bgr)
        except Exception as exc:
            logging.warning("[Embed] fallo preprocess o inferencia: %s", exc)
            continue

        logging.debug(
            "[Embed] score=%.3f dim=%d arcface=%s roll_fix=%s roll=%.1f",
            float(row[4]),
            vector.size,
            patch.used_arcface_align,
            patch.used_roll_fix,
            patch.roll_deg,
        )
        results.append((FaceEmbedding(vector=vector), row[:4].copy()))

    if not results:
        return [], t_ultimo_embed_ns
    return results, now_ns


def _release_runtime(
    obj: FaceDetector | FaceEmbedder | FaceMeshEstimator | None,
) -> None:
    """Libera runtime si el objeto expone release() (p. ej. RKNNLite en RK3568)."""
    if obj is None:
        return
    release = getattr(obj, "release", None)
    if callable(release):
        release()


def _release_face_detector(face: FaceDetector | None) -> None:
    """Libera runtime del detector si expone release() (p. ej. RKNNLite en RK3568)."""
    _release_runtime(face)


def _publish_vision_http_status(
    fsm: MotionFaceFsm,
    fsm_out: FsmTickResult,
    dets: FaceDetections | None,
    last_identity: IdentityMatch | None,
    now_ns: int,
) -> None:
    """Observacion HTTP: publica snapshot sin alterar FSM ni logs."""
    if not s.ENABLE_ENDPOINT:
        return
    try:
        from vision_http import derive_vision_status, vision_store

        vision_store.publish(
            derive_vision_status(
                fsm_state=fsm_out.state,
                dets=dets,
                display_identity = (
                    last_identity
                    if fsm_out.state == FlowState.FACE_RECOGNIZED
                    else None
                ),
                refresh_remaining_s=fsm.recognized_refresh_remaining_s(
                    now_ns / _NS_PER_S
                ),
            )
        )
    except Exception as exc:
        logging.warning("API vision: fallo al publicar snapshot: %s", exc)


def main() -> int:
    """
    Punto de entrada: valida config, warmup MOG2, bucle por frame, cleanup.

    Retorna 0 si termino bien; 1 si hubo excepcion no controlada en el bucle.
    """
    s.validar_todo()

    mog2_cfg = Mog2Config(
        process_width=s.MOG2_PROCESS_WIDTH,
        process_height=s.MOG2_PROCESS_HEIGHT,
        history=s.MOG2_HISTORY,
        var_threshold=s.MOG2_VAR_THRESHOLD,
        movimiento_pixeles=s.MOG2_MOVIMIENTO_PIXELES,
        warmup_frames=s.MOG2_WARMUP_FRAMES,
        warmup_learning_rate=s.MOG2_WARMUP_LEARNING_RATE,
    )
    fsm_cfg = FsmConfig(
        timeout_mov_s=s.FSM_TIMEOUT_MOV_S,
        timeout_face_s=s.FSM_TIMEOUT_FACE_S,
        recognized_refresh_s=s.FSM_RECOGNIZED_REFRESH_S,
        enable_mov_detection=s.ENABLE_MOV_DETECTION,
    )
    motion = Mog2MotionSensor(mog2_cfg, use_rga=s.USE_RGA)
    fsm = MotionFaceFsm(fsm_cfg)

    backend = s.INFERENCE_BACKEND
    if backend == "pc":
        retinaface_model = s.RETINAFACE_MODEL_PC
        mobilefacenet_model = s.MOBILEFACENET_MODEL_PC
    elif backend == "rk3568":
        retinaface_model = s.RETINAFACE_MODEL_RK3568
        mobilefacenet_model = s.MOBILEFACENET_MODEL_RK3568
    else:
        retinaface_model = ""
        mobilefacenet_model = ""

    face = build_face_detector(
        backend,
        retinaface_model,
        s.RETINAFACE_SCORE_DETECCION,
        s.RETINAFACE_SCORE_PRE_NMS,
        use_rga=s.USE_RGA,
    )
    if face is not None:
        logging.debug("RetinaFace activo (backend=%s)", s.INFERENCE_BACKEND)
    else:
        logging.debug(
            "RetinaFace desactivado (INFERENCE_BACKEND=%s)", s.INFERENCE_BACKEND
        )

    embedder = build_embedder(backend, mobilefacenet_model)
    if embedder is not None:
        logging.debug(
            "MobileFaceNet activo (backend=%s, embed_min_score=%.2f, cooldown=%.1f s)",
            s.INFERENCE_BACKEND,
            s.EMBED_MIN_SCORE,
            s.EMBED_AND_FACEDETEC_COOLDOWN_S,
        )
    else:
        logging.debug(
            "MobileFaceNet desactivado (INFERENCE_BACKEND=%s)", s.INFERENCE_BACKEND
        )

    if s.FACE_ALIGNMENT_ENABLE:
        gallery_npy = GALLERY_ALIGN_NPY_NAME
        gallery_meta = GALLERY_ALIGN_META_NAME
    else:
        gallery_npy = GALLERY_NPY_NAME
        gallery_meta = GALLERY_META_NAME

    matcher = build_identity_matcher(
        backend,
        s.EMBED_REF_GALLERY_DIR,
        s.EMBED_SIM_MIN_MATCH,
        gallery_npy,
        gallery_meta,
    )
    if matcher is not None and matcher.count > 0:
        logging.debug(
            "Matcher identidad activo (refs=%d, sim_min=%.2f, match=gallery@live)",
            matcher.count,
            s.EMBED_SIM_MIN_MATCH,
        )

    face_tracker = build_face_tracker(
        s.ENABLE_FACE_TRACKING,
        ByteTrackConfig(
            track_thresh=s.BYTETRACK_TRACK_THRESH,
            match_thresh=s.BYTETRACK_MATCH_THRESH,
            track_buffer=s.BYTETRACK_TRACK_BUFFER,
            frame_rate=s.BYTETRACK_FRAME_RATE,
        ),
    )
    if face_tracker is not None:
        logging.debug(
            "ByteTrack activo (track_thresh=%.2f, match_thresh=%.2f, buffer=%d, fps=%.1f)",
            s.BYTETRACK_TRACK_THRESH,
            s.BYTETRACK_MATCH_THRESH,
            s.BYTETRACK_TRACK_BUFFER,
            s.BYTETRACK_FRAME_RATE,
        )
    else:
        logging.debug("ByteTrack desactivado (ENABLE_FACE_TRACKING=false)")

    face_mesh: FaceMeshEstimator | None = None
    if s.ENABLE_FACEMESH:
        if backend == "pc":
            facemesh_model = s.FACEMESH_MODEL_PC
        elif backend == "rk3568":
            facemesh_model = s.FACEMESH_MODEL_RK3568
        else:
            facemesh_model = ""
        if facemesh_model:
            face_mesh = build_face_mesh(backend, facemesh_model)
        if face_mesh is not None:
            logging.debug(
                "FaceMesh activo (backend=%s, top_n=%d, every_n=%d + hold)",
                s.INFERENCE_BACKEND,
                s.FACE_MESH_TOP_N,
                s.FACE_MESH_EVERY_N_FRAMES,
            )
        else:
            logging.debug(
                "FaceMesh no creado (ENABLE_FACEMESH=%s, backend=%s)",
                s.ENABLE_FACEMESH,
                s.INFERENCE_BACKEND,
            )
    else:
        logging.debug("FaceMesh desactivado (ENABLE_FACEMESH=false)")

    display = PipelineDisplay.from_settings(
        enabled=s.DISPLAY_IS_ENABLE,
        force_full_screen=s.DISPLAY_FORCE_FULL_SCREEN,
        display_width=s.DISPLAY_WIDTH,
        display_height=s.DISPLAY_HEIGHT,
        banner=(
            DisplayBanner.try_from_path(s.DISPLAY_BANNER_PATH)
            if s.DISPLAY_IS_ENABLE
            else None
        ),
    )
    capture: CaptureCameras | None = None
    exit_code = 0

    try:
        if s.ENABLE_ENDPOINT:
            from vision_http import start_api_thread

            start_api_thread(s.HTTP_API_HOST, s.HTTP_API_PORT)

        display.setup()

        quality_snap = None
        if s.IMG_QUALITY_CHECK_ENABLE:
            from utils.img_quality_snap import ImgQualitySnapSaver

            quality_snap = ImgQualitySnapSaver(
                interval_s=s.IMG_QUALITY_CHECK_INTERVAL_S,
                out_dir=s.IMG_QUALITY_CHECK_DIR,
            )

        if s.MODO == "RTMP":
            stream_url = build_rtmp_url(
                s.IP_CAM_HOST,
                s.IP_CAM_USER,
                s.IP_CAM_PASS,
                s.IP_CAM_RTMP_PORT,
                s.IP_CAM_RTMP_STREAM_SELECTED,
            )
        else:
            stream_url = build_rtsp_url(
                s.IP_CAM_HOST,
                s.IP_CAM_USER,
                s.IP_CAM_PASS,
                s.IP_CAM_RTSP_PORT,
                s.IP_CAM_RTSP_STREAM_PATH_SELECTED_RESOLUTION,
            )
        snap_url = build_snap_url(
            s.IP_CAM_HOST,
            s.IP_CAM_USER,
            s.IP_CAM_PASS,
            s.IP_CAM_SNAP_RES_QUERY_SELECTED_RESOLUTION,
        )

        capture = CaptureCameras(
            mode=s.MODO,
            rtsp_url=stream_url,
            snap_url=snap_url,
            usb_index=s.USB_INDEX,
            warmup_frames=s.WARMUP_FRAMES,
            buffer_size=s.BUFFER_SIZE,
            reintento_seg=s.REINTENTO_SEG,
            http_timeout_s=s.HTTP_TIMEOUT_S,
            max_fps=s.MAX_FPS,
            log_cada_n_frames=s.LOG_CADA_N_FRAMES,
            cap_frame_width=s.CAP_FRAME_WIDTH,
            cap_frame_height=s.CAP_FRAME_HEIGHT,
            usb_rotate_deg=s.USB_ROTATE_DEG,
            usb_camera_image_mode=s.USB_CAMERA_IMAGE_MODE,
            usb_brightness=s.USB_BRIGHTNESS,
            usb_contrast=s.USB_CONTRAST,
            usb_saturation=s.USB_SATURATION,
            quality_snap=quality_snap,
        ).start()

        if s.ENABLE_MOV_DETECTION:
            motion.warmup_from_first_frame(
                capture.get_frame,
                n_frames=mog2_cfg.warmup_frames,
                timeout_s=s.MOG2_WARMUP_TIMEOUT_S,
            )
        else:
            logging.debug("MOG2 warmup omitido (ENABLE_MOV_DETECTION=false)")

        logging.debug(
            "Pipeline MOG2+FSM+RetinaFace+Embed+ID en marcha. Ctrl+C para salir."
        )
        #motion.reset_motion_log()

        t_ultimo_embed_ns: int | None = None
        last_identity: IdentityMatch | None = None
        identity_by_track: dict[int, IdentityMatch] = {}
        facemesh_hold: dict[int, FaceMeshLandmarks] = {}
        facemesh_frame_idx = 0

        while True:
            try:
                has_frame, frame = capture.get_frame()

                if has_frame and frame is not None:
                    now_ns = time.monotonic_ns()
                    now_s = now_ns / _NS_PER_S
                    mov, fsm_out = _tick_mog2_fsm(motion, fsm, frame, now_ns)
                    dets, fsm_out = _tick_retinaface_if_needed(
                        face, fsm, motion, frame, now_ns, fsm_out, t_ultimo_embed_ns
                    )
                    tracks = _tick_bytetrack_if_needed(face_tracker, dets)
                    embed_batch, t_ultimo_embed_ns = _tick_embed_if_needed(
                        embedder, frame, dets, fsm_out, now_ns, t_ultimo_embed_ns
                    )
                    live_identity: IdentityMatch | None = None
                    batch_any_match = False
                    batch_best_match: IdentityMatch | None = None
                    if embed_batch and matcher is not None:
                        for embedding, embedded_bbox in embed_batch:
                            matched = matcher.match(embedding.vector)
                            if matched is None:
                                continue
                            live_identity = matched
                            if matched.is_match:
                                batch_any_match = True
                                if (
                                    batch_best_match is None
                                    or matched.similarity > batch_best_match.similarity
                                ):
                                    batch_best_match = matched
                                last_identity = matched
                                correlado = _track_id_for_bbox(embedded_bbox, tracks)
                                if correlado is not None:
                                    identity_by_track[correlado] = matched
                                logging.debug(
                                    "[ID] MATCH fila=%d id=%s nombre=%r sim=%.3f (>=%.2f)",
                                    matched.row_index,
                                    matched.person_id,
                                    matched.nombre,
                                    matched.similarity,
                                    s.EMBED_SIM_MIN_MATCH,
                                )
                            else:
                                logging.debug(
                                    "[ID] NO_MATCH fila=%d id=%s nombre=%r sim=%.3f "
                                    "< umbral %.2f",
                                    matched.row_index,
                                    matched.person_id,
                                    matched.nombre,
                                    matched.similarity,
                                    s.EMBED_SIM_MIN_MATCH,
                                )

                        if batch_any_match and batch_best_match is not None:
                            _log_transitions(
                                fsm.notify_embed_match(
                                    batch_best_match.person_id, True, now_s
                                )
                            )
                            refresh_restante = fsm.recognized_refresh_remaining_s(now_s)
                            logging.info(
                                "[ID] MATCH id=%s nombre=%r refresh_restante=%.1f s",
                                batch_best_match.person_id,
                                batch_best_match.nombre,
                                refresh_restante if refresh_restante is not None else 0.0,
                            )
                        elif embed_batch:
                            state_antes = fsm.state
                            _log_transitions(
                                fsm.notify_embed_match("", False, now_s)
                            )
                            if fsm.state == FlowState.FACE_PROCESSED:
                                last_identity = None
                                identity_by_track.clear()
                            elif state_antes == FlowState.FACE_RECOGNIZED:
                                logging.debug(
                                    "[ID] NO_MATCH refresh (timer activo, "
                                    "se mantiene ultimo MATCH)"
                                )
                            refresh_restante = fsm.recognized_refresh_remaining_s(now_s)
                            if refresh_restante is None or refresh_restante <= 0.0:
                                n_personas = (
                                    int(dets.dets.shape[0])
                                    if dets is not None and dets.has_faces
                                    else 0
                                )
                                if n_personas == 1:
                                    logging.info("Hay 1 persona, sin identificar.")
                                else:
                                    logging.info(
                                        "Hay %d personas, sin identificar.",
                                        n_personas,
                                    )
                            elif last_identity is not None:
                                logging.info(
                                    "[ID] MATCH retenido id=%s nombre=%r "
                                    "refresh_restante=%.1f s",
                                    last_identity.person_id,
                                    last_identity.nombre,
                                    refresh_restante,
                                )

                    fsm_out = fsm.refresh_outputs(now_s)

                    if fsm_out.state == FlowState.IDLE:
                        last_identity = None
                        identity_by_track.clear()
                        facemesh_hold.clear()
                        display_identity = None
                        identity_stale = False
                    elif fsm_out.state == FlowState.FACE_RECOGNIZED and last_identity:
                        display_identity = last_identity
                        identity_stale = False
                    elif (
                        live_identity is not None
                        and fsm_out.state == FlowState.FACE_PROCESSED
                    ):
                        display_identity = live_identity
                        identity_stale = False
                    elif last_identity is not None:
                        display_identity = last_identity
                        identity_stale = True
                    else:
                        display_identity = None
                        identity_stale = False

                    facemesh_by_track = _tick_facemesh_if_needed(
                        face_mesh,
                        frame,
                        dets,
                        tracks,
                        identity_by_track,
                        facemesh_hold,
                        facemesh_frame_idx,
                    )
                    facemesh_frame_idx += 1

                    view = FrameView(
                        mov=mov,
                        fsm=fsm_out,
                        dets=dets,
                        identity=display_identity,
                        identity_is_stale=identity_stale,
                        tracks=tracks,
                        identity_by_track=identity_by_track or None,
                        facemesh_by_track=facemesh_by_track or None,
                    )
                    _publish_vision_http_status(
                        fsm, fsm_out, dets, last_identity, now_ns
                    )
                    display.show(frame, view)
                    if display.poll_quit():
                        logging.debug("Salida solicitada desde ventana (q).")
                        break
                else:
                    if display.poll_quit():
                        logging.debug("Salida solicitada desde ventana (q).")
                        break
                    time.sleep(0.001)
            except Exception as exc:
                logging.exception("Fallo en el bucle principal: %s", exc)
                time.sleep(0.01)

    except KeyboardInterrupt:
        logging.warning("Interrupcion por teclado. Cerrando...")
    finally:
        logging.debug("Liberando hardware y sockets...")
        if s.ENABLE_ENDPOINT:
            try:
                from vision_http import stop_api_thread

                stop_api_thread()
            except Exception as exc:
                logging.warning("API vision: error en cierre: %s", exc)
        _release_face_detector(face)
        _release_runtime(embedder)
        _release_runtime(face_mesh)
        if capture is not None:
            capture.stop()
        display.teardown()
        logging.debug("Proceso terminado.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
