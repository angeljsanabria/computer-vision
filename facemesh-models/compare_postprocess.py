"""
Compara postprocess Python (inference/facemesh) vs ONNX con postprocess embebido.

No requiere OpenCV: usa parche BGR sintetico 192x192 para aislar el remap.

  python extract_from_mesh_originals.py
  python compare_postprocess.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent

NUM_LANDMARKS = 468
MESH_SIZE = 192


def _landmarks_mesh_to_frame(
    mesh_points: np.ndarray,
    crop_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    """Copia de inference/facemesh/postprocess.py (evita import de cv2)."""
    pts = np.asarray(mesh_points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] != NUM_LANDMARKS:
        raise ValueError(f"esperado {NUM_LANDMARKS} puntos, got {pts.shape[0]}")
    x1, y1, x2, y2 = crop_xyxy
    crop_w = max(x2 - x1 + 1, 1)
    crop_h = max(y2 - y1 + 1, 1)
    scale = float(MESH_SIZE)
    out = pts.copy()
    out[:, 0] = x1 + (pts[:, 0] / scale) * crop_w
    out[:, 1] = y1 + (pts[:, 1] / scale) * crop_h
    return out

RAW_ONNX = Path(__file__).resolve().parent / "originals" / "face_mesh_192x192.onnx"
POST_ONNX = Path(__file__).resolve().parent / "originals" / "face_mesh_192x192_post.onnx"


def _synthetic_bgr192(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, 192, dtype=np.float32)
    r = np.tile(base, (192, 1))
    g = np.tile(base[:, None], (1, 192))
    b = rng.integers(0, 256, size=(192, 192), dtype=np.uint8)
    patch = np.stack(
        [
            np.clip(r + rng.normal(0, 8, (192, 192)), 0, 255).astype(np.uint8),
            np.clip(g + rng.normal(0, 8, (192, 192)), 0, 255).astype(np.uint8),
            b,
        ],
        axis=2,
    )
    return patch


def _bgr192_to_onnx_nchw(face_bgr: np.ndarray) -> np.ndarray:
    rgb = face_bgr[..., ::-1].astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


def main() -> None:
    if not RAW_ONNX.is_file() or not POST_ONNX.is_file():
        raise SystemExit("Correr primero: python extract_from_mesh_originals.py")

    patch = _synthetic_bgr192()
    crop_xyxy = (120, 80, 120 + 191, 80 + 191)
    x1, y1, x2, y2 = crop_xyxy
    crop_w = max(x2 - x1 + 1, 1)
    crop_h = max(y2 - y1 + 1, 1)

    feed = _bgr192_to_onnx_nchw(patch)

    sess_raw = ort.InferenceSession(str(RAW_ONNX), providers=["CPUExecutionProvider"])
    raw_out = sess_raw.run(None, {sess_raw.get_inputs()[0].name: feed})
    landmarks_raw = np.asarray(raw_out[0], dtype=np.float32).reshape(-1, 3)
    pts_py = _landmarks_mesh_to_frame(landmarks_raw, crop_xyxy)

    sess_post = ort.InferenceSession(str(POST_ONNX), providers=["CPUExecutionProvider"])
    inp = sess_post.get_inputs()[0].name
    result = sess_post.run(
        None,
        {
            inp: feed,
            "crop_x1": np.array([[x1]], dtype=np.int32),
            "crop_y1": np.array([[y1]], dtype=np.int32),
            "crop_width": np.array([[crop_w]], dtype=np.int32),
            "crop_height": np.array([[crop_h]], dtype=np.int32),
        },
    )
    out_names = [o.name for o in sess_post.get_outputs()]
    final_idx = out_names.index("final_landmarks") if "final_landmarks" in out_names else 0
    pts_onnx = np.asarray(result[final_idx], dtype=np.float32).reshape(-1, 3)

    n = min(len(pts_py), len(pts_onnx))
    diff = np.linalg.norm(pts_py[:n, :2] - pts_onnx[:n, :2], axis=1)
    print(f"puntos comparados: {n}")
    print(f"error xy medio: {diff.mean():.4f} px")
    print(f"error xy max:   {diff.max():.4f} px")
    if diff.mean() > 1.0:
        print("WARN: diferencia alta; revisar landmarks_mesh_to_frame vs post ONNX")
    else:
        print("OK: postprocess Python alineado (~sub-pixel vs face_mesh_192x192_post.onnx)")


if __name__ == "__main__":
    main()
