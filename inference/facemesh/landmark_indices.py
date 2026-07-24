"""Indices MediaPipe Face Mesh 468 para filtro UX (cara chica en pantalla).

Fuente: ``mediapipe/python/solutions/face_mesh_connections.py``
(FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE, FACEMESH_NOSE
+ punta/subnasal/aletas + anillos tesselacion perioral/perinasal, ocular y fosa nasal).
Indices precalculados; no se derivan de aristas en runtime.
"""
from __future__ import annotations

from inference.facemesh.constants import NUM_LANDMARKS

# Generado desde FACEMESH_LIPS (contorno labial; sin iris 468+).
LIPS_INDICES: tuple[int, ...] = (
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146,
    178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318,
    321, 324, 375, 402, 405, 409, 415,
)

# Generado desde FACEMESH_LEFT_EYE.
LEFT_EYE_INDICES: tuple[int, ...] = (
    249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398,
    466,
)

# Generado desde FACEMESH_RIGHT_EYE.
RIGHT_EYE_INDICES: tuple[int, ...] = (
    7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
)

# Anillo contorno ocular: vecinos 1-hop en tesselacion de FACEMESH_*_EYE.
_EYE_CONTOUR_RING: tuple[int, ...] = (
    22, 23, 24, 25, 26, 27, 28, 29, 30, 56, 110, 112, 130, 190, 243, 247, 252,
    253, 254, 255, 256, 257, 258, 259, 260, 286, 339, 341, 359, 414, 463, 467,
)

# Generado desde FACEMESH_NOSE.
NOSE_INDICES: tuple[int, ...] = (
    1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275,
    278, 294, 326, 327, 344, 440,
)

# Punta, subnasal, filtro y aletas: no estan todos en FACEMESH_NOSE pero densifican
# bajo la nariz en cara chica (anatomia MediaPipe + vecinos de tesselacion).
_NOSE_TIP_SUBNASAL_EXTRA: tuple[int, ...] = (
    3, 44, 49, 51, 102, 125, 129, 131, 134, 141, 164, 167, 209, 217, 236, 242,
    274, 281, 331, 354, 358, 360, 370, 393, 420, 429, 437, 462,
)

NOSE_REGION_INDICES: tuple[int, ...] = tuple(
    sorted(set(NOSE_INDICES) | set(_NOSE_TIP_SUBNASAL_EXTRA))
)

# Anillo perioral/perinasal: vecinos 1-hop en tesselacion MediaPipe de labios+nariz
# (fosas, comisuras, surco nasogeniano); omitidos en cara chica junto al nucleo.
_PERIORAL_PERINASAL_RING: tuple[int, ...] = (
    8, 11, 12, 15, 16, 18, 20, 38, 41, 42, 43, 47, 57, 62, 72, 73, 74, 76, 77,
    83, 85, 86, 89, 90, 92, 96, 99, 106, 114, 122, 126, 142, 165, 174, 179, 180,
    182, 183, 184, 186, 193, 196, 198, 203, 218, 219, 235, 237, 238, 240, 241,
    248, 250, 268, 271, 272, 273, 277, 279, 287, 292, 302, 303, 304, 306, 307,
    313, 315, 316, 319, 320, 322, 325, 328, 335, 343, 351, 355, 363, 371, 391,
    399, 403, 404, 406, 407, 408, 410, 417, 419, 423, 438, 439, 455, 456, 457,
    458, 460, 461,
)

# Fosas nasales: shell 2-hop desde indices de fosa (102, 49, 48, 115 / 331, 279, 278, 344).
_NASAL_FOSSA_RING: tuple[int, ...] = (
    59, 75, 79, 166, 289, 305, 309, 392,
)

MOUTH_NOSE_EXCLUDED_INDICES: tuple[int, ...] = tuple(
    sorted(
        set(LIPS_INDICES)
        | set(NOSE_REGION_INDICES)
        | set(_PERIORAL_PERINASAL_RING)
        | set(_NASAL_FOSSA_RING)
    )
)

# Ojos + boca/nariz + anillos perioral/oclar/nasal: omitidos cuando el bbox es pequeno.
SMALL_FACE_EXCLUDED_INDICES: tuple[int, ...] = tuple(
    sorted(
        set(LEFT_EYE_INDICES)
        | set(RIGHT_EYE_INDICES)
        | set(_EYE_CONTOUR_RING)
        | set(MOUTH_NOSE_EXCLUDED_INDICES)
    )
)

_SMALL_FACE_EXCLUDED_SET = frozenset(SMALL_FACE_EXCLUDED_INDICES)

# Resto del mesh: UX cuando la cara es chica en pantalla.
SMALL_FACE_DISPLAY_INDICES: tuple[int, ...] = tuple(
    i for i in range(NUM_LANDMARKS) if i not in _SMALL_FACE_EXCLUDED_SET
)
