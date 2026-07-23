#!/usr/bin/env bash
# Cross-compile my_rga wheel for RK3568 (aarch64) from WSL x86_64.
# Output: native/wheels/my_rga-*-cp312-cp312-linux_aarch64.whl
#
# Default: Python 3.12 arm64 (Conda anpr en /opt/anpr-core).
# Override: TARGET_PY_VER=3.11 ./native/build_my_rga_wsl.sh
set -euo pipefail

NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${NATIVE_DIR}/.." && pwd)"
EXT_DIR="${NATIVE_DIR}/my_rga"
WHEELS_DIR="${NATIVE_DIR}/wheels"
LIBRGA_DIR="${LIBRGA_DIR:-${HOME}/librga}"
VENV="${VENV:-${HOME}/venv-rknn311}"
TARGET_PY_VER="${TARGET_PY_VER:-3.12}"

if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "ERROR: no existe ${VENV}. Crear con: python3 -m venv ~/venv-rknn311"
    exit 1
fi

# shellcheck disable=SC1090
source "${VENV}/bin/activate"

echo "==> Host Python: $(which python) ($(python --version))"
echo "==> Target Python placa: ${TARGET_PY_VER} aarch64"
echo "==> Paquete: ${EXT_DIR}"

if ! command -v aarch64-linux-gnu-g++ >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        cmake \
        ninja-build \
        gcc-aarch64-linux-gnu \
        g++-aarch64-linux-gnu \
        wget
fi

if [[ ! -f "${LIBRGA_DIR}/libs/Linux/gcc-aarch64/librga.so" ]]; then
    echo "==> Clonando librga en ${LIBRGA_DIR}"
    git clone --depth 1 https://github.com/airockchip/librga.git "${LIBRGA_DIR}"
fi

RGA_LIB="${LIBRGA_DIR}/libs/Linux/gcc-aarch64/librga.so"
RGA_INCLUDE="${LIBRGA_DIR}/include"
if [[ ! -f "${RGA_LIB}" ]]; then
    echo "ERROR: no se encontro ${RGA_LIB}"
    exit 1
fi

python -m pip install --upgrade pip
python -m pip install pybind11 scikit-build-core build wheel

SYSROOT="${SYSROOT:-${HOME}/sysroot-aarch64-py${TARGET_PY_VER//./}}"
mkdir -p "${SYSROOT}" "${WHEELS_DIR}"

_fetch_deb_py() {
    local py_ver="$1"
    local tmpd base ver
    tmpd="$(mktemp -d)"
    if [[ "${py_ver}" == "3.11" ]]; then
        base="http://ftp.debian.org/debian/pool/main/p/python3.11"
        ver="3.11.2-6+deb12u8"
        cd "${tmpd}"
        for pkg in libpython3.11 libpython3.11-dev; do
            wget -q "${base}/${pkg}_${ver}_arm64.deb" || exit 1
        done
    elif [[ "${py_ver}" == "3.12" ]]; then
        cd "${tmpd}"
        base="http://ports.ubuntu.com/ubuntu-ports/pool/main/p/python3.12"
        ver="3.12.3-1ubuntu0.15"
        for pkg in libpython3.12t64 libpython3.12-dev; do
            wget -q "${base}/${pkg}_${ver}_arm64.deb" || exit 1
        done
    else
        echo "ERROR: TARGET_PY_VER no soportado: ${py_ver}"
        exit 1
    fi
    for deb in *.deb; do
        dpkg-deb -x "${deb}" "${SYSROOT}"
    done
    cd - >/dev/null
    rm -rf "${tmpd}"
}

PY_INCLUDE="${SYSROOT}/usr/include/python${TARGET_PY_VER}"
if [[ ! -f "${PY_INCLUDE}/Python.h" ]]; then
    echo "==> Descargando sysroot Python ${TARGET_PY_VER} arm64"
    _fetch_deb_py "${TARGET_PY_VER}"
fi

PY_LIB="${SYSROOT}/usr/lib/aarch64-linux-gnu/libpython${TARGET_PY_VER}.so"
if [[ ! -f "${PY_LIB}" ]]; then
    PY_LIB="${SYSROOT}/usr/lib/aarch64-linux-gnu/libpython${TARGET_PY_VER}t64.so"
fi
if [[ ! -f "${PY_LIB}" ]]; then
    PY_LIB="${SYSROOT}/usr/lib/libpython${TARGET_PY_VER}.so"
fi
if [[ ! -f "${PY_INCLUDE}/Python.h" || ! -f "${PY_LIB}" ]]; then
    echo "ERROR: sysroot Python ${TARGET_PY_VER} incompleto en ${SYSROOT}"
    exit 1
fi

CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=${EXT_DIR}/toolchain/aarch64-linux-gnu.cmake"
CMAKE_ARGS="${CMAKE_ARGS};-DRGA_INCLUDE_DIR=${RGA_INCLUDE}"
CMAKE_ARGS="${CMAKE_ARGS};-DRGA_LIBRARY=${RGA_LIB}"
CMAKE_ARGS="${CMAKE_ARGS};-DPython3_INCLUDE_DIR=${PY_INCLUDE}"
CMAKE_ARGS="${CMAKE_ARGS};-DPython3_LIBRARY=${PY_LIB}"
CMAKE_ARGS="${CMAKE_ARGS};-DPY_SYSROOT=${SYSROOT}"
export SKBUILD_CMAKE_ARGS="${CMAKE_ARGS}"

cd "${EXT_DIR}"
rm -rf dist build *.egg-info

python -m pip install -v . --no-build-isolation --force-reinstall

OUT_SO="$(find "${VENV}/lib" -path "*/site-packages/my_rga.so" 2>/dev/null | head -1)"
if [[ -z "${OUT_SO}" ]]; then
    OUT_SO="$(find "${VENV}/lib" -name 'my_rga*.so' 2>/dev/null | head -1)"
fi
if [[ -z "${OUT_SO}" || ! -f "${OUT_SO}" ]]; then
    echo "ERROR: no se encontro my_rga.so tras cross-compile"
    find "${VENV}/lib" -name 'my_rga*.so' 2>/dev/null || true
    exit 1
fi

echo "==> Extension compilada: ${OUT_SO}"
file "${OUT_SO}" || true

WHEEL_VERSION="0.1.0"
WHEEL_NAME="my_rga-${WHEEL_VERSION}-cp${TARGET_PY_VER//./}-cp${TARGET_PY_VER//./}-linux_aarch64.whl"
WHEEL_PATH="${WHEELS_DIR}/${WHEEL_NAME}"
rm -f "${WHEELS_DIR}"/my_rga-*.whl

python - <<PY
import base64
import hashlib
import zipfile
from pathlib import Path

so_src = Path("${OUT_SO}")
wheel_path = Path("${WHEEL_PATH}")
version = "${WHEEL_VERSION}"
py_tag = "cp${TARGET_PY_VER//./}"
dist_info = f"my_rga-{version}.dist-info"

metadata = (
    "Metadata-Version: 2.1\n"
    f"Name: my_rga\n"
    f"Version: {version}\n"
    "Summary: Rockchip RGA resize/letterbox/color helpers for RK3568\n"
    "Requires-Python: >=3.8\n"
)
wheel_meta = (
    "Wheel-Version: 1.0\n"
    "Generator: native/build_my_rga_wsl.sh\n"
    "Root-Is-Purelib: false\n"
    f"Tag: {py_tag}-{py_tag}-linux_aarch64\n"
)


def sha256_payload(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={b64}"


entries: list[tuple[str, bytes]] = [
    ("my_rga.so", so_src.read_bytes()),
    (f"{dist_info}/METADATA", metadata.encode()),
    (f"{dist_info}/WHEEL", wheel_meta.encode()),
]

records: list[str] = []
with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for arc, data in entries:
        zf.writestr(arc, data)
        records.append(f"{arc},{sha256_payload(data)},{len(data)}")
    record_body = "\n".join(records) + "\n"
    records.append(f"{dist_info}/RECORD,,")
    zf.writestr(f"{dist_info}/RECORD", record_body + f"{dist_info}/RECORD,,\n")

print(wheel_path)
PY

WHEEL="${WHEEL_PATH}"
if [[ -z "${WHEEL}" ]]; then
    echo "ERROR: no se genero wheel en ${WHEELS_DIR}"
    exit 1
fi

echo ""
echo "Build OK."
echo "Wheel: ${WHEEL}"
unzip -l "${WHEEL}" | head -20
file "${WHEEL}" || true
echo ""
echo "Copiar a placa:"
echo "  scp ${WHEEL} user@placa:/opt/anpr-core/rknn-toolkit-lite/"
echo "  pip install /opt/anpr-core/rknn-toolkit-lite/$(basename "${WHEEL}")"
