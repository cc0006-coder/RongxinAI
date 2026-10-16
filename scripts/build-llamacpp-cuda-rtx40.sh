#!/usr/bin/env bash
set -euo pipefail

# RTX 40-series (Ada) only: sm_89, CUDA 12.4 toolchain for broad driver support.
PLATFORM="${1:-linux}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${LLAMACPP_SRC:-${ROOT_DIR}/.cache/llama.cpp-src}"
BUILD_DIR="${SRC_DIR}/build-zhiyuan-cuda-rtx40-${PLATFORM}"
EDITION_ROOT="${ROOT_DIR}/vendor/llamacpp/cuda"

if [[ ! -d "${SRC_DIR}/.git" && ! -f "${SRC_DIR}/CMakeLists.txt" ]]; then
  echo "[build-llamacpp-cuda-rtx40] Clone llama.cpp into ${SRC_DIR} first (LLAMACPP_SRC)." >&2
  exit 1
fi

CMAKE_ARGS=(
  -S "${SRC_DIR}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=ON
  -DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=89
  -DCUDA_VERSION=12.4
)

case "${PLATFORM}" in
  linux)
    CMAKE_ARGS+=(-DGGML_METAL=OFF)
    EXECUTABLE_NAME=llama-server
    ;;
  win|windows)
    CMAKE_ARGS+=(-DGGML_METAL=OFF)
    EXECUTABLE_NAME=llama-server.exe
    ;;
  *)
    echo "Unsupported platform: ${PLATFORM}" >&2
    exit 1
    ;;
esac

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --config Release --target llama-server -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

BIN_DIR="${EDITION_ROOT}/bin"
mkdir -p "${BIN_DIR}"
cp "${BUILD_DIR}/bin/${EXECUTABLE_NAME}" "${BIN_DIR}/${EXECUTABLE_NAME}"
chmod +x "${BIN_DIR}/${EXECUTABLE_NAME}" 2>/dev/null || true

cat > "${EDITION_ROOT}/manifest.json" <<EOF
{
  "schemaVersion": 1,
  "backend": {
    "version": "bundled",
    "backend": "${PLATFORM}-x64-cuda-rtx40"
  },
  "executable": "bin/${EXECUTABLE_NAME}",
  "cpuFallbackExecutable": "../cpu/bin/${EXECUTABLE_NAME}"
}
EOF

echo "[build-llamacpp-cuda-rtx40] Built ${BIN_DIR}/${EXECUTABLE_NAME} (sm_89, CUDA 12.4)."
