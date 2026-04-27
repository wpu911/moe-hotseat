#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-}"
if [[ -z "$SRC" ]]; then
  echo "Usage: $0 /path/to/llama.cpp" >&2
  exit 1
fi

cd "$SRC"

if [[ ! -d build-hip ]]; then
  echo "build-hip not found. Configure llama.cpp HIP build first." >&2
  echo "Example:" >&2
  echo "  cmake -S . -B build-hip -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release" >&2
  exit 1
fi

# Force the changed loader file to rebuild. CMake sometimes enjoys pretending
# everything is up to date. Humanity deserves better, but here we are.
touch src/llama-model-loader.cpp
find build-hip -type f \( \
  -name '*llama-model-loader.cpp.o' -o \
  -name '*llama-model-loader.cpp.o.d' \
\) -print -delete || true

cmake --build build-hip --target llama-server -j "$(nproc)"
