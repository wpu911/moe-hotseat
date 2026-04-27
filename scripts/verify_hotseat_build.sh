#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-}"
if [[ -z "$SRC" ]]; then
  echo "Usage: $0 /path/to/llama.cpp" >&2
  exit 1
fi

cd "$SRC"

echo "===== source marker ====="
grep -nE 'MoE HotSeat|HOTSEAT_TENSOR_LAYERS' src/llama-model-loader.cpp || true

echo
echo "===== binary / shared library marker ====="
FOUND=0
for f in \
  build-hip/bin/libllama.so \
  build-hip/bin/llama-server \
  build/bin/libllama.so \
  build/bin/llama-server; do
  if [[ -f "$f" ]]; then
    echo "--- $f ---"
    if strings "$f" | grep -Ei 'MoE HotSeat|HOTSEAT_TENSOR_LAYERS|ROCm_Host' | head -20; then
      FOUND=1
    fi
  fi
done

if [[ "$FOUND" -ne 1 ]]; then
  echo "No HotSeat marker found in built artifacts." >&2
  echo "The source may be patched but the binary/shared library may not be rebuilt." >&2
  exit 2
fi

 echo "HotSeat marker found."
