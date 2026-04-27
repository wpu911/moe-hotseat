# Troubleshooting

## VRAM only shows 2%, but CPU is busy

Likely causes:

1. The patched source was not rebuilt into `libllama.so`.
2. `llama-server` is loading an old shared library.
3. Unified memory variables are interfering with VRAM-first placement.

Check:

```bash
strings build-hip/bin/libllama.so | grep -Ei 'MoE HotSeat|HOTSEAT_TENSOR_LAYERS|ROCm_Host'
ldd build-hip/bin/llama-server | grep -Ei 'llama|ggml|hip|rocm'
```

## Source has patch but binary does not

Force rebuild:

```bash
touch src/llama-model-loader.cpp
find build-hip -type f \( -name '*llama-model-loader.cpp.o' -o -name '*llama-model-loader.cpp.o.d' \) -delete
cmake --build build-hip --target llama-server -j $(nproc)
```

## llama-swap does not show a second model entry

Some llama-swap versions may behave oddly with duplicate model paths or unusual model names. Try:

1. Use a shorter model id, e.g. `qwen36-vision:256k`.
2. Restart llama-swap after config changes.
3. Use the Web UI to confirm the model list.

## Threads are too high and speed collapses

Do not assume `-t` larger is faster.

In one tested setup:

- `-t 32`: about 8.x token/s.
- `-t 4`: about 34.7 token/s.

High CPU threads can cause scheduler overhead and memory bandwidth contention. Delightful, if your hobby is watching silicon argue with itself.
