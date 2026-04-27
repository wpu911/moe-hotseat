MoE HotSeat v0 的做法是：
将前 N 个 transformer block 中的 MoE expert tensors 优先放入 VRAM，后面的 expert tensors 留在 RAM / Host 中。

这不是完整的动态热专家缓存，而是一个静态的 VRAM-first expert tensor placement 策略。

为什么做这个项目？

在消费级显卡上运行大 MoE 模型时，经常会遇到一个尴尬问题：

模型总参数量很大；
24GB 显存无法完整容纳全部权重；
全走 CPU / RAM 又太慢；
普通 offload 粒度不够细；
统一内存方案可能无法稳定做到 VRAM-first。

MoE HotSeat 的目标不是把整个模型强行塞进显存，而是优先把最值得加速的 expert tensors 放进显存。

当前策略：MoE HotSeat v0

当前版本采用静态策略：

HOTSEAT_TENSOR_LAYERS=18

表示：

blk.0  - blk.17 的 expert tensors -> VRAM
blk.18 - 后续 block 的 expert tensors -> RAM / Host

注意：

这不是“18 个 expert 放进显存”。
也不是“前 18 层全部放进显存”。
而是：前 18 个 transformer block 中的 packed MoE expert tensors 放进显存。

实测结果

测试环境示例：

CPU: AMD Ryzen 9 9950X
GPU: AMD Radeon RX 7900 XTX 24GB
RAM: 192GB
Backend: llama.cpp HIP / ROCm
Model: Qwen3.6 35B A3B Q8 GGUF
Context: 256K

实测表现：

Text model:
HotSeat layers: 18
Threads: -t 4
Generation speed: ~34.7 token/s

Vision model:
HotSeat layers: 16
Threads: -t 4
Generation speed: ~32.9 token/s

VRAM usage:
~90%+

这个结果说明：
在 24GB 显存的消费级 AMD GPU 上，MoE 35B Q8 + 256K 上下文可以通过 expert tensor placement 获得比较可用的本地推理速度。

与普通 -ngl 的区别

普通 -ngl 更像是按层进行 GPU offload。

MoE HotSeat 更细：

普通 -ngl: 以 transformer layer 为主要粒度；
MoE HotSeat: 针对 MoE expert tensors 做额外 placement 控制。

当前 v0 版本主要控制以下 packed expert tensors：

ffn_gate_exps
ffn_up_exps
ffn_down_exps
线程经验

实测中，CPU 线程并不是越多越好。

在当前环境中：

-t 32: 明显变慢，CPU 内耗严重
-t 16: 明显改善
-t 8 : 表现不错
-t 4 : 综合表现最好
-t 2 : CPU 更低，但略慢

最终推荐：

-t 4

线程过多时，CPU 调度、内存带宽竞争和 Host tensor 访问可能会拖慢整体推理。

llama-swap 示例

文本模型入口：

qwen36-35b-a3b-v2-q8:256k:
  ttl: 1200
  env:
    - "LD_LIBRARY_PATH=/app/share/llama_box/src/llama.cpp/build-hip/bin:/opt/rocm/lib"
    - "HIP_VISIBLE_DEVICES=0"
    - "HSA_OVERRIDE_GFX_VERSION=11.0.0"
    - "HOTSEAT_TENSOR_LAYERS=18"
  cmd: >
    /app/share/llama_box/src/llama.cpp/build-hip/bin/llama-server
    --host 127.0.0.1
    --port ${PORT}
    --jinja
    -m /app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/Qwen3.6-35B-A3B-abliterated-v2.Q8_0.gguf
    -c 262144
    -ngl 999
    -t 4
    -np 1
    -b 256
    -ub 64
    --cache-ram 0
    --no-mmap
    --mlock
    --verbose
  checkEndpoint: /health

视觉模型入口：

qwen36-vision:256k:
  ttl: 1200
  env:
    - "LD_LIBRARY_PATH=/app/share/llama_box/src/llama.cpp/build-hip/bin:/opt/rocm/lib"
    - "HIP_VISIBLE_DEVICES=0"
    - "HSA_OVERRIDE_GFX_VERSION=11.0.0"
    - "HOTSEAT_TENSOR_LAYERS=16"
  cmd: >
    /app/share/llama_box/src/llama.cpp/build-hip/bin/llama-server
    --host 127.0.0.1
    --port ${PORT}
    --jinja
    -m /app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/Qwen3.6-35B-A3B-abliterated-v2.Q8_0.gguf
    --mmproj /app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/mmproj-BF16.gguf
    -c 262144
    -ngl 999
    -t 4
    -np 1
    -b 256
    -ub 64
    --cache-ram 0
    --no-mmap
    --mlock
    --verbose
  checkEndpoint: /health
注意事项

当前建议不要同时启用以下统一内存变量：

HSA_XNACK=1
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1

原因是当前 HotSeat v0 的目标是 VRAM-first placement。
统一内存可能会让 VRAM 与 RAM 的边界变得模糊，导致显存使用不符合预期。

这不是说统一内存无用，而是它属于另一条路线。
后续可以考虑做 VRAM-first + UMA fallback，但不建议在 v0 阶段混在一起测试。

下一步计划：MoE HotSeat v1

v0 是静态策略：

前 N 个 block 的 expert tensors 进 VRAM

v1 计划变成真正的动态 HotSeat：

统计 router 实际调用了哪些 expert
记录 layer_id + expert_id 命中次数
找出真正高频的 hot experts
让 hot experts 常驻 VRAM
让 cold experts 留在 RAM / Host
必要时进行动态迁移和调度

一句话：

v0 是前排先上车；
v1 要做到谁热谁坐前排。

项目状态

当前项目仍处于实验阶段。
代码和补丁需要根据具体 llama.cpp 版本适配。
欢迎测试、修改和提交 issue。

English
Overview

MoE HotSeat is an experimental optimization strategy for local MoE model inference.

The core idea is simple:

VRAM is not a storage room.
It is a front-row seat.
The most important tensors should sit there first.

In MoE models, the largest tensors are often not attention, norm, or router weights, but packed expert tensors, such as:

ffn_gate_exps
ffn_up_exps
ffn_down_exps

MoE HotSeat v0 places the expert tensors of the first N transformer blocks into VRAM, while keeping later expert tensors in RAM / Host memory.

This is not yet a fully dynamic hot expert cache.
It is a static VRAM-first expert tensor placement strategy.

Why this project?

When running large MoE models on consumer GPUs, we often face a painful tradeoff:

The total model size is large;
24GB VRAM cannot hold everything comfortably;
CPU / RAM-only inference is too slow;
Traditional layer-level offload is not fine-grained enough;
Unified memory may not reliably enforce VRAM-first placement.

MoE HotSeat does not try to force the entire model into VRAM.
Instead, it prioritizes the most valuable expert tensors.

Current Strategy: MoE HotSeat v0

Example:

HOTSEAT_TENSOR_LAYERS=18

This means:

Expert tensors in blk.0  - blk.17 -> VRAM
Expert tensors in blk.18 and later -> RAM / Host

Important clarification:

This does not mean “18 experts are placed in VRAM”.
It also does not mean “the first 18 full transformer layers are placed in VRAM”.

It means:
the packed MoE expert tensors in the first 18 transformer blocks are placed in VRAM.

Benchmark Example

Test environment:

CPU: AMD Ryzen 9 9950X
GPU: AMD Radeon RX 7900 XTX 24GB
RAM: 192GB
Backend: llama.cpp HIP / ROCm
Model: Qwen3.6 35B A3B Q8 GGUF
Context: 256K

Observed results:

Text model:
HotSeat layers: 18
Threads: -t 4
Generation speed: ~34.7 token/s

Vision model:
HotSeat layers: 16
Threads: -t 4
Generation speed: ~32.9 token/s

VRAM usage:
~90%+

This shows that a 24GB consumer AMD GPU can run a MoE 35B Q8 model with a 256K context at practical speeds when expert tensor placement is handled carefully.

Difference from regular -ngl

Regular -ngl is mostly layer-level GPU offload.

MoE HotSeat is more fine-grained:

Regular -ngl: mainly transformer-layer granularity;
MoE HotSeat: additional placement control for packed MoE expert tensors.

Current v0 focuses on:

ffn_gate_exps
ffn_up_exps
ffn_down_exps
Thread Tuning

More CPU threads are not always better.

In this test setup:

-t 32: much slower, heavy CPU contention
-t 16: significantly better
-t 8 : good
-t 4 : best balanced result
-t 2 : lower CPU usage, slightly slower

Recommended setting:

-t 4

Too many CPU threads may increase scheduling overhead, memory bandwidth contention, and Host tensor access overhead.

llama-swap Example

Text model:

qwen36-35b-a3b-v2-q8:256k:
  ttl: 1200
  env:
    - "LD_LIBRARY_PATH=/app/share/llama_box/src/llama.cpp/build-hip/bin:/opt/rocm/lib"
    - "HIP_VISIBLE_DEVICES=0"
    - "HSA_OVERRIDE_GFX_VERSION=11.0.0"
    - "HOTSEAT_TENSOR_LAYERS=18"
  cmd: >
    /app/share/llama_box/src/llama.cpp/build-hip/bin/llama-server
    --host 127.0.0.1
    --port ${PORT}
    --jinja
    -m /app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/Qwen3.6-35B-A3B-abliterated-v2.Q8_0.gguf
    -c 262144
    -ngl 999
    -t 4
    -np 1
    -b 256
    -ub 64
    --cache-ram 0
    --no-mmap
    --mlock
    --verbose
  checkEndpoint: /health

Vision model:

qwen36-vision:256k:
  ttl: 1200
  env:
    - "LD_LIBRARY_PATH=/app/share/llama_box/src/llama.cpp/build-hip/bin:/opt/rocm/lib"
    - "HIP_VISIBLE_DEVICES=0"
    - "HSA_OVERRIDE_GFX_VERSION=11.0.0"
    - "HOTSEAT_TENSOR_LAYERS=16"
  cmd: >
    /app/share/llama_box/src/llama.cpp/build-hip/bin/llama-server
    --host 127.0.0.1
    --port ${PORT}
    --jinja
    -m /app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/Qwen3.6-35B-A3B-abliterated-v2.Q8_0.gguf
    --mmproj /app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/mmproj-BF16.gguf
    -c 262144
    -ngl 999
    -t 4
    -np 1
    -b 256
    -ub 64
    --cache-ram 0
    --no-mmap
    --mlock
    --verbose
  checkEndpoint: /health
Notes

For the current v0 implementation, it is recommended not to enable:

HSA_XNACK=1
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1

The reason is that v0 focuses on VRAM-first placement.
Unified memory may blur the boundary between VRAM and RAM, making placement harder to reason about.

This does not mean unified memory is useless.
It may be useful in a future VRAM-first + UMA fallback design.

Roadmap: MoE HotSeat v1

v0 is static:

Expert tensors of the first N blocks go into VRAM.

v1 should become a real dynamic HotSeat:

Track router-selected experts
Record layer_id + expert_id hit counts
Identify truly hot experts
Keep hot experts resident in VRAM
Keep cold experts in RAM / Host
Support dynamic migration and scheduling

In one sentence:

v0 lets the front rows board first;
v1 lets the truly hot experts sit in front.

Status

This project is experimental.
The patch may need adaptation for different llama.cpp versions.
Issues and experiments are welcome.
EOF
