# MoE HotSeat: Dynamic Hybrid + Arena

[中文](#中文) | [English](#english)

> Experimental llama.cpp patch set for consumer-GPU MoE inference. It combines static HotSeat placement, per-expert HotExpert caching, Dynamic-Hybrid full-layer promotion/demotion, and an optional adaptive VRAM Arena.

## 中文

### 这是什么

这个仓库把原来的 **MoE HotSeat** 从“前 N 层专家张量固定进显存”升级为一套合体运行时：

- **HotSeat**：控制 MoE packed expert tensors 的 VRAM-first placement。
- **HotExpert**：按 `layer_id + expert_id` 统计 router 命中并维护热点专家缓存。
- **Dynamic-Hybrid**：同时管理 Full-Layer bank 与 Top-64 expert bank，可在运行时提升/降级完整 MoE 层。
- **Full Shadow Bank**：完整层切换使用独立 shadow bank，切换后轮换 shadow，避免下一次 swap 覆盖正在服务的 full bank。
- **Arena**：利用真实 workload 测得的显存 low-water mark，为额外热点专家/整层预留弹性 VRAM 缓存。

目标不是把整个 MoE 模型硬塞进显存，而是让最值得加速的权重优先留在 GPU，并在热点变化时动态调整。

### 当前合体版结构

```text
Host / RAM
  └─ 全量 MoE expert tensors（Dynamic-Hybrid 模式下作为稳定后备）

GPU / VRAM
  ├─ Full-Layer banks
  ├─ Full Shadow bank
  ├─ Top-64 banks
  ├─ Dynamic expert pool
  └─ Optional Arena
```

### 关键模式

```bash
HOTSEAT_RUNTIME_MODE=dynamic-hybrid
```

在 `dynamic-hybrid` 下：

1. MoE expert tensors 保持 host-resident 后备副本；
2. runtime 拥有固定数量的 Full-Layer banks 与 Top-64 banks；
3. router 命中统计驱动专家替换；
4. 当完整层晋升条件满足时，可发生 `full_layer_swap`；
5. Arena 可额外利用安全显存余量，降低热点抖动和 PCIe 重搬运。

### Full Shadow 修复

本仓库的最终合体源码已经包含 shadow 轮换修复：

```cpp
s.full_shadow_idx = old_full;
```

完整层交换完成后，被降级的旧 full bank 会成为新的 idle shadow bank。这样下一次完整层交换不会错误覆盖刚刚晋升、正在服务的 `new_full`。

### Arena

Arena 默认不是“看见剩余显存就全吃掉”。推荐两阶段使用。

第一阶段，测量真实 workload 的显存低水位：

```bash
HOTSEAT_AUTO_RESERVE_ARENA=1
HOTSEAT_AUTO_PROFILE_DIR=/path/to/arena-profile-dir
```

跑一段真实请求后，会按模型 fingerprint 在指定目录产生 profile。建议文本、Vision、不同模型使用不同目录，避免同主 GGUF + mmproj 场景互相覆盖测量结果。

第二阶段，应用测量结果：

```bash
HOTSEAT_AUTO_RESERVE_ARENA=1
HOTSEAT_ARENA_APPLY_PROFILE=/path/to/<fingerprint>.profile.json
```

然后重新加载模型，Arena 才按测得的安全余量正式预留。

### 示例配置

Qwen3.6 35B A3B 文本版示例：

```yaml
qwen3.6-35b:256k:
  ttl: 1200
  env:
    - "LD_LIBRARY_PATH=/path/to/llama.cpp/build-hip/bin:/opt/rocm/lib"
    - "HIP_VISIBLE_DEVICES=0"
    - "HSA_OVERRIDE_GFX_VERSION=11.0.0"
    - "HOTSEAT_TENSOR_LAYERS=0"
    - "HOTSEAT_LAYER_PLAN_FILE=/path/to/hotseat/hot-layer-plan.json"
    - "HOTSEAT_RUNTIME_MODE=dynamic-hybrid"
    - "HOTSEAT_RUNTIME_PROFILE=/path/to/hotseat/hotexpert-profile.json"
    - "HOTSEAT_RUNTIME_INIT_FULL_LAYERS=1,2,0,3,11,23,10,22,15,12"
    - "HOTSEAT_FULL_LAYER_COUNT=10"
    - "HOTSEAT_FULL_SHADOW=1"
    - "HOTSEAT_AUTO_RESERVE_ARENA=1"
    - "HOTSEAT_AUTO_PROFILE_DIR=/path/to/hotseat/arena-text"
    - "HOTSEAT_RUNTIME_LOG=/path/to/hotseat/swaps.jsonl"
  cmd: >
    /path/to/llama.cpp/build-hip/bin/llama-server
    --host 127.0.0.1
    --port ${PORT}
    --jinja
    -m /path/to/model.gguf
    -c 262144
    -ngl 999
    -t 6
    -tb 16
    -np 1
    -b 2048
    -ub 1024
    --cache-ram 0
    --no-mmap
    --mlock
  checkEndpoint: /health
```

### 日志

动态专家交换常见事件：

```json
{"event":"expert_swap", ...}
```

完整层交换：

```json
{"event":"full_layer_swap", ...}
```

Arena 路径还可能出现：

```text
arena_expand
arena_retire
arena_full_swap
```

### 源码包

当前合体源码包位于：

```text
source/moe-dynamic-hybrid-arena-source-20260807.tar.gz
```

解压后源码位于 `patch/`，结构包括：

```text
patch/
├─ ggml/include/ggml-hotexpert.h
├─ ggml/src/ggml-hotexpert.cpp
├─ ggml/src/ggml-cuda/hotexpert-*.cu/.cuh
├─ ggml/src/ggml-cuda/mmq.cu
├─ ggml/src/ggml-cuda/mmvq.cu
├─ ggml/src/ggml-cuda/topk-moe.cu
├─ src/llama-model-loader.cpp
├─ src/llama-graph.cpp
└─ tools/server/server.cpp
```

这些文件是按 **HotSeat → HotExpert → Dynamic-Hybrid/Arena** 的顺序合并后的最终版本，不需要再手工把三层补丁互相覆盖。

### 状态

这是实验性 llama.cpp 补丁，不是上游官方功能。不同 llama.cpp 提交之间可能需要手工适配，尤其是 CUDA/HIP kernel、MoE graph、server 和 model-loader 发生变化时。

---

## English

### What is this?

This repository upgrades the original static **MoE HotSeat** idea into an integrated runtime for large MoE inference on memory-constrained consumer GPUs:

- **HotSeat**: VRAM-first placement for packed MoE expert tensors.
- **HotExpert**: router-hit profiling and per-expert hot cache management.
- **Dynamic-Hybrid**: runtime management of both Full-Layer banks and Top-64 expert banks, including full-layer promotion/demotion.
- **Full Shadow Bank**: a dedicated staging bank for safe full-layer swaps.
- **Arena**: an optional adaptive VRAM cache sized from the observed physical free-memory low-water mark.

The goal is not to force the entire model into VRAM. The goal is to keep the most valuable MoE weights on the GPU and adapt when the hot set changes.

### Runtime layout

```text
Host / RAM
  └─ complete MoE expert tensor backing store

GPU / VRAM
  ├─ Full-Layer banks
  ├─ Full Shadow bank
  ├─ Top-64 banks
  ├─ Dynamic expert pool
  └─ Optional Arena
```

### Dynamic-Hybrid mode

```bash
HOTSEAT_RUNTIME_MODE=dynamic-hybrid
```

In this mode, the runtime owns GPU expert banks while host memory remains the stable backing store. Router statistics drive expert replacement, and sufficiently persistent layer-level hotness can trigger `full_layer_swap` events.

### Full Shadow rotation fix

This integrated source includes the shadow rotation fix:

```cpp
s.full_shadow_idx = old_full;
```

After a full-layer swap, the demoted full bank becomes the new idle shadow bank. This prevents the next full-layer swap from overwriting the bank that was just promoted and is already serving traffic.

### Arena workflow

Arena is intentionally measured before it is applied.

Measurement phase:

```bash
HOTSEAT_AUTO_RESERVE_ARENA=1
HOTSEAT_AUTO_PROFILE_DIR=/path/to/arena-profile-dir
```

Run representative workloads and collect the generated fingerprint profile.

Application phase:

```bash
HOTSEAT_AUTO_RESERVE_ARENA=1
HOTSEAT_ARENA_APPLY_PROFILE=/path/to/<fingerprint>.profile.json
```

Reload the model after applying the profile. Use separate profile directories for text, vision, and different models when their VRAM pressure differs.

### Source bundle

The integrated source bundle is stored at:

```text
source/moe-dynamic-hybrid-arena-source-20260807.tar.gz
```

After extraction, the final implementation lives under `patch/`. It is already layered in this order:

```text
HotSeat -> HotExpert -> Dynamic-Hybrid + Arena
```

You do not need to manually overlay the three historical patch stages.

### Status

Experimental. This is not an upstream llama.cpp feature. Expect adaptation work when rebasing onto newer llama.cpp revisions, especially around model loading, MoE graph construction, CUDA/HIP kernels, and server integration.

## License

MIT. See `LICENSE`.
