# MoE HotSeat: Dynamic-Hybrid + Arena

[中文](#中文) | [English](#english)

> Experimental `llama.cpp` patch set for large Mixture-of-Experts inference on memory-constrained GPUs.  
> This repository combines **HotSeat + HotExpert + Dynamic-Hybrid + Full Shadow + optional Arena** into one integrated source tree.

> 实验性 `llama.cpp` MoE 推理优化补丁。当前合体版已经把 **HotSeat + HotExpert + Dynamic-Hybrid + Full Shadow + Arena** 合并为一套源码，不需要再按历史版本逐层覆盖补丁。

---

## 中文

### 1. 项目目标

大 MoE 模型在消费级 GPU 上有一个很现实的问题：模型总权重很大，但每个 token 实际只激活一部分 expert。传统按 Transformer 层做 `-ngl` offload 的粒度并不总是适合 MoE，而把整个模型都塞进 24GB 显存又不现实。

这个项目的目标不是“把所有东西都塞进 GPU”，而是：

1. 把最值得加速的 MoE expert 权重放进 VRAM；
2. 统计运行时真实 router 命中；
3. 让热点 expert 动态替换冷 expert；
4. 当某个 MoE 层长期足够热时，允许它晋升为完整 Full Layer；
5. 利用 Full Shadow Bank 安全完成完整层交换；
6. 可选地把稳定剩余显存做成 Arena，进一步缓存热点 expert / 完整层；
7. Host/RAM 始终保留稳定后备，不把系统绑死在一次性的静态布局上。

一句话：

> **显存不是仓库，是热数据的前排座位。谁更热，谁坐前面。**

---

### 2. 从 HotSeat 到 Dynamic-Hybrid

这个仓库经历了三层思路，当前 `patch/` 已经是最终合体结果。

#### HotSeat

最早的 HotSeat 是静态 VRAM-first placement：

```text
HOTSEAT_TENSOR_LAYERS=N
```

前 N 个 Transformer block 的 packed MoE expert tensors 优先放 VRAM，后续 expert tensors 留 Host/RAM。

它控制的主要是：

```text
ffn_gate_exps
ffn_up_exps
ffn_down_exps
ffn_gate_up_exps   # 取决于模型/llama.cpp 版本
```

#### HotExpert

HotExpert 开始记录 router 对 `layer_id + expert_id` 的真实使用次数，并维护每层的热点专家集合。这样不再只按“层号靠前”决定谁在 GPU，而是按真实访问热度调整。

#### Dynamic-Hybrid

最终 Dynamic-Hybrid 同时管理两种 GPU residency：

- **Full Layer**：整层 MoE expert tensors 完整驻留 GPU；
- **Top-N / Top-64**：非 Full 层只保留最热的一部分 expert；
- **Dynamic expert pool**：用额外 slot 做单 expert 的动态 promotion；
- **Full Shadow Bank**：给完整层 swap 提供安全中转；
- **Arena**：可选的弹性 VRAM 二级缓存。

当前核心模式：

```bash
HOTSEAT_RUNTIME_MODE=dynamic-hybrid
```

---

### 3. Runtime 内存结构

```text
Host / RAM
│
├─ 完整 GGUF / Host backing
└─ Dynamic-Hybrid 下所有 MoE expert tensors 的稳定后备副本

GPU / VRAM
│
├─ 普通 llama.cpp GPU tensors
├─ KV cache / runtime buffers
│
├─ Full-Layer banks
│   └─ 默认 10 个，可由 HOTSEAT_FULL_LAYER_COUNT 调整
│
├─ Full Shadow bank
│   └─ 专门用于普通 full_layer_swap 的 staging / rotation
│
├─ Top-N banks
│   └─ 非 Full 层的热点 expert 集合
│
├─ Dynamic expert pool
│   └─ 日志里的 pool_slot
│
└─ Optional Arena
    ├─ 额外 hot experts
    └─ 可选 arena-backed full layer
```

这几个概念不要混：

| 组件 | 作用 |
|---|---|
| Full Shadow Bank | 完整 Full Layer 交换的中转槽 |
| Dynamic expert pool | 单 expert 的固定动态槽位 |
| Arena | 根据剩余显存动态预留的额外弹性缓存 |

---

### 4. Dynamic-Hybrid 的完整层交换

完整层不会因为某个窗口突然变热就立刻搬家。源码会先积累足够统计数据，并要求候选层持续优于当前最冷 Full Layer。

当前源码默认的 Full-Layer readiness 条件：

```text
HOTSEAT_PROFILE_MIN_DECODE  = 16384
HOTSEAT_PROFILE_MIN_PREFILL = 16384
HOTSEAT_PROFILE_MIN_REQUESTS = 8
```

三个条件默认都要满足，之后才允许普通 `full_layer_swap` 进入候选判断。

普通完整层判定的关键默认值：

```text
HOTSEAT_LAYER_RATIO             = 1.25
HOTSEAT_LAYER_CONFIRM_WINDOWS   = 3
HOTSEAT_LAYER_COOLDOWN_REQUESTS = 8
```

也就是候选非 Full 层的收益分数要达到当前最冷 Full 层约 `1.25x`，并连续确认 3 个窗口，才执行完整层 promotion / demotion。

运行日志中普通完整层交换事件：

```json
{"event":"full_layer_swap", ...}
```

---

### 5. Full Shadow Bank 修复

当前合体源码包含一个很重要的 Full Shadow 轮换修复：

```cpp
s.full_shadow_idx = old_full;
```

完整层交换后：

```text
new_full        -> 正式服务晋升层
old_full        -> 被降级并成为新的 idle shadow
full_shadow_idx -> old_full
```

如果不轮换 shadow，下一次 Full-Layer swap 可能再次把数据拷进已经在服务的 `new_full` bank，存在覆盖正在使用 bank 的风险。

本仓库 `patch/ggml/src/ggml-cuda/hotexpert-runtime.cu` 已包含该修复。

---

### 6. Dynamic Expert

普通 expert 级动态替换默认以更短窗口工作。

核心默认参数：

```text
HOTSEAT_EXPERT_EVAL_TOKENS     = 4096
HOTSEAT_EXPERT_CONFIRM_WINDOWS = 2
HOTSEAT_EXPERT_RATIO           = 1.25
HOTSEAT_EXPERT_MAX_SWAP        = 8
```

当非 resident expert 的短期 EWMA 足够高、持续超过当前 resident 最冷 expert，并通过确认窗口后，会生成 `expert_swap`。

日志示例：

```json
{
  "event": "expert_swap",
  "generation": 12,
  "layer": 29,
  "expert": 123,
  "victim": 45,
  "pool_slot": 17,
  "bytes": 12345678,
  "copy_us": 1234.5
}
```

`pool_slot` 属于 Dynamic expert pool，不是 Arena。

---

### 7. Arena 是什么

Arena 是一块**根据真实 workload 的显存低水位测出来的额外 VRAM 缓存**。

它的目的不是盲目吃满显存，而是把长期稳定剩余的那部分显存变成动态缓存，减少 Host/RAM -> GPU 的重复 PCIe 搬运。

典型场景：

```text
A expert 热 -> 进 GPU
B expert 热 -> 进 GPU
A 暂时降温
A 很快又热
```

没有额外弹性容量时，A 可能被反复换出/换入。Arena 有安全余量时，可以让部分热点继续留在 VRAM，减少抖动。

---

### 8. Arena 两阶段工作流

Arena **不是只开一个变量就直接按当前瞬时空闲显存分配**。当前源码采用两阶段：

#### 阶段 1：Measure

```bash
HOTSEAT_AUTO_RESERVE_ARENA=1
HOTSEAT_AUTO_PROFILE_DIR=/path/to/profile-dir
```

当没有 `HOTSEAT_ARENA_APPLY_PROFILE` 时，runtime 会进入 measure mode：

```text
tracking low-water only, capacity off
```

它持续记录真实运行过程中物理显存最低空闲值，并写入 fingerprint-scoped profile。

默认 profile 目录如果没有显式配置，是：

```text
/app/share/openclaw_tools/hotexpert-profiles
```

推荐每个模型/模式显式拆目录，例如：

```text
Qwen Text   -> hotseat/arena-text/
Qwen Vision -> hotseat/arena-vision/
Ornith      -> hotseat/arena-ornith/
```

这是因为 Text 与 Vision 即使共用同一主 GGUF，Vision 还有 `mmproj` 显存压力，不能理所当然共用同一 Arena low-water 结果。

#### 阶段 2：Apply

测量完成后：

```bash
HOTSEAT_AUTO_RESERVE_ARENA=1
HOTSEAT_ARENA_APPLY_PROFILE=/path/to/<fingerprint>.profile.json
```

重新 unload/load 模型后，Arena 才真正按历史 low-water 计算容量。

当前源码默认 sizing 参数：

```text
HOTSEAT_ARENA_TARGET_FREE_MB = 800
HOTSEAT_ARENA_JITTER_MB      = 256
HOTSEAT_ARENA_GROW_STEP_MB   = 256
HOTSEAT_ARENA_PAGE_MB        = 2
```

逻辑近似为：

```text
可分配 Arena
≈ measured_low_water
  - target_free
  - jitter
```

然后按 grow step 向下取整。低于约 256 MiB 时不启用 Arena。

这意味着 Arena 会主动给后续 Prefill、KV、HIP workspace 等波动留下缓冲，不是把显存压到 0 MiB 再祈祷。

---

### 9. Arena 动态事件

Arena 生效后可能出现：

```text
arena_expand
arena_retire
```

如果另外显式打开：

```bash
HOTSEAT_ARENA_FULL_LAYER_ENABLE=1
```

还允许 Arena 走独立的完整层 promotion 路径，日志可能出现：

```text
arena_full_swap
```

Arena Full Layer 和普通 Full Shadow `full_layer_swap` 是两条不同路径。

---

### 10. 推荐目录结构

```text
moe-hotseat/
├─ README.md
├─ LICENSE
├─ MANIFEST.md
├─ examples/
│  ├─ arena-env.example
│  └─ llama-swap-dynamic-hybrid-arena.yaml
└─ patch/
   ├─ ggml/
   │  ├─ include/
   │  │  └─ ggml-hotexpert.h
   │  └─ src/
   │     ├─ CMakeLists.txt
   │     ├─ ggml-hotexpert.cpp
   │     └─ ggml-cuda/
   │        ├─ common.cuh
   │        ├─ ggml-cuda.cu
   │        ├─ hotexpert-arena.cu
   │        ├─ hotexpert-arena.cuh
   │        ├─ hotexpert-cache.cu
   │        ├─ hotexpert-cache.cuh
   │        ├─ hotexpert-descriptor.cu
   │        ├─ hotexpert-descriptor.cuh
   │        ├─ hotexpert-planner.cu
   │        ├─ hotexpert-planner.cuh
   │        ├─ hotexpert-profiler.cu
   │        ├─ hotexpert-profiler.cuh
   │        ├─ hotexpert-runtime.cu
   │        ├─ hotexpert-runtime.cuh
   │        ├─ mmq.cu
   │        ├─ mmvq.cu
   │        └─ topk-moe.cu
   ├─ src/
   │  ├─ llama-graph.cpp
   │  └─ llama-model-loader.cpp
   └─ tools/server/
      └─ server.cpp
```

`patch/` 是已经按下面顺序合并后的最终源码：

```text
HotSeat -> HotExpert -> Dynamic-Hybrid + Arena -> Full Shadow rotation fix
```

不需要再手工把历史三套源码互相覆盖。

---

### 11. 如何套到 llama.cpp

> 注意：这是实验补丁，不保证对任意 upstream `llama.cpp` commit 无冲突。建议先复制一份源码树再操作。

假设：

```text
/path/to/llama.cpp
/path/to/moe-hotseat
```

可以把最终 patch tree 覆盖到目标 llama.cpp：

```bash
cp -a /path/to/moe-hotseat/patch/. /path/to/llama.cpp/
```

然后重新配置/编译 HIP 版本。一个常见示例是：

```bash
cd /path/to/llama.cpp
cmake -B build-hip \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j"$(nproc)" --target llama-server llama-cli
```

如果你的上游版本已经改变了 MoE graph、CUDA/HIP kernel、model loader 或 server，应该按 diff 手工 rebase，而不是闭眼覆盖。软件最喜欢在“应该没事”之后安排节目。

---

### 12. 当前验证环境

当前这套合体源码主要在以下环境验证：

```text
CPU     AMD Ryzen 9 9950X
RAM     ~192 GB
GPU     AMD Radeon RX 7900 XTX 24 GB
OS      Ubuntu
Backend llama.cpp HIP / ROCm
Serving llama-swap -> llama-server
Context 256K
```

主要验证模型：

```text
Qwen3.6 35B A3B Q8_0
Qwen3.6 35B A3B Q8_0 + mmproj Vision
Huihui Ornith 35B A3B Q8_0
```

这里的配置是针对 24GB VRAM 调出来的示例，不应当被当成所有 GPU 的最佳值。

---

## 13. llama-swap 完整启动参数

完整可复制配置同时放在：

```text
examples/llama-swap-dynamic-hybrid-arena.yaml
```

下面是当前实际使用的三个模型配置。这里先处于 **Arena Measure 阶段**，所以还没有写 `HOTSEAT_ARENA_APPLY_PROFILE`。

### 13.1 Qwen3.6 35B A3B Text, 256K

```yaml
qwen3.6-35b:256k:
  ttl: 1200
  env:
    - "LD_LIBRARY_PATH=/app/share/llama_box/src/llama.cpp-b10235-hotexpert/build-hip/bin:/opt/rocm/lib"
    - "HIP_VISIBLE_DEVICES=0"
    - "HSA_OVERRIDE_GFX_VERSION=11.0.0"
    - "HOTSEAT_TENSOR_LAYERS=0"
    - "HOTSEAT_LAYER_PLAN_FILE=/app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/hotseat/qwen36-hotseat-hybrid-10.json"
    - "HOTSEAT_RUNTIME_MODE=dynamic-hybrid"
    - "HOTSEAT_RUNTIME_PROFILE=/app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/hotseat/qwen36-hotexpert-profile-1024.json"
    - "HOTSEAT_RUNTIME_INIT_FULL_LAYERS=1,2,0,3,11,23,10,22,15,12"
    - "HOTSEAT_FULL_LAYER_COUNT=10"
    - "HOTSEAT_FULL_SHADOW=1"
    - "HOTSEAT_AUTO_RESERVE_ARENA=1"
    - "HOTSEAT_AUTO_PROFILE_DIR=/app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/hotseat/arena-text"
    - "HOTSEAT_RUNTIME_LOG=/app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/hotseat/swaps.jsonl"
  cmd: >
    /app/share/llama_box/src/llama.cpp-b10235-hotexpert/build-hip/bin/llama-server
    --host 127.0.0.1
    --port ${PORT}
    --jinja
    -m /app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/Qwen3.6-35B-A3B-abliterated-v2.Q8_0.gguf
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
  cmdStop: |
    bash -lc 'pkill -f "llama-server.*--port ${PORT}" || true'
  checkEndpoint: /health
```

### 13.2 Qwen3.6 35B A3B Vision, 256K

Vision 使用同一主 GGUF，但额外挂 `mmproj-BF16.gguf`，为了显存余量当前使用 9 个初始 Full Layer，而不是文本版的 10 个。

```yaml
qwen3.6-35b-vision:256k:
  ttl: 1200
  env:
    - "LD_LIBRARY_PATH=/app/share/llama_box/src/llama.cpp-b10235-hotexpert/build-hip/bin:/opt/rocm/lib"
    - "HIP_VISIBLE_DEVICES=0"
    - "HSA_OVERRIDE_GFX_VERSION=11.0.0"
    - "HOTSEAT_TENSOR_LAYERS=0"
    - "HOTSEAT_LAYER_PLAN_FILE=/app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/hotseat/qwen36-hotseat-hybrid-10.json"
    - "HOTSEAT_RUNTIME_MODE=dynamic-hybrid"
    - "HOTSEAT_RUNTIME_PROFILE=/app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/hotseat/qwen36-hotexpert-profile-1024.json"
    - "HOTSEAT_RUNTIME_INIT_FULL_LAYERS=1,2,0,3,11,23,10,22,15"
    - "HOTSEAT_FULL_LAYER_COUNT=9"
    - "HOTSEAT_FULL_SHADOW=1"
    - "HOTSEAT_AUTO_RESERVE_ARENA=1"
    - "HOTSEAT_AUTO_PROFILE_DIR=/app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/hotseat/arena-vision"
    - "HOTSEAT_RUNTIME_LOG=/app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/hotseat/swaps-vision.jsonl"
  cmd: >
    /app/share/llama_box/src/llama.cpp-b10235-hotexpert/build-hip/bin/llama-server
    --host 127.0.0.1
    --port ${PORT}
    --jinja
    -m /app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/Qwen3.6-35B-A3B-abliterated-v2.Q8_0.gguf
    --mmproj /app/share/llm/Qwen3.6-35B-A3B-abliterated-v2-Q8_0/mmproj-BF16.gguf
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
  cmdStop: |
    bash -lc 'pkill -f "llama-server.*--port ${PORT}" || true'
  checkEndpoint: /health
```

### 13.3 Huihui Ornith 35B A3B, 256K

```yaml
ornith-35b:256k:
  ttl: 1200
  env:
    - "LD_LIBRARY_PATH=/app/share/llama_box/src/llama.cpp-b10235-hotexpert/build-hip/bin:/opt/rocm/lib"
    - "HIP_VISIBLE_DEVICES=0"
    - "HSA_OVERRIDE_GFX_VERSION=11.0.0"
    - "HOTSEAT_TENSOR_LAYERS=0"
    - "HOTSEAT_LAYER_PLAN_FILE=/app/share/llm/Huihui-Ornith-1.0-35B-abliterated-GGUF/hotseat/qwen36-hotseat-hybrid-10.json"
    - "HOTSEAT_RUNTIME_MODE=dynamic-hybrid"
    - "HOTSEAT_RUNTIME_PROFILE=/app/share/llm/Huihui-Ornith-1.0-35B-abliterated-GGUF/hotseat/qwen36-hotexpert-profile-1024.json"
    - "HOTSEAT_RUNTIME_INIT_FULL_LAYERS=1,2,0,3,11,23,10,22,15,12"
    - "HOTSEAT_FULL_LAYER_COUNT=10"
    - "HOTSEAT_FULL_SHADOW=1"
    - "HOTSEAT_AUTO_RESERVE_ARENA=1"
    - "HOTSEAT_AUTO_PROFILE_DIR=/app/share/llm/Huihui-Ornith-1.0-35B-abliterated-GGUF/hotseat/arena-ornith"
    - "HOTSEAT_RUNTIME_LOG=/app/share/llm/Huihui-Ornith-1.0-35B-abliterated-GGUF/hotseat/swaps.jsonl"
  cmd: >
    /app/share/llama_box/src/llama.cpp-b10235-hotexpert/build-hip/bin/llama-server
    --host 127.0.0.1
    --port ${PORT}
    --jinja
    -m /app/share/llm/Huihui-Ornith-1.0-35B-abliterated-GGUF/ornith-1.0-35b-Q8_0.gguf
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
  cmdStop: |
    bash -lc 'pkill -f "llama-server.*--port ${PORT}" || true'
  checkEndpoint: /health
```

### 13.4 Arena Apply 阶段怎么加

每个模型先跑出自己的 profile。之后在对应 `env:` 中追加：

```yaml
- "HOTSEAT_ARENA_APPLY_PROFILE=/对应模型/arena目录/<fingerprint>.profile.json"
```

然后 unload -> load。

建议继续保持三套目录分离：

```text
arena-text
arena-vision
arena-ornith
```

---

### 14. 常用环境变量

下面这些来自当前合体源码，默认值以当前源码实现为准。

| 环境变量 | 默认/示例 | 作用 |
|---|---:|---|
| `HOTSEAT_RUNTIME_MODE` | `static` | `dynamic-experts` / `dynamic-hybrid` |
| `HOTSEAT_RUNTIME_PROFILE` | 自动/可指定 | 初始 expert ranking / runtime profile |
| `HOTSEAT_RUNTIME_INIT_FULL_LAYERS` | 可指定 | Dynamic-Hybrid 初始 Full Layer 集合 |
| `HOTSEAT_FULL_LAYER_COUNT` | `10` | Full Layer bank 数量 |
| `HOTSEAT_FULL_SHADOW` | `1` | `0` 关闭 Full Shadow，默认开启 |
| `HOTSEAT_EXPERT_EVAL_TOKENS` | `4096` | expert 评估 token 周期 |
| `HOTSEAT_EXPERT_CONFIRM_WINDOWS` | `2` | expert promotion 连续确认窗口 |
| `HOTSEAT_EXPERT_RATIO` | `1.25` | expert 候选/最冷 resident 比例阈值 |
| `HOTSEAT_EXPERT_MAX_SWAP` | `8` | 每次评估最多普通 expert swap 数 |
| `HOTSEAT_PROFILE_MIN_DECODE` | `16384` | Full-Layer migration 最低累计 decode |
| `HOTSEAT_PROFILE_MIN_PREFILL` | `16384` | Full-Layer migration 最低累计 prefill |
| `HOTSEAT_PROFILE_MIN_REQUESTS` | `8` | Full-Layer migration 最低请求数 |
| `HOTSEAT_LAYER_RATIO` | `1.25` | Full-Layer promotion 比例阈值 |
| `HOTSEAT_LAYER_CONFIRM_WINDOWS` | `3` | Full-Layer 连续确认窗口 |
| `HOTSEAT_LAYER_COOLDOWN_REQUESTS` | `8` | Full swap 后冷却请求数 |
| `HOTSEAT_RUNTIME_LOG` | 可指定 | JSONL runtime 事件日志 |
| `HOTSEAT_AUTO_RESERVE_ARENA` | `0` | `1` 开启 Arena measure/apply 逻辑 |
| `HOTSEAT_AUTO_PROFILE_DIR` | `/app/share/openclaw_tools/hotexpert-profiles` | profile / plan 默认目录 |
| `HOTSEAT_ARENA_APPLY_PROFILE` | 无 | 指定上一轮 low-water profile |
| `HOTSEAT_ARENA_TARGET_FREE_MB` | `800` | Arena 后仍希望保留的物理空闲显存 |
| `HOTSEAT_ARENA_JITTER_MB` | `256` | 额外显存波动安全垫 |
| `HOTSEAT_ARENA_GROW_STEP_MB` | `256` | Arena 容量增长粒度 |
| `HOTSEAT_ARENA_PAGE_MB` | `2` | Arena page 大小 |
| `HOTSEAT_ARENA_MAX_EXPERT_CHANGES` | `16` | 单次 Arena expert 变化数量上限 |
| `HOTSEAT_ARENA_MAX_REBALANCE_MB` | `256` | 单次 Arena rebalance 字节预算 |
| `HOTSEAT_ARENA_RETIRE_FLOOR` | `0.01` | Arena cold expert retirement floor |
| `HOTSEAT_ARENA_FULL_LAYER_ENABLE` | `0` | `1` 开启 Arena-backed full-layer 路径 |

---

### 15. 如何确认真的生效

加载模型后先确认实际二进制和动态库来自 patched tree：

```bash
PID=$(pgrep -xo llama-server)
readlink -f /proc/$PID/exe
grep -E 'libggml-hip\.so|libllama\.so|libllama-server-impl\.so' /proc/$PID/maps \
  | awk '{print $6}' | sort -u
tr '\0' '\n' < /proc/$PID/environ | grep '^HOTSEAT_' | sort
```

重点日志事件：

```text
expert_swap       普通单 expert 动态替换
full_layer_swap   普通 Full Shadow 完整层交换
arena_expand      Arena 加入额外 expert
arena_retire      Arena 回收冷 expert
arena_full_swap   Arena-backed 完整层交换
```

如果只看到 `expert_swap`，并不等于 Full Layer 动态逻辑失效。普通 `full_layer_swap` 需要先满足 profile readiness、ratio、confirm windows 和 cooldown 条件。

---

### 16. 已知边界与风险

- 这是实验性 llama.cpp patch，不是 upstream 官方 feature。
- 不同 llama.cpp 版本的 graph / model loader / server / CUDA-HIP kernel 可能变化，需要 rebase。
- 24GB 卡上 Arena 不能只看启动后的瞬时 free VRAM，必须用真实 workload low-water。
- Vision 与 Text 应分开测 Arena profile。
- `HOTSEAT_ARENA_FULL_LAYER_ENABLE=1` 是额外路径，建议先把普通 Full Shadow 路径跑稳再开。
- 长上下文、大 Prefill、大 batch、Vision `mmproj` 都可能改变显存最低水位。
- 当前公开源码是实验工程代码，建议先在独立 llama.cpp tree 编译验证，不要覆盖唯一生产源码。

---

## English

### 1. Goal

Large MoE models are awkward on consumer GPUs: total parameter size is huge, while each token activates only a subset of experts. Traditional layer-level offload is often too coarse, and forcing the entire model into a 24 GB GPU is unrealistic.

This project tries to make VRAM residency adaptive instead:

1. keep the most valuable MoE expert weights in VRAM;
2. observe real router traffic;
3. promote hot experts and evict colder residents;
4. promote persistently hot MoE layers into full GPU-resident layers;
5. use a rotating Full Shadow bank for safe structural swaps;
6. optionally turn stable free VRAM into an adaptive Arena;
7. keep host memory as the stable backing store.

In short:

> **VRAM is not a warehouse. It is the front row for hot data.**

---

### 2. Evolution: HotSeat -> HotExpert -> Dynamic-Hybrid

#### HotSeat

The original HotSeat implementation was static VRAM-first placement:

```text
HOTSEAT_TENSOR_LAYERS=N
```

Packed MoE expert tensors in the first N transformer blocks were preferentially placed in VRAM, while later expert tensors remained in Host/RAM.

#### HotExpert

HotExpert adds router-hit accounting for `layer_id + expert_id` and maintains per-layer hot expert sets. GPU residency can therefore follow actual traffic rather than block order alone.

#### Dynamic-Hybrid

Dynamic-Hybrid manages multiple residency classes at once:

- Full-Layer banks;
- Top-N / Top-64 banks for non-full layers;
- a dynamic expert pool;
- one Full Shadow bank;
- an optional Arena.

Enable it with:

```bash
HOTSEAT_RUNTIME_MODE=dynamic-hybrid
```

---

### 3. Runtime memory layout

```text
Host / RAM
  └─ stable backing copy of all MoE expert tensors

GPU / VRAM
  ├─ normal llama.cpp GPU tensors
  ├─ KV / runtime buffers
  ├─ Full-Layer banks
  ├─ Full Shadow bank
  ├─ Top-N banks
  ├─ Dynamic expert pool
  └─ Optional Arena
```

Do not confuse these three mechanisms:

| Component | Purpose |
|---|---|
| Full Shadow bank | staging slot for ordinary full-layer swaps |
| Dynamic expert pool | fixed slots used by per-expert promotions |
| Arena | extra elastic VRAM capacity sized from measured low-water |

---

### 4. Full-layer promotion

The runtime does not move a full layer after a single hot window. The current source requires profile readiness first.

Default readiness thresholds:

```text
HOTSEAT_PROFILE_MIN_DECODE   = 16384
HOTSEAT_PROFILE_MIN_PREFILL  = 16384
HOTSEAT_PROFILE_MIN_REQUESTS = 8
```

Default structural decision parameters:

```text
HOTSEAT_LAYER_RATIO             = 1.25
HOTSEAT_LAYER_CONFIRM_WINDOWS   = 3
HOTSEAT_LAYER_COOLDOWN_REQUESTS = 8
```

A non-full candidate must remain sufficiently more valuable than the coldest current full layer for multiple evaluation windows before an ordinary `full_layer_swap` is committed.

---

### 5. Full Shadow rotation fix

The integrated tree includes this fix:

```cpp
s.full_shadow_idx = old_full;
```

After the swap, the demoted full bank becomes the next idle shadow. Without rotating the shadow index, a later swap could stage into the bank that was just promoted and is already serving traffic.

---

### 6. Dynamic expert swaps

Default expert-level scheduling parameters include:

```text
HOTSEAT_EXPERT_EVAL_TOKENS     = 4096
HOTSEAT_EXPERT_CONFIRM_WINDOWS = 2
HOTSEAT_EXPERT_RATIO           = 1.25
HOTSEAT_EXPERT_MAX_SWAP        = 8
```

Typical log event:

```json
{"event":"expert_swap", "layer":29, "expert":123, "victim":45, "pool_slot":17}
```

`pool_slot` belongs to the dynamic expert pool. It is not the Arena.

---

### 7. Arena

Arena turns **stable free VRAM** into an elastic second-level cache. It is deliberately sized from a previous real workload instead of from a single mid-flight `free VRAM` sample.

#### Phase 1: measure

```bash
HOTSEAT_AUTO_RESERVE_ARENA=1
HOTSEAT_AUTO_PROFILE_DIR=/path/to/profile-dir
```

Without `HOTSEAT_ARENA_APPLY_PROFILE`, the runtime tracks the physical free-memory low-water and keeps Arena capacity disabled.

Use separate profile directories for workloads with different VRAM pressure, especially text vs. vision.

#### Phase 2: apply

```bash
HOTSEAT_AUTO_RESERVE_ARENA=1
HOTSEAT_ARENA_APPLY_PROFILE=/path/to/<fingerprint>.profile.json
```

Reload the model after adding the apply profile.

Default sizing controls in the current source:

```text
HOTSEAT_ARENA_TARGET_FREE_MB = 800
HOTSEAT_ARENA_JITTER_MB      = 256
HOTSEAT_ARENA_GROW_STEP_MB   = 256
HOTSEAT_ARENA_PAGE_MB        = 2
```

The implementation roughly reserves:

```text
measured low-water - target free - jitter
```

rounded down to the grow step, and disables Arena when the remaining capacity is too small.

---

### 8. Arena events

Expected event types include:

```text
arena_expand
arena_retire
```

Optional Arena-backed full-layer promotion is separately gated by:

```bash
HOTSEAT_ARENA_FULL_LAYER_ENABLE=1
```

and may emit:

```text
arena_full_swap
```

This is a different path from the ordinary Full Shadow `full_layer_swap`.

---

### 9. Integrated source tree

The final merged implementation is expanded under `patch/` and already reflects this overlay order:

```text
HotSeat -> HotExpert -> Dynamic-Hybrid + Arena -> Full Shadow rotation fix
```

You do not need to manually overlay the three historical patch stages.

See the Chinese section above for the full file tree.

---

### 10. Applying to llama.cpp

Use a disposable or backed-up llama.cpp tree first:

```bash
cp -a /path/to/moe-hotseat/patch/. /path/to/llama.cpp/
```

Typical HIP build example:

```bash
cd /path/to/llama.cpp
cmake -B build-hip \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j"$(nproc)" --target llama-server llama-cli
```

Rebase manually when upstream changes touch model loading, MoE graph construction, HIP/CUDA kernels, or server integration.

---

### 11. Tested setup

Primary validation environment:

```text
CPU     AMD Ryzen 9 9950X
RAM     ~192 GB
GPU     AMD Radeon RX 7900 XTX 24 GB
OS      Ubuntu
Backend llama.cpp HIP / ROCm
Serving llama-swap -> llama-server
Context 256K
```

Primary models:

```text
Qwen3.6 35B A3B Q8_0
Qwen3.6 35B A3B Q8_0 + BF16 mmproj
Huihui Ornith 35B A3B Q8_0
```

These are tuning examples for a 24 GB GPU, not universal optimal settings.

---

### 12. Complete llama-swap examples

See:

```text
examples/llama-swap-dynamic-hybrid-arena.yaml
```

It contains the complete current text, vision, and Ornith startup entries, including separated Arena profile directories.

The examples are intentionally in Arena **measure mode** first. After a fingerprint profile has been collected, add:

```yaml
- "HOTSEAT_ARENA_APPLY_PROFILE=/path/to/<fingerprint>.profile.json"
```

and reload the model.

---

### 13. Verification

Confirm that the running process and libraries come from the patched build tree:

```bash
PID=$(pgrep -xo llama-server)
readlink -f /proc/$PID/exe
grep -E 'libggml-hip\.so|libllama\.so|libllama-server-impl\.so' /proc/$PID/maps \
  | awk '{print $6}' | sort -u
tr '\0' '\n' < /proc/$PID/environ | grep '^HOTSEAT_' | sort
```

Useful runtime events:

```text
expert_swap
full_layer_swap
arena_expand
arena_retire
arena_full_swap
```

Seeing expert swaps without full-layer swaps is not by itself an error. Full-layer migration has stricter readiness and confirmation requirements.

---

### 14. Status and caveats

- Experimental, not an upstream llama.cpp feature.
- Rebase work may be required on newer llama.cpp revisions.
- Arena should be sized from representative real traffic, not startup free VRAM.
- Text and vision should normally use separate Arena measurement directories.
- Long context, large prefill, batch size, HIP workspaces, and `mmproj` can all change low-water.
- Test in a separate source tree before replacing a production build.

---

## License

MIT. See `LICENSE`.

This repository contains modified files derived from `llama.cpp`; preserve applicable upstream copyright and license notices when redistributing modified upstream files.
