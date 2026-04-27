# MoE HotSeat v0

> 给 MoE 模型里的 packed expert tensors 安排“显存热座位”。
>
> v0 目标很朴素：**前 N 个 transformer block 的 expert tensor 进 VRAM，后面的 expert tensor 留在 Host/RAM**。

这不是一个完整的动态 hot expert cache。它是一个足够简单、可验证、可回滚的 llama.cpp 实验补丁，用来解决消费级 24GB 显存跑 MoE 大模型时，显存吃不起来、CPU/RAM 过度忙的问题。

## 这东西解决什么问题？

MoE 模型里最“肥”的通常是 expert tensors，例如：

```text
blk.N.ffn_gate_exps.weight
blk.N.ffn_up_exps.weight
blk.N.ffn_down_exps.weight
```

MoE 每个 token 只激活部分 expert，但模型文件里所有 expert 权重都要存在。24GB 显存装不下 35B Q8 + 256K 上下文 + 所有运行 buffer 时，普通 offload 粒度可能不够细。

MoE HotSeat v0 的做法是：

```text
HOTSEAT_TENSOR_LAYERS=18

blk.0  ~ blk.17 的 packed expert tensors -> GPU/VRAM
blk.18 ~ 末尾的 packed expert tensors -> CPU/Host/RAM
```

注意：这不是“18 个 expert 进显存”，也不是“前 18 层完整进显存”。更准确地说，是：

> 前 18 个 transformer block 中的 packed MoE expert tensors 进显存。

## 实测参考

测试环境之一：

- GPU: AMD Radeon RX 7900 XTX 24GB
- CPU: AMD Ryzen 9 9950X
- RAM: 192GB
- 模型: Qwen3.6 35B MoE Q8 GGUF
- 上下文: 256K
- 后端: llama.cpp HIP/ROCm
- 模型服务: llama-swap -> llama-server

实测结果：

| 场景 | HotSeat 层数 | 线程 | 速度 |
|---|---:|---:|---:|
| 文本 | 18 | `-t 4` | 约 34.7 token/s |
| 视觉 | 16 | `-t 4` | 约 32.9 token/s |
| 文本旧参数 | 18 | `-t 32` | 约 8.x token/s |

结论很粗暴：线程不是越多越好。`-t 4` 在这个环境里是甜点位。CPU 不再装作很忙，GPU 终于像个 GPU，世界短暂恢复理智。

## 快速使用

### 1. 打补丁

在 llama.cpp 根目录外执行：

```bash
python3 scripts/apply_hotseat_v0.py /path/to/llama.cpp
```

或者先 dry-run：

```bash
python3 scripts/apply_hotseat_v0.py /path/to/llama.cpp --dry-run
```

### 2. 重新编译 HIP 版 llama-server

```bash
scripts/build_hip_llama_server.sh /path/to/llama.cpp
```

### 3. 验证补丁是否进了 `libllama.so`

```bash
scripts/verify_hotseat_build.sh /path/to/llama.cpp
```

如果看到这些字符串，说明编译产物吃到补丁：

```text
MoE HotSeat-Tensor
HOTSEAT_TENSOR_LAYERS
ROCm_Host
```

### 4. 启动示例

文本模型示例：

```bash
HOTSEAT_TENSOR_LAYERS=18 \
./build-hip/bin/llama-server \
  -m /path/to/model.gguf \
  --host 127.0.0.1 \
  --port 5800 \
  --jinja \
  -c 262144 \
  -ngl 999 \
  -t 4 \
  -np 1 \
  -b 256 \
  -ub 64 \
  --cache-ram 0 \
  --no-mmap \
  --mlock \
  --verbose
```

视觉模型示例：

```bash
HOTSEAT_TENSOR_LAYERS=16 \
./build-hip/bin/llama-server \
  -m /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --host 127.0.0.1 \
  --port 5800 \
  --jinja \
  -c 262144 \
  -ngl 999 \
  -t 4 \
  -np 1 \
  -b 256 \
  -ub 64 \
  --cache-ram 0 \
  --no-mmap \
  --mlock \
  --verbose
```

## llama-swap 示例

见：

```text
llama-swap/examples/config-qwen36-hotseat.yaml
```

推荐拆成两个逻辑入口：

```text
qwen36-35b-a3b-v2-q8:256k      文本版，HotSeat 18
qwen36-vision:256k             视觉版，HotSeat 16 + mmproj
```

## 不建议同时启用统一内存

本实验目标是 VRAM-first placement。建议先不要在这个模型入口里开：

```text
HSA_XNACK=1
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
```

这两个不是永远不能用，而是会把“显存优先”实验搅成一锅粥。软件工程已经够苦了，别再给日志加胡椒粉。

## 局限性

v0 是静态策略：

```text
层号 < HOTSEAT_TENSOR_LAYERS -> VRAM
层号 >= HOTSEAT_TENSOR_LAYERS -> Host/RAM
```

它不是动态统计 router，也没有做到单个 expert 级别迁移。

## 下一步：真正的 MoE HotSeat v1

理想的 v1 应该做：

1. 统计每层 router 实际调用的 `layer_id + expert_id`。
2. 找出真正高频 hot experts。
3. 高频 expert 常驻 VRAM。
4. 低频 expert 留在 RAM/Host。
5. 必要时动态迁移。

也就是：

> v0 是前排先上车；v1 要做到谁热谁坐前排。

## 许可证

MIT。折腾归折腾，别拿它去跑核电站，机器会哭，作者也会装死。
