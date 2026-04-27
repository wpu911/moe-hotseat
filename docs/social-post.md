# 小红书文案：MoE HotSeat

🔥 24G 显存跑 MoE 35B Q8，34 token/s，算快吗？

我觉得：**算快，而且能干活。**

最近做了一个本地大模型优化方案：**MoE HotSeat**。

核心思路很简单：**显存不是杂物间，而是前排座位。谁最关键，谁先坐进去。**

MoE 模型里最胖的，通常不是 attention、norm、router，而是 **expert tensor**，比如：

`ffn_gate_exps`  
`ffn_up_exps`  
`ffn_down_exps`

我现在的 v0 方案是：

🔥 前 N 个 block 的 expert tensor → 放进 VRAM  
🧊 后面的 expert tensor → 留在 RAM / Host  
⚡ 让显存优先服务最值得加速的部分

注意：这还不是真正的“动态热专家”。现在不是统计哪个 expert 最热，而是先静态把前 18 个 block 里的 expert tensors 放进显存。

实测结果：

🧠 MoE 35B  
⚙️ Q8 量化  
📚 256K 上下文  
💾 24GB 显存  
🚀 文本约 34.7 token/s  
👁️ 视觉约 32.9 token/s  
💾 VRAM 占用 90%+

还有个坑：线程不是越多越好。我从 `-t 32` 测到 `-t 2`，最后发现 **`-t 4` 是甜点位**。线程太多，CPU 只是在内耗，像一群人围着一台打印机开会。

下一步：做真正的 **MoE HotSeat v1**。

📊 统计 router 调用了哪些 expert  
🔥 找出真正高频的 hot expert  
💾 热 expert 常驻 VRAM  
🧊 冷 expert 留在 RAM / Host  
🔁 必要时动态迁移

一句话总结：

**v0 是前排先上车；v1 要做到谁热谁坐前排。**

#MoE #MoEHotSeat #本地大模型 #AI部署 #llamacpp #llamaswap #Qwen #AMD显卡 #显存优化 #本地AI
