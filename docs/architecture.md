# MoE HotSeat architecture notes

## v0: static packed expert tensor placement

MoE HotSeat v0 operates at tensor placement time inside llama.cpp model loading.

It looks for packed MoE expert tensors by name:

```text
blk.N.ffn_down_exps.*
blk.N.ffn_gate_exps.*
blk.N.ffn_up_exps.*
blk.N.ffn_gate_up_exps.*
```

If `HOTSEAT_TENSOR_LAYERS=N` is set:

- block index `< N`: select normal preferred weight buffer, usually GPU/device when `-ngl` is high.
- block index `>= N`: select CPU/Host weight buffer.

## Why this is not dynamic hot expert caching

The v0 patch does not inspect router decisions. It does not know which individual expert is used frequently at runtime. It also does not split packed expert tensors into individual experts.

It simply puts the first N blocks' packed expert tensors into the faster seat.

## Why it can still help

Every generated token traverses the block stack. The early blocks are a stable prefix path, and packed expert tensors are large. Moving a prefix of those tensors to VRAM can dramatically reduce Host/RAM pressure.

## v1 direction

A real HotSeat should:

1. Instrument MoE router selection.
2. Count `layer_id + expert_id` hits.
3. Split or index packed expert tensors per expert.
4. Keep hot experts resident in VRAM.
5. Keep cold experts in Host/RAM.
6. Optionally migrate experts based on recent access windows.

That is a different project. v0 is the safe crowbar. v1 is the actual machinery.
