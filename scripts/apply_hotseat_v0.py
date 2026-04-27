#!/usr/bin/env python3
"""Apply MoE HotSeat-Tensor v0 patch to llama.cpp.

This patch inserts tensor-level placement logic into:

    src/llama-model-loader.cpp

The inserted logic uses:

    HOTSEAT_TENSOR_LAYERS=N

When set to N >= 0:

    blk.0..blk.(N-1).ffn_*_exps.* -> normal preferred weight buffer
    blk.N..end.ffn_*_exps.*       -> CPU / Host weight buffer

This is not dynamic expert-level caching. It is a static prefix offload policy
for packed MoE expert tensors.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

MARKER = "// MoE HotSeat-Tensor v0:"

PATCH_BLOCK = r'''
        // MoE HotSeat-Tensor v0:
        // If HOTSEAT_TENSOR_LAYERS is set to N, force packed MoE tensors:
        //   blk.0..blk.(N-1).ffn_*_exps.* -> default layer buffer
        //   blk.N..end.ffn_*_exps.*       -> CPU/system memory
        //
        // This is tensor-level placement, not expert-level placement.
        // It is meant as the first safe step before real hot expert splitting.
        {
            static int hotseat_tensor_layers = []() -> int {
                const char * env = std::getenv("HOTSEAT_TENSOR_LAYERS");
                if (!env || !*env) {
                    return -1;
                }
                return std::atoi(env);
            }();

            if (hotseat_tensor_layers >= 0) {
                const std::string tensor_name = tn.str();

                int hotseat_layer = -1;
                bool hotseat_is_moe_exps = false;

                if (tensor_name.rfind("blk.", 0) == 0) {
                    const size_t p0 = 4;
                    const size_t p1 = tensor_name.find('.', p0);

                    if (p1 != std::string::npos && p1 > p0) {
                        bool numeric = true;

                        for (size_t i = p0; i < p1; ++i) {
                            if (tensor_name[i] < '0' || tensor_name[i] > '9') {
                                numeric = false;
                                break;
                            }
                        }

                        if (numeric) {
                            hotseat_layer = std::atoi(tensor_name.substr(p0, p1 - p0).c_str());
                            const std::string rest = tensor_name.substr(p1 + 1);

                            hotseat_is_moe_exps =
                                rest.rfind("ffn_down_exps", 0) == 0 ||
                                rest.rfind("ffn_gate_exps", 0) == 0 ||
                                rest.rfind("ffn_up_exps", 0) == 0 ||
                                rest.rfind("ffn_gate_up_exps", 0) == 0 ||
                                rest.rfind("ffn_down_chexps", 0) == 0 ||
                                rest.rfind("ffn_gate_chexps", 0) == 0 ||
                                rest.rfind("ffn_up_chexps", 0) == 0 ||
                                rest.rfind("ffn_gate_up_chexps", 0) == 0;
                        }
                    }
                }

                if (hotseat_is_moe_exps) {
                    if (hotseat_layer >= 0 && hotseat_layer < hotseat_tensor_layers) {
                        // Use the normal repeating-layer preferred buffer.
                        // With a high -ngl value this should be the GPU/device buffer.
                        buft = select_weight_buft(hparams, t_meta, op, buft_list);
                    } else {
                        // Keep remaining packed MoE tensors in CPU/system memory.
                        buft = select_weight_buft(hparams, t_meta, op, buft_list_cpu);
                    }

                    if (!buft) {
                        throw std::runtime_error(format(
                            "MoE HotSeat-Tensor failed to select buffer type for tensor %s",
                            tensor_name.c_str()));
                    }

                    LLAMA_LOG_INFO(
                        "MoE HotSeat-Tensor: tensor %s (%zu MiB %s) -> %s [HOTSEAT_TENSOR_LAYERS=%d]\n",
                        tensor_name.c_str(),
                        ggml_nbytes(t_meta) / 1024 / 1024,
                        ggml_type_name(t_meta->type),
                        ggml_backend_buft_name(buft),
                        hotseat_tensor_layers);
                }
            }
        }
'''

ANCHOR = "        ggml_backend_buffer_type_t buft = nullptr;\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("llama_cpp", help="Path to llama.cpp source root")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be changed")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a timestamped .bak file")
    args = parser.parse_args()

    root = Path(args.llama_cpp).expanduser().resolve()
    target = root / "src" / "llama-model-loader.cpp"

    if not target.is_file():
        raise SystemExit(f"Target file not found: {target}")

    text = target.read_text(encoding="utf-8")

    if MARKER in text:
        print(f"Already patched: {target}")
        return 0

    if ANCHOR not in text:
        raise SystemExit(
            "Could not find insertion anchor. The llama.cpp source may have changed.\n"
            f"Expected anchor: {ANCHOR!r}"
        )

    new_text = text.replace(ANCHOR, ANCHOR + "\n" + PATCH_BLOCK + "\n", 1)

    print(f"Target: {target}")
    print("Patch: MoE HotSeat-Tensor v0")

    if args.dry_run:
        print("Dry-run only. No files changed.")
        return 0

    if not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = target.with_suffix(target.suffix + f".moe-hotseat-v0.{stamp}.bak")
        shutil.copy2(target, backup)
        print(f"Backup: {backup}")

    target.write_text(new_text, encoding="utf-8")
    print("Patch applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
