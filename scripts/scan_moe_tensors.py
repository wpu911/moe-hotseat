#!/usr/bin/env python3
"""Extract MoE expert tensor placement lines from llama-server logs.

Usage:
  python3 scan_moe_tensors.py llama-server.log
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAT = re.compile(
    r"MoE HotSeat-Tensor: tensor (?P<tensor>\S+) \((?P<mib>\d+) MiB (?P<dtype>[^)]+)\) -> (?P<where>\S+)"
)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: scan_moe_tensors.py llama-server.log", file=sys.stderr)
        return 1
    p = Path(sys.argv[1])
    text = p.read_text(errors="replace")
    print("tensor,mib,dtype,where")
    for m in PAT.finditer(text):
        print(f"{m.group('tensor')},{m.group('mib')},{m.group('dtype')},{m.group('where')}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
