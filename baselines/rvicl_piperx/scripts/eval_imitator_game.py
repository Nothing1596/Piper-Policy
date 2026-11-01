#!/usr/bin/env python3
"""Single-arm Piper batch entry; the original IG/Panda evaluator is unchanged."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rvicl_ig.piper_evaluate import main
if __name__ == "__main__":
    raise SystemExit(main())
