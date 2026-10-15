#!/usr/bin/env python3
"""Public entry point for the unified SkillRise/ReAct evaluator."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from eval_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
