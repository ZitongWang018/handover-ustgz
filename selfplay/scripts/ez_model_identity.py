#!/usr/bin/env python3
"""Print a read-only metadata inventory for one local model directory."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ez_model_identity import inspect_model_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="local model directory; hub IDs are not fetched")
    parser.add_argument("--hash-weights", action="store_true",
                        help="opt into reading every discovered weight shard for SHA256")
    parser.add_argument("--out", help="optional JSON output path (must not already exist)")
    args = parser.parse_args()
    result = inspect_model_identity(args.model, hash_weights=args.hash_weights)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        path = Path(args.out)
        if path.exists():
            parser.error(f"refusing to overwrite existing file: {path}")
        path.write_text(payload, encoding="utf-8", newline="\n")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
