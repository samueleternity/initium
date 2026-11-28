"""Generate deterministic symbolic state-view, transition, and logic records."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from initium.data.language.state_records import write_synthetic_state_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.count < 20:
        parser.error("count must be at least 20 to populate train, validation, and test splits")
    print(write_synthetic_state_data(args.output, count=args.count, seed=args.seed))


if __name__ == "__main__":
    main()
