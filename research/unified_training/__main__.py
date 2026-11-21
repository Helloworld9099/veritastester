"""Research-only CLI; inspection does not launch a training process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_config
from .engine import inspect_legacy


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        # argparse error text can echo local locations provided by the caller.
        raise ValueError("Invalid inspection arguments")


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("operation", choices=("inspect",))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    try:
        args = parser.parse_args(argv)
        config = load_config(args.config)
        plan = inspect_legacy(config, repo_root=args.repo_root)
        result = plan.to_mapping()
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError):
        print(json.dumps({"status": "rejected", "code": "invalid_or_unsupported_inspection"}),
              file=sys.stderr)
        return 2
    return 0 if result.get("status") == "inspection_only" else 3


if __name__ == "__main__":
    raise SystemExit(main())
