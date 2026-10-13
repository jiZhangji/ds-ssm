#!/usr/bin/env python3
"""Download one ModelScope dataset file without modifying an existing environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelscope_hub.api import HubApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = HubApi().download_file(args.repo, "dataset", args.file, local_dir=output)
    print(path, flush=True)


if __name__ == "__main__":
    main()
