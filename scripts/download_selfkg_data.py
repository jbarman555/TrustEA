#!/usr/bin/env python3
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


FILES = [
    "ent_ids_1",
    "ent_ids_2",
    "triples_1",
    "triples_2",
    "cleaned_rel_ids_1",
    "cleaned_rel_ids_2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one SelfKG relation dataset folder from GitHub.")
    parser.add_argument("--dataset", default="DBP15K", choices=["DBP15K", "DWY100K"])
    parser.add_argument("--subset", default="zh_en", help="DBP15K: zh_en/fr_en/ja_en; DWY100K: dbp_wd/dbp_yg.")
    parser.add_argument("--output-root", type=Path, default=Path("data/relation"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = f"https://raw.githubusercontent.com/THUDM/SelfKG/main/data/relation/{args.dataset}/{args.subset}"
    output_dir = args.output_root / args.dataset / args.subset
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = f"{base_url}/{name}"
        target = output_dir / name
        print(f"Downloading {url} -> {target}")
        urllib.request.urlretrieve(url, target)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

