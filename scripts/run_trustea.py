#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trustea.io import load_kg, read_pairs
from trustea.model import TrustEA, TrustEAConfig
from trustea.text import hashed_text_embeddings


def read_llm_scores(path: Path) -> dict[str, np.ndarray]:
    scores: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        handle.seek(0)
        if first.rstrip("\n").split("\t")[:3] == ["kg_id", "triple_index", "score"]:
            reader = csv.DictReader(handle, delimiter="\t")
            for line_number, row in enumerate(reader, start=2):
                try:
                    scores.setdefault(row["kg_id"], []).append(float(row["score"]))
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number} has an invalid score row") from exc
        else:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.strip().split()
                if len(fields) == 1:
                    kg_id, score = "1", fields[0]
                elif len(fields) >= 2:
                    kg_id, score = fields[0], fields[-1]
                else:
                    continue
                try:
                    scores.setdefault(kg_id, []).append(float(score))
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number} has an invalid score") from exc
    return {kg_id: np.asarray(values, dtype=np.float32) for kg_id, values in scores.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TrustEA on a SelfKG-style dataset folder.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Folder with ent_ids_*, triples_* and/or noisy_triples_*.")
    parser.add_argument("--output-dir", type=Path, help="Default: <data-dir>/trustea_output.")
    parser.add_argument("--kg-ids", nargs=2, default=["1", "2"], help="KG suffixes. Default: 1 2.")
    parser.add_argument("--use-clean-triples", action="store_true", help="Ignore noisy_triples_* even when present.")
    parser.add_argument("--llm-scores", type=Path, help="Optional TSV containing kg_id and score columns.")
    parser.add_argument("--reference-pairs", type=Path, help="Optional two-column file for Hits@1 evaluation.")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--ea-epochs", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--prune-threshold", type=float, default=0.25)
    parser.add_argument("--pseudo-threshold", type=float, default=0.62)
    parser.add_argument("--margin-threshold", type=float, default=0.03)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate_hits1(pairs_path: Path, predicted: set[tuple[int, int]]) -> dict[str, float]:
    gold = read_pairs(pairs_path)
    if not gold:
        return {"gold_pairs": 0.0, "hits1": 0.0}
    hits = sum((pair.left, pair.right) in predicted for pair in gold)
    return {"gold_pairs": float(len(gold)), "hits1": hits / len(gold)}


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (args.output_dir or data_dir / "trustea_output").expanduser().resolve()
    prefer_noisy = not args.use_clean_triples

    left = load_kg(data_dir, args.kg_ids[0], prefer_noisy=prefer_noisy)
    right = load_kg(data_dir, args.kg_ids[1], prefer_noisy=prefer_noisy)
    left_init = hashed_text_embeddings(left.entities, args.dim, args.seed)
    right_init = hashed_text_embeddings(right.entities, args.dim, args.seed)

    config = TrustEAConfig(
        dim=args.dim,
        warmup_epochs=args.warmup_epochs,
        ea_epochs=args.ea_epochs,
        alpha=args.alpha,
        prune_threshold=args.prune_threshold,
        pseudo_threshold=args.pseudo_threshold,
        margin_threshold=args.margin_threshold,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    model = TrustEA(left, right, left_init, right_init, config)
    if args.llm_scores:
        for kg_id, scores in read_llm_scores(args.llm_scores.expanduser().resolve()).items():
            model.override_llm_scores(kg_id, scores)

    pairs = model.train()
    model.save_outputs(output_dir, pairs)

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "triple_files": {left.kg_id: str(left.triple_file), right.kg_id: str(right.triple_file)},
        "entities": {left.kg_id: len(left.entities), right.kg_id: len(right.entities)},
        "triples": {left.kg_id: len(left.triples), right.kg_id: len(right.triples)},
        "kept_triples": {
            left.kg_id: int(np.sum(model.left.refined_mask)),
            right.kg_id: int(np.sum(model.right.refined_mask)),
        },
        "pseudo_pairs": len(pairs),
        "history": model.history,
    }
    if args.reference_pairs:
        predicted_original = {
            (left.entities[pair.left].original_id, right.entities[pair.right].original_id)
            for pair in pairs
        }
        summary["evaluation"] = evaluate_hits1(
            args.reference_pairs.expanduser().resolve(),
            predicted_original,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
