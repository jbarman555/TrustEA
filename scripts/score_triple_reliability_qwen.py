#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trustea.io import load_kg
from trustea.prompts import build_incident_index, build_reliability_prompt, format_triple


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score noisy KG triple reliability with a free/open-weight Qwen model."
    )
    parser.add_argument("--data-dir", required=True, type=Path, help="Folder containing ent_ids_*, rel_ids/cleaned_rel_ids_*, and noisy_triples_*.")
    parser.add_argument("--output", type=Path, help="Default: <data-dir>/llm_reliability_scores.tsv.")
    parser.add_argument("--kg-ids", nargs="+", default=["1", "2"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-context", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, help="Debug only: score at most this many triples per KG.")
    parser.add_argument("--resume", action="store_true", help="Append after already-scored rows in the output file.")
    parser.add_argument("--use-clean-triples", action="store_true", help="Score triples_* instead of noisy_triples_*.")
    return parser.parse_args()


def load_model(model_name: str, device_map: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing LLM dependencies. Install them with:\n"
            "  pip install -r requirements-llm.txt\n"
            "Then rerun this script."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device_map,
    )
    model.eval()
    return tokenizer, model, torch


def parse_score(text: str) -> tuple[float, str]:
    match = JSON_RE.search(text)
    payload = match.group(0) if match else text
    try:
        data = json.loads(payload)
        score = float(data["score"])
        reason = str(data.get("reason", "")).replace("\t", " ").replace("\n", " ")
    except Exception:
        number = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", text)
        score = float(number.group(0)) if number else 0.5
        reason = "Could not parse strict JSON; used first numeric score."
    return min(max(score, 0.0), 1.0), reason[:300]


def generate_score(tokenizer, model, torch, prompt: str, args: argparse.Namespace) -> tuple[float, str]:
    messages = [
        {
            "role": "system",
            "content": "You are a careful knowledge-graph reliability judge. Output only JSON.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)
    new_tokens = generated[:, inputs.input_ids.shape[1] :]
    response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return parse_score(response)


def read_existing_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            kg_id = row.get("kg_id")
            if kg_id:
                counts[kg_id] = counts.get(kg_id, 0) + 1
    return counts


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output = (args.output or data_dir / "llm_reliability_scores.tsv").expanduser().resolve()
    tokenizer, model, torch = load_model(args.model, args.device_map)

    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and output.exists() else "w"
    existing_counts = read_existing_counts(output) if args.resume else {}
    with output.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if mode == "w":
            writer.writerow(["kg_id", "triple_index", "score", "head", "relation", "tail", "model", "reason"])
        for kg_id in args.kg_ids:
            kg = load_kg(data_dir, kg_id, prefer_noisy=not args.use_clean_triples)
            incident = build_incident_index(kg)
            skip = existing_counts.get(kg_id, 0)
            total = len(kg.triples) if args.limit is None else min(len(kg.triples), args.limit)
            for triple_index in range(skip, total):
                prompt = build_reliability_prompt(kg, incident, triple_index, args.max_context)
                score, reason = generate_score(tokenizer, model, torch, prompt, args)
                triple = kg.triples[triple_index]
                writer.writerow(
                    [
                        kg_id,
                        triple_index,
                        f"{score:.6f}",
                        triple.raw_head,
                        triple.relation,
                        triple.raw_tail,
                        args.model,
                        reason,
                    ]
                )
                handle.flush()
                if (triple_index + 1) % 25 == 0 or triple_index + 1 == total:
                    print(f"KG {kg_id}: scored {triple_index + 1}/{total} triples; latest {format_triple(kg, triple)} -> {score:.3f}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
