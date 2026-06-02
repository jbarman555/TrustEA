#!/usr/bin/env python3
"""

Expected default folder layout:
    data_dir/
      triples_1
      triples_2
      ent_ids_1
      ent_ids_2

Default output layout:
    data_dir/
      noisy_triples_1
      noisy_triples_2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence


DelimiterName = Literal["auto", "tab", "comma", "space"]
NoiseMode = Literal["tail", "head", "head_or_tail"]
CountMethod = Literal["round", "floor", "ceil"]


@dataclass(frozen=True)
class TripleColumns:
    head: int
    relation: int
    tail: int


@dataclass
class ParsedLine:
    line_number: int
    raw: str
    fields: list[str] | None
    delimiter: DelimiterName


@dataclass
class TripleRecord:
    parsed_index: int
    line_number: int
    head: str
    relation: str
    tail: str


@dataclass(frozen=True)
class NoiseEdit:
    line_number: int
    changed_column: Literal["head", "tail"]
    old_value: str
    new_value: str
    old_triple: tuple[str, str, str]
    new_triple: tuple[str, str, str]


@dataclass(frozen=True)
class KgPaths:
    kg_id: str
    triple_path: Path
    entity_path: Path
    output_path: Path


def normalize_argv(argv: Sequence[str]) -> list[str]:
    """Allow convenient flags such as --30% or --50%."""
    normalized: list[str] = []
    for arg in argv:
        match = re.fullmatch(r"--(\d+(?:\.\d+)?)%", arg)
        if match:
            normalized.extend(["--noise-percent", match.group(1)])
        else:
            normalized.append(arg)
    return normalized


def parse_noise_ratio(value: str) -> float:
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise argparse.ArgumentTypeError("--noise-ratio must be between 0 and 1")
    return ratio


def parse_noise_percent(value: str) -> float:
    percent = float(value.rstrip("%"))
    if percent < 0.0 or percent > 100.0:
        raise argparse.ArgumentTypeError("--noise-percent must be between 0 and 100")
    return percent / 100.0


def parse_noise(value: str) -> float:
    cleaned = value.strip()
    if cleaned.endswith("%"):
        return parse_noise_percent(cleaned)

    number = float(cleaned)
    if 0.0 <= number <= 1.0:
        return number
    if 1.0 < number <= 100.0:
        return number / 100.0
    raise argparse.ArgumentTypeError("--noise must be a ratio in [0,1] or a percent in [0,100]")


def parse_columns(value: str) -> TripleColumns:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("columns must be formatted as head,relation,tail")
    try:
        head, relation, tail = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("columns must be zero-based integers") from exc
    if min(head, relation, tail) < 0:
        raise argparse.ArgumentTypeError("columns must be zero-based integers")
    if len({head, relation, tail}) != 3:
        raise argparse.ArgumentTypeError("head, relation, and tail columns must be different")
    return TripleColumns(head=head, relation=relation, tail=tail)


def detect_delimiter(line: str) -> DelimiterName:
    if "\t" in line:
        return "tab"
    if "," in line:
        return "comma"
    return "space"


def split_line(line: str, delimiter: DelimiterName) -> tuple[list[str], DelimiterName]:
    actual = detect_delimiter(line) if delimiter == "auto" else delimiter
    stripped = line.rstrip("\n\r")
    if actual == "tab":
        return stripped.split("\t"), actual
    if actual == "comma":
        return next(csv.reader([stripped])), actual
    return stripped.split(), actual


def join_fields(fields: Sequence[str], delimiter: DelimiterName) -> str:
    if delimiter == "tab":
        return "\t".join(fields)
    if delimiter == "comma":
        from io import StringIO

        buffer = StringIO()
        writer = csv.writer(buffer, lineterminator="")
        writer.writerow(fields)
        return buffer.getvalue()
    return " ".join(fields)


def read_entity_ids(
    path: Path,
    *,
    delimiter: DelimiterName,
    entity_id_column: int,
    comment_prefix: str,
) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or (comment_prefix and stripped.startswith(comment_prefix)):
                continue

            fields, _ = split_line(raw, delimiter)
            if len(fields) <= entity_id_column:
                raise ValueError(
                    f"{path}:{line_number} does not contain entity column {entity_id_column}"
                )

            entity_id = fields[entity_id_column]
            if entity_id not in seen:
                seen.add(entity_id)
                entities.append(entity_id)

    if len(entities) < 2:
        raise ValueError(f"{path} must contain at least two unique entity IDs")
    return entities


def read_triples(
    path: Path,
    *,
    columns: TripleColumns,
    delimiter: DelimiterName,
    has_header: bool,
    comment_prefix: str,
) -> tuple[list[ParsedLine], list[TripleRecord]]:
    parsed_lines: list[ParsedLine] = []
    triples: list[TripleRecord] = []
    max_column = max(columns.head, columns.relation, columns.tail)

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            is_header = has_header and line_number == 1
            is_comment = bool(comment_prefix) and stripped.startswith(comment_prefix)

            if not stripped or is_header or is_comment:
                parsed_lines.append(ParsedLine(line_number, raw, None, delimiter))
                continue

            fields, actual_delimiter = split_line(raw, delimiter)
            if len(fields) <= max_column:
                raise ValueError(
                    f"{path}:{line_number} has {len(fields)} columns, "
                    f"but column index {max_column} is required"
                )

            parsed_index = len(parsed_lines)
            parsed_lines.append(ParsedLine(line_number, raw, fields, actual_delimiter))
            triples.append(
                TripleRecord(
                    parsed_index=parsed_index,
                    line_number=line_number,
                    head=fields[columns.head],
                    relation=fields[columns.relation],
                    tail=fields[columns.tail],
                )
            )

    if not triples:
        raise ValueError(f"{path} does not contain any triples")
    return parsed_lines, triples


def compute_noise_count(total_triples: int, noise_ratio: float, method: CountMethod) -> int:
    raw_count = total_triples * noise_ratio
    if method == "floor":
        return math.floor(raw_count)
    if method == "ceil":
        return math.ceil(raw_count)
    return math.floor(raw_count + 0.5)


def choose_entity_replacement(
    *,
    rng: random.Random,
    entities: Sequence[str],
    old_entity: str,
    proposed_triple_builder,
    original_triples: set[tuple[str, str, str]],
    avoid_existing: bool,
    max_attempts: int,
) -> str:
    for _ in range(max_attempts):
        candidate = entities[rng.randrange(len(entities))]
        if candidate == old_entity:
            continue
        if avoid_existing and proposed_triple_builder(candidate) in original_triples:
            continue
        return candidate

    extra = " while avoiding existing triples" if avoid_existing else ""
    raise RuntimeError(
        f"could not find a replacement for entity {old_entity!r}{extra}; "
        "try increasing --max-attempts or remove --avoid-existing"
    )


def inject_noise_for_one_kg(
    *,
    parsed_lines: list[ParsedLine],
    triples: list[TripleRecord],
    entities: Sequence[str],
    columns: TripleColumns,
    noise_ratio: float,
    count_method: CountMethod,
    mode: NoiseMode,
    rng: random.Random,
    avoid_existing: bool,
    max_attempts: int,
) -> list[NoiseEdit]:
    original_triples = {(triple.head, triple.relation, triple.tail) for triple in triples}
    noise_count = compute_noise_count(len(triples), noise_ratio, count_method)
    selected_indices = set(rng.sample(range(len(triples)), noise_count))
    edits: list[NoiseEdit] = []

    for triple_index, triple in enumerate(triples):
        if triple_index not in selected_indices:
            continue

        if mode == "head_or_tail":
            changed_column: Literal["head", "tail"] = rng.choice(["head", "tail"])
        else:
            changed_column = mode

        if changed_column == "tail":
            old_value = triple.tail

            def build_proposed(candidate: str) -> tuple[str, str, str]:
                return (triple.head, triple.relation, candidate)

        else:
            old_value = triple.head

            def build_proposed(candidate: str) -> tuple[str, str, str]:
                return (candidate, triple.relation, triple.tail)

        new_value = choose_entity_replacement(
            rng=rng,
            entities=entities,
            old_entity=old_value,
            proposed_triple_builder=build_proposed,
            original_triples=original_triples,
            avoid_existing=avoid_existing,
            max_attempts=max_attempts,
        )

        parsed_line = parsed_lines[triple.parsed_index]
        if parsed_line.fields is None:
            raise RuntimeError("internal error: selected triple has no parsed fields")

        old_triple = (triple.head, triple.relation, triple.tail)
        if changed_column == "tail":
            parsed_line.fields[columns.tail] = new_value
            new_triple = (triple.head, triple.relation, new_value)
        else:
            parsed_line.fields[columns.head] = new_value
            new_triple = (new_value, triple.relation, triple.tail)

        edits.append(
            NoiseEdit(
                line_number=triple.line_number,
                changed_column=changed_column,
                old_value=old_value,
                new_value=new_value,
                old_triple=old_triple,
                new_triple=new_triple,
            )
        )

    return edits


def write_triples(parsed_lines: Iterable[ParsedLine], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for parsed_line in parsed_lines:
            if parsed_line.fields is None:
                handle.write(parsed_line.raw)
            else:
                handle.write(join_fields(parsed_line.fields, parsed_line.delimiter))
                handle.write("\n")


def write_noise_map(edits: Sequence[NoiseEdit], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "line_number",
                "changed_column",
                "old_value",
                "new_value",
                "old_head",
                "old_relation",
                "old_tail",
                "new_head",
                "new_relation",
                "new_tail",
            ]
        )
        for edit in edits:
            writer.writerow(
                [
                    edit.line_number,
                    edit.changed_column,
                    edit.old_value,
                    edit.new_value,
                    *edit.old_triple,
                    *edit.new_triple,
                ]
            )


def build_kg_paths(args: argparse.Namespace) -> list[KgPaths]:
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else data_dir
    kg_paths: list[KgPaths] = []

    for kg_id in args.kg_ids:
        kg_paths.append(
            KgPaths(
                kg_id=kg_id,
                triple_path=data_dir / f"{args.triple_prefix}{kg_id}",
                entity_path=data_dir / f"{args.entity_prefix}{kg_id}",
                output_path=output_dir / f"{args.output_prefix}{kg_id}",
            )
        )
    return kg_paths


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create noisy_triples_1/noisy_triples_2 by replacing tails in a "
            "percentage of triples for each KG independently."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Dataset folder containing triples_1, triples_2, ent_ids_1, and ent_ids_2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output folder. Default: write noisy_triples_* inside --data-dir.",
    )
    parser.add_argument(
        "--kg-ids",
        nargs="+",
        default=["1", "2"],
        help="KG suffixes to process. Default: 1 2.",
    )
    parser.add_argument(
        "--triple-prefix",
        default="triples_",
        help="Prefix for input triple files. Default: triples_.",
    )
    parser.add_argument(
        "--entity-prefix",
        default="ent_ids_",
        help="Prefix for entity ID files. Default: ent_ids_.",
    )
    parser.add_argument(
        "--output-prefix",
        default="noisy_triples_",
        help="Prefix for output noisy triple files. Default: noisy_triples_.",
    )

    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--noise-ratio",
        type=parse_noise_ratio,
        help="Noise ratio between 0 and 1, for example 0.30 or 0.50.",
    )
    noise_group.add_argument(
        "--noise-percent",
        type=parse_noise_percent,
        help="Noise percentage between 0 and 100, for example 30 or 50.",
    )
    noise_group.add_argument(
        "--noise",
        type=parse_noise,
        help="Noise as 0.30, 30, or 30%%.",
    )

    parser.add_argument(
        "--mode",
        choices=["tail", "head", "head_or_tail"],
        default="tail",
        help="Which entity to replace. Default: tail, matching the paper.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--count-method",
        choices=["round", "floor", "ceil"],
        default="round",
        help="How to convert percentage * triples into a count. Default: round.",
    )
    parser.add_argument(
        "--columns",
        type=parse_columns,
        default=TripleColumns(0, 1, 2),
        help="Zero-based triple columns as head,relation,tail. Default: 0,1,2.",
    )
    parser.add_argument(
        "--delimiter",
        choices=["auto", "tab", "comma", "space"],
        default="auto",
        help="Delimiter for triples and entity files. Default: auto.",
    )
    parser.add_argument(
        "--entity-id-column",
        type=int,
        default=0,
        help="Zero-based entity ID column in ent_ids_* files. Default: 0.",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Treat the first line of each triples_* file as a header.",
    )
    parser.add_argument(
        "--comment-prefix",
        default="#",
        help="Lines starting with this prefix are ignored/preserved. Default: #.",
    )
    parser.add_argument(
        "--avoid-existing",
        action="store_true",
        help="Avoid generating a triple that already exists in the same KG.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1000,
        help="Replacement attempts per selected triple. Default: 1000.",
    )
    parser.add_argument(
        "--save-maps",
        action="store_true",
        help="Also save noise_map_1.tsv and noise_map_2.tsv files.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        help="Optional JSON summary file. Default: print summary only.",
    )
    return parser.parse_args(normalize_argv(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    noise_ratio = args.noise_ratio
    if noise_ratio is None:
        noise_ratio = args.noise_percent
    if noise_ratio is None:
        noise_ratio = args.noise
    if noise_ratio is None:
        noise_ratio = 0.30

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"dataset folder does not exist: {data_dir}")
    if args.entity_id_column < 0:
        raise ValueError("--entity-id-column must be zero or greater")

    rng = random.Random(args.seed)
    summaries: list[dict[str, object]] = []

    for kg_paths in build_kg_paths(args):
        if not kg_paths.triple_path.exists():
            raise FileNotFoundError(f"missing triple file: {kg_paths.triple_path}")
        if not kg_paths.entity_path.exists():
            raise FileNotFoundError(f"missing entity file: {kg_paths.entity_path}")

        parsed_lines, triples = read_triples(
            kg_paths.triple_path,
            columns=args.columns,
            delimiter=args.delimiter,
            has_header=args.has_header,
            comment_prefix=args.comment_prefix,
        )
        entities = read_entity_ids(
            kg_paths.entity_path,
            delimiter=args.delimiter,
            entity_id_column=args.entity_id_column,
            comment_prefix=args.comment_prefix,
        )

        edits = inject_noise_for_one_kg(
            parsed_lines=parsed_lines,
            triples=triples,
            entities=entities,
            columns=args.columns,
            noise_ratio=noise_ratio,
            count_method=args.count_method,
            mode=args.mode,
            rng=rng,
            avoid_existing=args.avoid_existing,
            max_attempts=args.max_attempts,
        )
        write_triples(parsed_lines, kg_paths.output_path)

        map_path = None
        if args.save_maps:
            map_path = kg_paths.output_path.with_name(f"noise_map_{kg_paths.kg_id}.tsv")
            write_noise_map(edits, map_path)

        summaries.append(
            {
                "kg_id": kg_paths.kg_id,
                "input_triples": str(kg_paths.triple_path),
                "input_entities": str(kg_paths.entity_path),
                "output_triples": str(kg_paths.output_path),
                "noise_map": str(map_path) if map_path else None,
                "total_triples": len(triples),
                "requested_noise_ratio": noise_ratio,
                "requested_noise_percent": noise_ratio * 100.0,
                "noisy_triples": len(edits),
                "actual_noise_ratio": len(edits) / len(triples),
                "mode": args.mode,
                "entity_pool_size": len(entities),
            }
        )

    summary: dict[str, object] = {
        "data_dir": str(data_dir),
        "seed": args.seed,
        "count_method": args.count_method,
        "avoid_existing": args.avoid_existing,
        "kg_summaries": summaries,
    }

    if args.summary_file:
        summary_path = args.summary_file.expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
