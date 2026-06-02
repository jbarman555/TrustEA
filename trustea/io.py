from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Entity:
    idx: int
    original_id: int
    name: str


@dataclass(frozen=True)
class Triple:
    head: int
    relation: int
    tail: int
    raw_head: int
    raw_tail: int


@dataclass
class KgData:
    kg_id: str
    entities: list[Entity]
    triples: list[Triple]
    relation_names: dict[int, str]
    triple_file: Path


@dataclass
class PairData:
    left: int
    right: int


def split_fields(line: str) -> list[str]:
    return line.strip().split()


def read_entities(path: Path) -> list[Entity]:
    entities: list[Entity] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = split_fields(line)
            if len(fields) < 1:
                continue
            try:
                idx = int(fields[0])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} has a non-integer entity id") from exc
            name = fields[1] if len(fields) > 1 else str(idx)
            entities.append(Entity(idx=len(entities), original_id=idx, name=name))
    if not entities:
        raise ValueError(f"{path} does not contain entities")
    return entities


def read_triples(path: Path, entity_to_row: dict[int, int]) -> list[Triple]:
    triples: list[Triple] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = split_fields(line)
            if len(fields) < 3:
                raise ValueError(f"{path}:{line_number} expected three columns")
            try:
                raw_head, relation, raw_tail = (int(value) for value in fields[:3])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} has non-integer triple fields") from exc
            if raw_head not in entity_to_row:
                raise ValueError(f"{path}:{line_number} head id {raw_head} is not in ent_ids")
            if raw_tail not in entity_to_row:
                raise ValueError(f"{path}:{line_number} tail id {raw_tail} is not in ent_ids")
            triples.append(
                Triple(
                    head=entity_to_row[raw_head],
                    relation=relation,
                    tail=entity_to_row[raw_tail],
                    raw_head=raw_head,
                    raw_tail=raw_tail,
                )
            )
    if not triples:
        raise ValueError(f"{path} does not contain triples")
    return triples


def read_relation_names(path: Path | None) -> dict[int, str]:
    names: dict[int, str] = {}
    if path is None or not path.exists():
        return names
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = split_fields(line)
            if len(fields) < 1:
                continue
            try:
                idx = int(fields[0])
            except ValueError:
                continue
            names[idx] = fields[1] if len(fields) > 1 else str(idx)
    return names


def find_triple_file(data_dir: Path, kg_id: str, prefer_noisy: bool) -> Path:
    noisy = data_dir / f"noisy_triples_{kg_id}"
    clean = data_dir / f"triples_{kg_id}"
    if prefer_noisy and noisy.exists():
        return noisy
    if clean.exists():
        return clean
    if noisy.exists():
        return noisy
    raise FileNotFoundError(f"missing triples_{kg_id} or noisy_triples_{kg_id} in {data_dir}")


def find_entity_file(data_dir: Path, kg_id: str) -> Path:
    for candidate in (
        data_dir / f"ent_ids_{kg_id}",
        data_dir / f"cleaned_ent_ids_{kg_id}",
        data_dir / f"id_ent_{kg_id}",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"missing ent_ids_{kg_id}, cleaned_ent_ids_{kg_id}, or id_ent_{kg_id} in {data_dir}"
    )


def load_kg(data_dir: Path, kg_id: str, prefer_noisy: bool = True) -> KgData:
    entity_file = find_entity_file(data_dir, kg_id)
    triple_file = find_triple_file(data_dir, kg_id, prefer_noisy)
    rel_file = next(
        (
            candidate
            for candidate in (
                data_dir / f"cleaned_rel_ids_{kg_id}",
                data_dir / f"rel_ids_{kg_id}",
                data_dir / f"relation_ids_{kg_id}",
            )
            if candidate.exists()
        ),
        None,
    )
    entities = read_entities(entity_file)
    entity_to_row = {entity.original_id: entity.idx for entity in entities}
    return KgData(
        kg_id=kg_id,
        entities=entities,
        triples=read_triples(triple_file, entity_to_row),
        relation_names=read_relation_names(rel_file),
        triple_file=triple_file,
    )


def read_pairs(path: Path) -> list[PairData]:
    pairs: list[PairData] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = split_fields(line)
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number} expected two columns")
            pairs.append(PairData(left=int(fields[0]), right=int(fields[1])))
    return pairs


def write_triples(path: Path, triples: Iterable[Triple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for triple in triples:
            handle.write(f"{triple.raw_head}\t{triple.relation}\t{triple.raw_tail}\n")
