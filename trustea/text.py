from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict

import numpy as np

from .io import Entity, Triple


TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]|[\u3040-\u30ff]|[\uac00-\ud7af]")


def label_from_uri(value: str) -> str:
    label = value.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    label = label.replace("_", " ")
    return label or value


def tokenize(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(label_from_uri(value))]


def hashed_text_embeddings(entities: list[Entity], dim: int, seed: int) -> np.ndarray:
    vectors = np.zeros((len(entities), dim), dtype=np.float32)
    salt = str(seed).encode("utf-8")
    for entity in entities:
        tokens = tokenize(entity.name)
        if not tokens:
            tokens = [str(entity.idx)]
        for token in tokens:
            pieces = [token]
            if len(token) > 3:
                pieces.extend(token[i : i + 3] for i in range(len(token) - 2))
            for piece in pieces:
                digest = hashlib.blake2b(piece.encode("utf-8") + salt, digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[entity.idx, bucket] += sign
    return l2_normalize(vectors)


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def relation_text_vectors(
    triples: list[Triple],
    relation_names: dict[int, str],
    dim: int,
    seed: int,
) -> dict[int, np.ndarray]:
    fake_entities = [
        Entity(idx=i, original_id=i, name=name) for i, name in enumerate(relation_names.values())
    ]
    rel_ids = list(relation_names.keys())
    if not fake_entities:
        return {}
    raw = hashed_text_embeddings(fake_entities, dim, seed + 7919)
    return {rel_id: raw[pos] for pos, rel_id in enumerate(rel_ids)}


def semantic_reliability_prior(
    triples: list[Triple],
    entity_vectors: np.ndarray,
    relation_names: dict[int, str],
    seed: int,
) -> np.ndarray:
    """Offline proxy for the paper's frozen-LLM prior.

    The paper asks a frozen LLM for contextual triple plausibility. For a local,
    reproducible implementation, this combines text similarity, relation
    specificity, and neighborhood support. If external LLM scores are available,
    the runner can override this vector from a TSV file.
    """
    dim = entity_vectors.shape[1]
    rel_text = relation_text_vectors(triples, relation_names, dim, seed)
    relation_counts: dict[int, int] = defaultdict(int)
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    degree: dict[int, int] = defaultdict(int)
    for triple in triples:
        relation_counts[triple.relation] += 1
        pair_counts[(triple.head, triple.tail)] += 1
        degree[triple.head] += 1
        degree[triple.tail] += 1

    raw = np.zeros(len(triples), dtype=np.float32)
    max_relation_count = max(relation_counts.values()) if relation_counts else 1
    max_degree = max(degree.values()) if degree else 1
    for i, triple in enumerate(triples):
        head = entity_vectors[triple.head]
        tail = entity_vectors[triple.tail]
        text_sim = float(np.dot(head, tail))
        text_sim = 0.5 + 0.5 * text_sim
        rel_vec = rel_text.get(triple.relation)
        if rel_vec is not None:
            rel_tail = 0.5 + 0.5 * float(np.dot(head + rel_vec, tail) / (np.linalg.norm(head + rel_vec) + 1e-12))
        else:
            rel_tail = 0.5
        support = math.log1p(pair_counts[(triple.head, triple.tail)]) / math.log1p(max(pair_counts.values()))
        specificity = 1.0 - math.log1p(relation_counts[triple.relation]) / math.log1p(max_relation_count + 1)
        hub_penalty = math.sqrt(degree[triple.tail] / max_degree)
        raw[i] = 0.45 * text_sim + 0.25 * rel_tail + 0.20 * support + 0.10 * specificity - 0.10 * hub_penalty
    return np.clip(raw, 0.05, 0.95).astype(np.float32)
