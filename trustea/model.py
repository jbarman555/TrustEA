from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .io import KgData, Triple, write_triples
from .text import l2_normalize, semantic_reliability_prior


@dataclass
class TrustEAConfig:
    dim: int = 128
    warmup_epochs: int = 8
    ea_epochs: int = 8
    alpha: float = 0.55
    prune_threshold: float = 0.25
    agreement_epsilon: float = 0.35
    propagation_rate: float = 0.65
    alignment_lr: float = 0.25
    pseudo_threshold: float = 0.62
    margin_threshold: float = 0.03
    topk: int = 2
    chunk_size: int = 2048
    seed: int = 42


@dataclass
class KgState:
    kg: KgData
    initial: np.ndarray
    z: np.ndarray
    llm: np.ndarray
    structural: np.ndarray
    fused: np.ndarray
    relation_embeddings: dict[int, np.ndarray] = field(default_factory=dict)
    refined_mask: np.ndarray | None = None


@dataclass(frozen=True)
class PseudoPair:
    left: int
    right: int
    confidence: float
    similarity: float
    margin: float
    semantic_similarity: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def robust_unit_interval(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi <= lo:
        hi = float(values.max()) + 1e-6
        lo = float(values.min())
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


class TrustEA:
    def __init__(self, left: KgData, right: KgData, left_init: np.ndarray, right_init: np.ndarray, config: TrustEAConfig):
        self.config = config
        self.left = self._build_state(left, left_init)
        self.right = self._build_state(right, right_init)
        self.history: list[dict[str, float]] = []

    def _build_state(self, kg: KgData, initial: np.ndarray) -> KgState:
        llm = semantic_reliability_prior(kg.triples, initial, kg.relation_names, self.config.seed)
        return KgState(
            kg=kg,
            initial=initial.copy(),
            z=initial.copy(),
            llm=llm,
            structural=llm.copy(),
            fused=llm.copy(),
        )

    def override_llm_scores(self, kg_id: str, scores: np.ndarray) -> None:
        state = self.left if self.left.kg.kg_id == kg_id else self.right
        if scores.shape != state.llm.shape:
            raise ValueError(f"LLM score shape for KG {kg_id} must be {state.llm.shape}, got {scores.shape}")
        state.llm = np.clip(scores.astype(np.float32), 0.0, 1.0)
        state.fused = state.llm.copy()

    def train(self) -> list[PseudoPair]:
        for epoch in range(self.config.warmup_epochs):
            left_stats = self._warmup_epoch(self.left)
            right_stats = self._warmup_epoch(self.right)
            self.history.append({
                "phase": 0.0,
                "epoch": float(epoch + 1),
                "left_kg_loss": left_stats[0],
                "right_kg_loss": right_stats[0],
                "left_consistency": left_stats[1],
                "right_consistency": right_stats[1],
            })

        self.left.refined_mask = self.left.fused >= self.config.prune_threshold
        self.right.refined_mask = self.right.fused >= self.config.prune_threshold

        pairs: list[PseudoPair] = []
        for epoch in range(self.config.ea_epochs):
            self.left.z = self._propagate(self.left, refined=True)
            self.right.z = self._propagate(self.right, refined=True)
            pairs = self.generate_pseudo_pairs()
            self._alignment_step(pairs)
            self.history.append({
                "phase": 1.0,
                "epoch": float(epoch + 1),
                "pseudo_pairs": float(len(pairs)),
                "mean_confidence": float(np.mean([p.confidence for p in pairs])) if pairs else 0.0,
            })
        return pairs

    def _warmup_epoch(self, state: KgState) -> tuple[float, float]:
        state.relation_embeddings = self._estimate_relations(state)
        residual = self._triple_residuals(state)
        residual_norm = robust_unit_interval(residual)
        state.structural = sigmoid(3.0 - 6.0 * residual_norm).astype(np.float32)
        q = np.abs(2.0 * state.llm - 1.0) * (np.abs(state.llm - state.structural) < self.config.agreement_epsilon)
        consistency = float(np.mean(q * np.square(state.structural - state.llm)))
        state.fused = (
            self.config.alpha * state.llm + (1.0 - self.config.alpha) * state.structural
        ).astype(np.float32)
        kg_loss = float(np.mean(state.fused * residual))
        state.z = self._propagate(state, refined=False)
        return kg_loss, consistency

    def _estimate_relations(self, state: KgState) -> dict[int, np.ndarray]:
        sums: dict[int, np.ndarray] = {}
        weights: dict[int, float] = {}
        for i, triple in enumerate(state.kg.triples):
            weight = float(state.fused[i])
            diff = state.z[triple.tail] - state.z[triple.head]
            if triple.relation not in sums:
                sums[triple.relation] = np.zeros(state.z.shape[1], dtype=np.float32)
                weights[triple.relation] = 0.0
            sums[triple.relation] += weight * diff
            weights[triple.relation] += weight
        return {rel: vec / max(weights[rel], 1e-6) for rel, vec in sums.items()}

    def _triple_residuals(self, state: KgState) -> np.ndarray:
        residuals = np.zeros(len(state.kg.triples), dtype=np.float32)
        zero = np.zeros(state.z.shape[1], dtype=np.float32)
        for i, triple in enumerate(state.kg.triples):
            rel = state.relation_embeddings.get(triple.relation, zero)
            error = state.z[triple.head] + rel - state.z[triple.tail]
            residuals[i] = float(np.dot(error, error))
        return residuals

    def _propagate(self, state: KgState, refined: bool) -> np.ndarray:
        triples = state.kg.triples
        active = state.refined_mask if refined and state.refined_mask is not None else None
        z = state.z
        out = z.copy()
        denom = np.ones((z.shape[0], 1), dtype=np.float32)
        zero = np.zeros(z.shape[1], dtype=np.float32)
        for i, triple in enumerate(triples):
            if active is not None and not bool(active[i]):
                continue
            weight = float(state.fused[i])
            rel = state.relation_embeddings.get(triple.relation, zero)
            out[triple.head] += weight * (z[triple.tail] - rel)
            out[triple.tail] += weight * (z[triple.head] + rel)
            denom[triple.head, 0] += weight
            denom[triple.tail, 0] += weight
        aggregated = np.tanh(out / np.maximum(denom, 1e-6))
        mixed = (1.0 - self.config.propagation_rate) * z + self.config.propagation_rate * aggregated
        return l2_normalize(mixed.astype(np.float32))

    def _top2(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = a.shape[0]
        best_idx = np.full(n, -1, dtype=np.int64)
        best = np.full(n, -np.inf, dtype=np.float32)
        second = np.full(n, -np.inf, dtype=np.float32)
        for start in range(0, n, self.config.chunk_size):
            scores = a[start : start + self.config.chunk_size] @ b.T
            if scores.shape[1] == 1:
                idx = np.zeros(scores.shape[0], dtype=np.int64)
                vals = scores[:, 0]
                sec = np.full(scores.shape[0], -np.inf, dtype=np.float32)
            else:
                part = np.argpartition(scores, -2, axis=1)[:, -2:]
                vals2 = np.take_along_axis(scores, part, axis=1)
                order = np.argsort(vals2, axis=1)[:, ::-1]
                idx = np.take_along_axis(part, order, axis=1)[:, 0]
                vals = np.take_along_axis(vals2, order, axis=1)[:, 0]
                sec = np.take_along_axis(vals2, order, axis=1)[:, 1]
            sl = slice(start, start + scores.shape[0])
            best_idx[sl] = idx
            best[sl] = vals
            second[sl] = sec
        return best_idx, best, second

    def generate_pseudo_pairs(self) -> list[PseudoPair]:
        right_for_left, sim_lr, second_lr = self._top2(self.left.z, self.right.z)
        left_for_right, _, _ = self._top2(self.right.z, self.left.z)
        semantic_best = self._semantic_for_pairs(right_for_left)
        pairs: list[PseudoPair] = []
        for left_idx, right_idx in enumerate(right_for_left):
            if right_idx < 0 or left_for_right[right_idx] != left_idx:
                continue
            margin = float(sim_lr[left_idx] - second_lr[left_idx])
            if margin < self.config.margin_threshold:
                continue
            sem = float(semantic_best[left_idx])
            sim01 = 0.5 + 0.5 * float(sim_lr[left_idx])
            margin01 = min(max(margin / 0.20, 0.0), 1.0)
            confidence = 0.55 * sim01 + 0.25 * margin01 + 0.20 * sem
            if confidence >= self.config.pseudo_threshold:
                pairs.append(PseudoPair(left_idx, int(right_idx), confidence, float(sim_lr[left_idx]), margin, sem))
        pairs.sort(key=lambda pair: pair.confidence, reverse=True)
        return pairs

    def _semantic_for_pairs(self, right_indices: np.ndarray) -> np.ndarray:
        values = np.zeros(len(right_indices), dtype=np.float32)
        valid = right_indices >= 0
        if np.any(valid):
            left_rows = self.left.initial[valid]
            right_rows = self.right.initial[right_indices[valid]]
            values[valid] = 0.5 + 0.5 * np.sum(left_rows * right_rows, axis=1)
        return values

    def _alignment_step(self, pairs: list[PseudoPair]) -> None:
        if not pairs:
            return
        left_new = self.left.z.copy()
        right_new = self.right.z.copy()
        for pair in pairs:
            c = self.config.alignment_lr * pair.confidence
            lvec = self.left.z[pair.left]
            rvec = self.right.z[pair.right]
            left_new[pair.left] = (1.0 - c) * lvec + c * rvec
            right_new[pair.right] = (1.0 - c) * rvec + c * lvec
        self.left.z = l2_normalize(left_new.astype(np.float32))
        self.right.z = l2_normalize(right_new.astype(np.float32))

    def refined_triples(self, state: KgState) -> list[Triple]:
        mask = state.refined_mask
        if mask is None:
            return list(state.kg.triples)
        return [triple for keep, triple in zip(mask, state.kg.triples) if bool(keep)]

    def save_outputs(self, output_dir: Path, pairs: list[PseudoPair]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_reliability(output_dir / f"triple_reliability_{self.left.kg.kg_id}.tsv", self.left)
        self._write_reliability(output_dir / f"triple_reliability_{self.right.kg.kg_id}.tsv", self.right)
        write_triples(output_dir / f"refined_triples_{self.left.kg.kg_id}", self.refined_triples(self.left))
        write_triples(output_dir / f"refined_triples_{self.right.kg.kg_id}", self.refined_triples(self.right))
        with (output_dir / "pseudo_alignments.tsv").open("w", encoding="utf-8") as handle:
            handle.write(
                "left_row\tright_row\tleft_entity\tright_entity\tleft_name\tright_name\t"
                "confidence\tsimilarity\tmargin\tsemantic_similarity\n"
            )
            for pair in pairs:
                left_entity = self.left.kg.entities[pair.left]
                right_entity = self.right.kg.entities[pair.right]
                handle.write(
                    f"{pair.left}\t{pair.right}\t{left_entity.original_id}\t{right_entity.original_id}\t"
                    f"{left_entity.name}\t{right_entity.name}\t"
                    f"{pair.confidence:.6f}\t{pair.similarity:.6f}\t"
                    f"{pair.margin:.6f}\t{pair.semantic_similarity:.6f}\n"
                )

    def _write_reliability(self, path: Path, state: KgState) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("line\thead\trelation\ttail\tllm_prior\tstructural\tfused\tkept\n")
            mask = state.refined_mask if state.refined_mask is not None else np.ones(len(state.kg.triples), dtype=bool)
            for i, triple in enumerate(state.kg.triples):
                handle.write(
                    f"{i + 1}\t{triple.raw_head}\t{triple.relation}\t{triple.raw_tail}\t"
                    f"{state.llm[i]:.6f}\t{state.structural[i]:.6f}\t{state.fused[i]:.6f}\t{int(mask[i])}\n"
                )
