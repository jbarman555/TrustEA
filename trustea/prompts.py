from __future__ import annotations

from collections import defaultdict

from .io import KgData, Triple
from .text import label_from_uri


def relation_label(kg: KgData, relation_id: int) -> str:
    return label_from_uri(kg.relation_names.get(relation_id, str(relation_id)))


def entity_label(kg: KgData, row_id: int) -> str:
    return label_from_uri(kg.entities[row_id].name)


def format_triple(kg: KgData, triple: Triple) -> str:
    head = entity_label(kg, triple.head)
    relation = relation_label(kg, triple.relation)
    tail = entity_label(kg, triple.tail)
    return f"({head}, {relation}, {tail})"


def build_incident_index(kg: KgData) -> dict[int, list[int]]:
    incident: dict[int, list[int]] = defaultdict(list)
    for index, triple in enumerate(kg.triples):
        incident[triple.head].append(index)
        incident[triple.tail].append(index)
    return incident


def local_context_facts(
    kg: KgData,
    incident: dict[int, list[int]],
    triple_index: int,
    max_context: int,
) -> list[str]:
    target = kg.triples[triple_index]
    facts: list[str] = []
    seen = {triple_index}
    for entity in (target.head, target.tail):
        for neighbor_index in incident.get(entity, []):
            if neighbor_index in seen:
                continue
            seen.add(neighbor_index)
            facts.append(format_triple(kg, kg.triples[neighbor_index]))
            if len(facts) >= max_context:
                return facts
    return facts


def build_reliability_prompt(
    kg: KgData,
    incident: dict[int, list[int]],
    triple_index: int,
    max_context: int,
) -> str:
    target = kg.triples[triple_index]
    context = local_context_facts(kg, incident, triple_index, max_context)
    context_text = "\n".join(f"- {fact}" for fact in context) if context else "- No local context facts available."
    return f"""You are scoring the semantic reliability of one knowledge-graph triple.

Use your general knowledge and the provided local KG context to judge whether the target relationship is likely to hold. Do not assume the target triple is correct or incorrect in advance. Treat the context as supporting evidence, but resolve the final score from semantic plausibility, entity compatibility, relation compatibility, and consistency with the local facts.

Return only compact JSON with:
{{"score": <number from 0.0 to 1.0>, "reason": "<short reason>"}}

Scoring rubric:
- 0.90-1.00: very likely true and strongly supported by knowledge or local context.
- 0.70-0.89: plausible, with no serious contradiction.
- 0.40-0.69: uncertain, generic, ambiguous, or weakly supported.
- 0.10-0.39: unlikely to hold or inconsistent with entity/relation semantics.
- 0.00-0.09: impossible or clearly contradicted.

When scoring, consider:
1. Does the head entity type fit the relation?
2. Does the tail entity type fit the relation?
3. Is the relationship factually or commonsensically plausible?
4. Does the local context support, weaken, or contradict the target triple?

Target triple:
{format_triple(kg, target)}

Raw IDs:
head={target.raw_head}
relation={target.relation}
tail={target.raw_tail}

Local context around the head and tail entities:
{context_text}
"""
