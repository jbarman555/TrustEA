# TrustEA over Noisy SelfKG Data

This repo implements the TrustEA pipeline from `TrustEA.pdf` for SelfKG-style entity-alignment datasets.

The noise-injection step is intentionally separate. Use your provided `inject_kg_noise.py` first, then run TrustEA on the same data folder.

## Expected Data Folder

Copy your dataset into a folder named `Data`:

```text
Data/
  ent_ids_1                # or cleaned_ent_ids_1 / id_ent_1
  ent_ids_2                # or cleaned_ent_ids_2 / id_ent_2
  triples_1
  triples_2
  cleaned_rel_ids_1        # or rel_ids_1 / relation_ids_1
  cleaned_rel_ids_2        # or rel_ids_2 / relation_ids_2
```

SelfKG relation datasets use this layout. Entity IDs may be sparse; the loader keeps original IDs for output and uses compact internal row IDs for fast NumPy computation.

Important SelfKG note: the GitHub repository does not directly show `<SELFKG_ROOT>/data/DBP15K/fr_en` in the file tree. That folder is created after running SelfKG's `data/getdata.sh`, which downloads and unzips `DBP15K.zip`, `DWY100K.zip`, and `LaBSE.zip` from Zenodo into `<SELFKG_ROOT>/data/`. If you only browse GitHub, you may instead see smaller files under `data/relation/DBP15K/fr_en`.

Install the base dependency:

```bash
pip install -r requirements.txt
```

## Step 1: Inject Noise

Use your script exactly like this:

```bash
python inject_kg_noise.py --data-dir Data --noise-percent 30 --save-maps --seed 42
```

By default, the script samples 30% of triples in each KG and replaces the tail entity with a random entity from the same KG. It writes:

```text
Data/noisy_triples_1
Data/noisy_triples_2
Data/noise_map_1.tsv
Data/noise_map_2.tsv
```

TrustEA automatically prefers `noisy_triples_*` when they exist.

## Step 2 Optional: Score Triple Reliability with Qwen

For a free/open-weight local model, the default scorer uses:

```text
Qwen/Qwen3-4B-Instruct-2507
```

Install optional LLM dependencies:

```bash
pip install -r requirements-llm.txt
```

Then score reliability for the triples that TrustEA will use:

```bash
python scripts/score_triple_reliability_qwen.py \
  --data-dir Data \
  --output Data/llm_reliability_scores.tsv \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --max-context 8 \
  --resume
```

For a quick test only:

```bash
python scripts/score_triple_reliability_qwen.py --data-dir Data --limit 20
```

Do not pass a partial `--limit` score file into TrustEA; the final file must contain one score per triple for each KG.

## Qwen Reliability Prompt

The scorer builds one prompt per triple:

```text
You are scoring the semantic reliability of one knowledge-graph triple.

Use your general knowledge and the provided local KG context to judge whether the target relationship is likely to hold. Do not assume the target triple is correct or incorrect in advance. Treat the context as supporting evidence, but resolve the final score from semantic plausibility, entity compatibility, relation compatibility, and consistency with the local facts.

Return only compact JSON with:
{"score": <number from 0.0 to 1.0>, "reason": "<short reason>"}

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
(head_label, relation_label, tail_label)

Raw IDs:
head=<raw_head_id>
relation=<relation_id>
tail=<raw_tail_id>

Local context around the head and tail entities:
- (neighbor_head, neighbor_relation, neighbor_tail)
- ...
```

The actual implementation is in `trustea/prompts.py`.

## Step 3: Run TrustEA

With Qwen scores:

```bash
python scripts/run_trustea.py \
  --data-dir Data \
  --llm-scores Data/llm_reliability_scores.tsv \
  --warmup-epochs 8 \
  --ea-epochs 8
```

Without Qwen scores:

```bash
python scripts/run_trustea.py --data-dir Data --warmup-epochs 8 --ea-epochs 8
```

If `--llm-scores` is omitted, TrustEA uses a deterministic local semantic prior so the full algorithm still runs offline.

## Outputs

TrustEA writes to `Data/trustea_output/` by default:

```text
triple_reliability_1.tsv
triple_reliability_2.tsv
refined_triples_1
refined_triples_2
pseudo_alignments.tsv
summary.json
```

`pseudo_alignments.tsv` includes original SelfKG entity IDs and labels.

## Useful Dataset Download Helper

If you want a SelfKG example folder:

```bash
python scripts/download_selfkg_data.py --dataset DBP15K --subset zh_en --output-root .
```

This creates `DBP15K/zh_en`; rename or copy it to `Data` before following the commands above.
