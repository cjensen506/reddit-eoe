# reddit-eoe

Open-ended analysis of self-reported symptoms in r/EosinophilicE posts.

Eosinophilic Esophagitis (EoE) is an allergic inflammatory condition of the esophagus. This project takes a JSONL scrape of ~10,000 posts from the [r/EosinophilicE](https://www.reddit.com/r/EosinophilicE/) subreddit and produces an aggregate picture of what symptoms users associate with their EoE — **in their own words**, not against a predefined symptom list.

## Headline result

| Symptom | % of symptomatic posts |
|---|---:|
| dysphagia | 35.4% |
| food impaction | 29.8% |
| throat tightness | 14.7% |
| chest pain | 13.0% |
| acid reflux | 11.7% |
| vomiting | 11.4% |
| stomach pain | 10.8% |
| choking | 8.6% |
| difficulty eating | 7.9% |
| nausea | 7.4% |

Across **4,361 symptomatic posts** by **2,811 distinct authors**. Full chart in [.data/charts/top_symptoms.png](.data/charts/top_symptoms.png); see also the word cloud and co-occurrence heatmap in the same directory, the per-symptom CSV at [.data/symptom_counts.csv](.data/symptom_counts.csv), and 5 verbatim grounding quotes per canonical symptom in [.data/symptom_quotes.md](.data/symptom_quotes.md).

## How it works

Three-stage pipeline using the Anthropic API. Each stage writes resumable artifacts to `.data/` and pauses for human review before the next.

### Stage 1 — Per-post symptom extraction

[`src/eoe/extract_submit.py`](src/eoe/extract_submit.py) and [`src/eoe/extract_collect.py`](src/eoe/extract_collect.py) drive the [Anthropic Message Batches API](https://platform.claude.com/docs/en/build-with-claude/batch-processing) over Sonnet 4.6. For each post, Claude extracts EoE symptoms in the author's own words plus a grounding quote, with explicit prompt rules to skip negated mentions, hypothetical/educational framing, medication side effects, and comorbidity-list bleed-through. Allowing self-reported symptoms attributed to specific named family members (e.g. a parent describing a child's symptoms) is explicit. See [`src/eoe/prompts.py`](src/eoe/prompts.py) for the system prompt.

Output: [.data/symptoms_raw.jsonl](.data/symptoms_raw.jsonl) — one row per post with `{post_id, author, created_utc, permalink, title, symptoms: [{phrase, quote}, ...]}`.

### Stage 2 — Cluster raw phrases into canonical groups

[`src/eoe/cluster.py`](src/eoe/cluster.py) does this in two passes:

- **Pass 1** sends phrases with count ≥ 2 (~1,000 phrases) to Claude in a single call to establish ~60 canonical groups (e.g. `dysphagia`, `food impaction`, `throat tightness`).
- **`--fixup`** patches Pass 1 output: targeted LLM mapping for any high-count phrases the model missed, plus deterministic dedup of phrases the model placed in two groups (winner = higher-total group).
- **Pass 2 (`--pass2`)** chunk-maps the long-tail singletons (~6,000 phrases) onto the established canonicals. Phrases the model declines stay in `unmapped` as the genuine long tail.

Output: [.data/symptom_mapping.json](.data/symptom_mapping.json) — hand-editable, sorted by group total, every phrase carries its raw count.

A `--validate` flag reports unmapped phrases and any phrase that ended up in two groups.

### Stage 3 — Aggregate and visualize

[`src/eoe/analyze.py`](src/eoe/analyze.py) applies the mapping to per-post symptom lists (deduped within a post so `dysphagia` mentioned 4 times in one post counts once), then writes:

- **[symptom_counts.csv](.data/symptom_counts.csv)** — per-canonical post count, post share, author count, author share, plus example phrases. Both per-post and dedupe-by-author counts side by side, so prolific posters can be spotted.
- **[charts/top_symptoms.png](.data/charts/top_symptoms.png)** — top-30 horizontal bar chart.
- **[charts/wordcloud.png](.data/charts/wordcloud.png)** — word cloud sized by post count.
- **[charts/cooccurrence.png](.data/charts/cooccurrence.png)** — top-20 co-occurrence heatmap, conditional probability `P(column | row)`, diagonal masked.
- **[symptom_quotes.md](.data/symptom_quotes.md)** — 5 randomly sampled verbatim quotes per canonical symptom, with permalink and post id.

Percentages use the **symptomatic-post denominator** (the 4,361 posts that report at least one symptom), not the full 8,262 analyzed posts. The remaining ~3,900 are correctly empty: treatment-only posts, dietary advice, hypothetical questions, etc.

## Run it yourself

```sh
uv sync
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Place the scraped JSONL at .data/r_EosinophilicE_posts.jsonl

# Stage 1 — extract
uv run python -m eoe.extract_submit --limit 20 --no-batch     # dry-run on 20 posts
uv run python -m eoe.extract_submit                            # submit full corpus batch
uv run python -m eoe.extract_collect --status                  # check progress
uv run python -m eoe.extract_collect                           # poll → download when done

# Stage 2 — cluster
uv run python -m eoe.cluster                                   # Pass 1: establish canonicals
uv run python -m eoe.cluster --fixup                           # patch unmapped + dedup
uv run python -m eoe.cluster --pass2                           # absorb the long tail
uv run python -m eoe.cluster --validate                        # sanity-check the mapping

# Stage 3 — aggregate + plot
uv run python -m eoe.analyze
```

Stage 1 on Sonnet 4.6 via the Batches API is roughly **$5–7** for ~8,000 posts (50% off list pricing). Stages 2 and 3 are cents.

## Notes on methodology

- **Open-ended extraction, not a checklist.** The extraction prompt does not list canonical symptoms — Claude returns whatever the author describes. Canonicals emerge from the clustering stage.
- **Post-level dedup, author-level dedup reported side-by-side.** A post mentioning `food impaction` four times counts once; an author posting about `food impaction` ten times counts once toward the author column. Both views appear in the CSV.
- **Authors marked `[deleted]` or `None` are excluded from the author counts** rather than collapsed into one fake "anonymous" author.
- **Singletons get genuine LLM judgment, not just deterministic mapping.** Pass 2 is an LLM call, so a singleton like `food getting stuck deep in my throat` ends up in `food impaction` rather than its own group. Phrases that are genuinely different (allergic reactions, GI bleeding signs, etc.) are left in `unmapped` — ~143 phrases / 1% of mentions.
- **No time-trend chart.** Posts span ~2014–2026 but treatment availability changed dramatically over that window (Dupixent approved 2022), so a longitudinal view would conflate corpus drift with anything substantive.

## Layout

```
src/eoe/
  prompts.py            # extraction + clustering + fixup prompts
  json_parsing.py       # tolerant JSON extraction (handles fences/multi-block)
  extract_submit.py     # Stage 1a: build + submit Message Batch
  extract_collect.py    # Stage 1b: poll + download → symptoms_raw.jsonl
  cluster.py            # Stage 2: cluster phrases (Pass 1, --fixup, --pass2, --validate)
  analyze.py            # Stage 3: counts + charts + quotes
.data/                  # gitignored; pipeline inputs/outputs
```

Built with [uv](https://github.com/astral-sh/uv) and the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python).
