"""Stage 2: cluster raw symptom phrases into canonical groups.

Reads .data/symptoms_raw.jsonl, builds a frequency table of distinct phrases,
and asks Claude (Sonnet 4.6 by default) to group synonyms under canonical names.
Writes the result as a hand-editable .data/symptom_mapping.json.

The mapping is the single source of truth for Stage 3 aggregation, so review
and edit symptom_mapping.json before proceeding.

Usage:
  uv run python -m eoe.cluster                 # build mapping
  uv run python -m eoe.cluster --validate      # check every raw phrase is mapped
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from eoe.json_parsing import parse_model_json
from eoe.prompts import CLUSTERING_SYSTEM_PROMPT

DATA_DIR = Path(".data")
RESULTS_FILE = DATA_DIR / "symptoms_raw.jsonl"
MAPPING_FILE = DATA_DIR / "symptom_mapping.json"

CLUSTER_MODEL = "claude-sonnet-4-6"
CLUSTER_MAX_TOKENS = 16000


def load_phrase_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    with RESULTS_FILE.open() as f:
        for line in f:
            row = json.loads(line)
            for s in row.get("symptoms", []):
                phrase = (s.get("phrase") or "").strip().lower()
                if phrase:
                    counts[phrase] += 1
    return counts


def build_user_message(counts: Counter[str]) -> str:
    """Pass the phrase list as JSON for stable parsing."""
    items = [
        {"phrase": p, "count": c}
        for p, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return (
        "Group the following self-reported symptom phrases (with their occurrence "
        "counts) into canonical clusters per the system instructions.\n\n"
        f"PHRASES ({len(items)} distinct):\n```json\n{json.dumps(items, indent=2)}\n```"
    )


def request_grouping(client: anthropic.Anthropic, counts: Counter[str]) -> dict:
    response = client.messages.create(
        model=CLUSTER_MODEL,
        max_tokens=CLUSTER_MAX_TOKENS,
        system=CLUSTERING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(counts)}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "").strip()
    parsed = parse_model_json(text)
    if parsed is None or "groups" not in parsed:
        raise RuntimeError(
            f"Could not parse grouping response. Raw text:\n{text[:2000]}"
        )
    return parsed


def normalize_mapping(grouping: dict, counts: Counter[str]) -> dict:
    """Produce the on-disk mapping shape: list of {canonical, total, members}.

    Sorted by total count desc. Members are sorted by per-phrase count desc.
    """
    out_groups = []
    seen_phrases: set[str] = set()
    for g in grouping["groups"]:
        canonical = (g.get("canonical") or "").strip().lower()
        members = [(m or "").strip().lower() for m in g.get("members", [])]
        members = [m for m in members if m]
        members_with_counts = sorted(
            ((m, counts.get(m, 0)) for m in members),
            key=lambda kv: (-kv[1], kv[0]),
        )
        total = sum(c for _, c in members_with_counts)
        seen_phrases.update(m for m, _ in members_with_counts)
        out_groups.append(
            {
                "canonical": canonical,
                "total": total,
                "members": [{"phrase": m, "count": c} for m, c in members_with_counts],
            }
        )
    out_groups.sort(key=lambda g: (-g["total"], g["canonical"]))

    unmapped = [(p, c) for p, c in counts.items() if p not in seen_phrases]
    unmapped.sort(key=lambda kv: (-kv[1], kv[0]))
    return {
        "model": CLUSTER_MODEL,
        "groups": out_groups,
        "unmapped": [{"phrase": p, "count": c} for p, c in unmapped],
    }


def validate_mapping() -> int:
    if not MAPPING_FILE.exists():
        print(f"No mapping at {MAPPING_FILE}. Run without --validate first.")
        return 2
    mapping = json.loads(MAPPING_FILE.read_text())
    counts = load_phrase_counts()

    mapped: set[str] = set()
    for g in mapping["groups"]:
        for m in g["members"]:
            mapped.add(m["phrase"])

    unmapped = [(p, c) for p, c in counts.items() if p not in mapped]
    duplicate_phrase_groups: dict[str, list[str]] = {}
    seen_in_groups: dict[str, str] = {}
    for g in mapping["groups"]:
        for m in g["members"]:
            phrase = m["phrase"]
            if phrase in seen_in_groups and seen_in_groups[phrase] != g["canonical"]:
                duplicate_phrase_groups.setdefault(phrase, []).append(g["canonical"])
                duplicate_phrase_groups[phrase].append(seen_in_groups[phrase])
            seen_in_groups[phrase] = g["canonical"]

    print(f"raw distinct phrases:      {len(counts):,}")
    print(f"mapped phrases:            {len(mapped):,}")
    print(f"unmapped phrases:          {len(unmapped):,}")
    print(f"phrases in >1 group:       {len(duplicate_phrase_groups):,}")
    if unmapped:
        print("\nTop 20 unmapped:")
        for p, c in sorted(unmapped, key=lambda kv: -kv[1])[:20]:
            print(f"  {c:>4}  {p}")
    if duplicate_phrase_groups:
        print("\nDuplicate-membership phrases (need manual fix):")
        for p, groups in duplicate_phrase_groups.items():
            print(f"  {p} → {sorted(set(groups))}")
    return 0 if (not unmapped and not duplicate_phrase_groups) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate symptom_mapping.json against symptoms_raw.jsonl. "
        "Reports unmapped phrases and any phrases appearing in multiple groups.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Cluster only phrases with at least this many mentions (default: 2). "
        "Long-tail phrases below this stay in `unmapped` for a Pass 2.",
    )
    args = parser.parse_args()

    if args.validate:
        sys.exit(validate_mapping())

    load_dotenv()
    if not RESULTS_FILE.exists():
        print(f"No {RESULTS_FILE}. Run extract_collect.py first.", file=sys.stderr)
        sys.exit(2)

    full_counts = load_phrase_counts()
    print(
        f"Loaded {sum(full_counts.values()):,} symptom mentions covering "
        f"{len(full_counts):,} distinct phrases"
    )

    pass1_counts = Counter(
        {p: c for p, c in full_counts.items() if c >= args.min_count}
    )
    deferred = sum(c for p, c in full_counts.items() if c < args.min_count)
    print(
        f"Clustering Pass 1: {len(pass1_counts):,} phrases with count >= {args.min_count} "
        f"(covers {sum(pass1_counts.values()):,}/{sum(full_counts.values()):,} mentions, "
        f"{sum(pass1_counts.values())/sum(full_counts.values())*100:.1f}%). "
        f"Deferred to Pass 2: {len(full_counts) - len(pass1_counts):,} phrases / "
        f"{deferred:,} mentions."
    )

    client = anthropic.Anthropic()
    grouping = request_grouping(client, pass1_counts)
    mapping = normalize_mapping(grouping, full_counts)
    mapping["pass1_min_count"] = args.min_count
    MAPPING_FILE.write_text(json.dumps(mapping, indent=2))
    print(f"\nWrote {len(mapping['groups']):,} canonical groups to {MAPPING_FILE}")
    if mapping["unmapped"]:
        print(
            f"  {len(mapping['unmapped'])} phrases left unmapped "
            f"(includes the deferred long tail) — see the 'unmapped' field"
        )
    print("Review/edit the file before deciding on Pass 2.")


if __name__ == "__main__":
    main()
