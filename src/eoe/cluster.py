"""Stage 2: cluster raw symptom phrases into canonical groups.

Reads .data/symptoms_raw.jsonl, builds a frequency table of distinct phrases,
and asks Claude (Sonnet 4.6 by default) to group synonyms under canonical names.
Writes the result as a hand-editable mappings/symptom_mapping.json (tracked in
git so reviewers can see and propose edits to the canonical groupings).

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
from eoe.prompts import CLUSTERING_SYSTEM_PROMPT, FIXUP_ASSIGNMENT_SYSTEM_PROMPT

DATA_DIR = Path(".data")
MAPPINGS_DIR = Path("mappings")
RESULTS_FILE = DATA_DIR / "symptoms_raw.jsonl"
MAPPING_FILE = MAPPINGS_DIR / "symptom_mapping.json"

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


def request_unmapped_assignments(
    client: anthropic.Anthropic,
    unmapped: list[dict],
    canonicals: list[str],
) -> list[dict]:
    """Ask Sonnet to map each unmapped phrase onto an existing canonical."""
    user_message = (
        f"CANONICALS ({len(canonicals)}):\n"
        f"```json\n{json.dumps(canonicals, indent=2)}\n```\n\n"
        f"UNMAPPED PHRASES ({len(unmapped)}):\n"
        f"```json\n{json.dumps(unmapped, indent=2)}\n```"
    )
    response = client.messages.create(
        model=CLUSTER_MODEL,
        max_tokens=CLUSTER_MAX_TOKENS,
        system=FIXUP_ASSIGNMENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "").strip()
    parsed = parse_model_json(text)
    if parsed is None or "assignments" not in parsed:
        raise RuntimeError(
            f"Could not parse fixup response. Raw text:\n{text[:2000]}"
        )
    return parsed["assignments"]


def apply_fixups(
    client: anthropic.Anthropic, mapping: dict, counts: Counter[str]
) -> dict:
    """Apply two surgical fixes to a Pass 1 mapping:
    1. LLM-map any count >= 2 phrases that landed in `unmapped` onto canonicals.
    2. Deterministically dedup phrases listed in two groups by keeping only the
       higher-total-count group's membership.
    """
    canonicals = [g["canonical"] for g in mapping["groups"]]
    by_canonical = {g["canonical"]: g for g in mapping["groups"]}

    # --- Step 1: LLM-map unmapped count >= 2 phrases onto canonicals
    unmapped_to_fix = [u for u in mapping["unmapped"] if u["count"] >= 2]
    print(
        f"Step 1: mapping {len(unmapped_to_fix)} unmapped count>=2 phrases "
        f"onto {len(canonicals)} canonicals..."
    )
    fixed = 0
    skipped = 0
    if unmapped_to_fix:
        assignments = request_unmapped_assignments(client, unmapped_to_fix, canonicals)
        for a in assignments:
            phrase = (a.get("phrase") or "").strip().lower()
            canonical = (a.get("canonical") or "").strip().lower()
            if not phrase:
                continue
            if canonical and canonical != "none" and canonical in by_canonical:
                by_canonical[canonical]["members"].append(
                    {"phrase": phrase, "count": counts.get(phrase, 0)}
                )
                fixed += 1
                print(f"  + {phrase}  ({counts.get(phrase, 0)}) → {canonical}")
            else:
                skipped += 1
                print(f"  ? {phrase}  → '{canonical or 'none'}' (left unmapped)")
    print(f"  → mapped {fixed}, left unmapped {skipped}")

    # --- Step 2: Dedup phrases that appear in multiple groups
    phrase_to_groups: dict[str, list[str]] = {}
    for g in mapping["groups"]:
        for m in g["members"]:
            phrase_to_groups.setdefault(m["phrase"], []).append(g["canonical"])
    duplicates = {
        p: list(set(gs)) for p, gs in phrase_to_groups.items() if len(set(gs)) > 1
    }
    print(f"\nStep 2: deduping {len(duplicates)} phrases in multiple groups...")
    for phrase, group_names in duplicates.items():
        winner = max(group_names, key=lambda c: by_canonical[c]["total"])
        for c in group_names:
            if c == winner:
                continue
            g = by_canonical[c]
            g["members"] = [m for m in g["members"] if m["phrase"] != phrase]
        print(f"  • {phrase}: keep in '{winner}', remove from "
              f"{[c for c in group_names if c != winner]}")

    # --- Step 3: Re-normalize totals, dedup-within-group, sort, regenerate unmapped
    for g in mapping["groups"]:
        seen: dict[str, int] = {}
        for m in g["members"]:
            seen[m["phrase"]] = m["count"]
        g["members"] = sorted(
            ({"phrase": p, "count": c} for p, c in seen.items()),
            key=lambda m: (-m["count"], m["phrase"]),
        )
        g["total"] = sum(m["count"] for m in g["members"])
    mapping["groups"].sort(key=lambda g: (-g["total"], g["canonical"]))

    seen_phrases: set[str] = set()
    for g in mapping["groups"]:
        for m in g["members"]:
            seen_phrases.add(m["phrase"])
    unmapped_pairs = [(p, c) for p, c in counts.items() if p not in seen_phrases]
    unmapped_pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    mapping["unmapped"] = [{"phrase": p, "count": c} for p, c in unmapped_pairs]

    return mapping


def apply_pass2(
    client: anthropic.Anthropic,
    mapping: dict,
    counts: Counter[str],
    chunk_size: int,
) -> dict:
    """Map every remaining unmapped phrase (mostly singletons) onto an existing
    canonical, in chunks. Phrases the LLM declines (\"none\") stay in the
    unmapped list as the genuinely-different long tail.
    """
    canonicals = [g["canonical"] for g in mapping["groups"]]
    by_canonical = {g["canonical"]: g for g in mapping["groups"]}

    unmapped = mapping["unmapped"]
    if not unmapped:
        print("Nothing to map: unmapped list is empty.")
        return mapping

    chunks = [unmapped[i : i + chunk_size] for i in range(0, len(unmapped), chunk_size)]
    print(
        f"Pass 2: mapping {len(unmapped):,} unmapped phrases onto "
        f"{len(canonicals)} canonicals in {len(chunks)} chunks of "
        f"≤ {chunk_size}..."
    )

    total_mapped = 0
    total_none = 0
    for i, chunk in enumerate(chunks, 1):
        print(f"  chunk {i}/{len(chunks)}: {len(chunk)} phrases ...", end="", flush=True)
        assignments = request_unmapped_assignments(client, chunk, canonicals)
        c_mapped = 0
        c_none = 0
        for a in assignments:
            phrase = (a.get("phrase") or "").strip().lower()
            canonical = (a.get("canonical") or "").strip().lower()
            if not phrase:
                continue
            if canonical and canonical != "none" and canonical in by_canonical:
                by_canonical[canonical]["members"].append(
                    {"phrase": phrase, "count": counts.get(phrase, 0)}
                )
                c_mapped += 1
            else:
                c_none += 1
        total_mapped += c_mapped
        total_none += c_none
        print(f"  mapped={c_mapped} none={c_none}")

    print(
        f"\nPass 2 complete: mapped {total_mapped:,}, "
        f"left in unmapped {total_none:,}"
    )

    # Re-normalize
    for g in mapping["groups"]:
        seen: dict[str, int] = {}
        for m in g["members"]:
            seen[m["phrase"]] = m["count"]
        g["members"] = sorted(
            ({"phrase": p, "count": c} for p, c in seen.items()),
            key=lambda m: (-m["count"], m["phrase"]),
        )
        g["total"] = sum(m["count"] for m in g["members"])
    mapping["groups"].sort(key=lambda g: (-g["total"], g["canonical"]))

    seen_phrases: set[str] = set()
    for g in mapping["groups"]:
        for m in g["members"]:
            seen_phrases.add(m["phrase"])
    unmapped_pairs = [(p, c) for p, c in counts.items() if p not in seen_phrases]
    unmapped_pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    mapping["unmapped"] = [{"phrase": p, "count": c} for p, c in unmapped_pairs]

    return mapping


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
    parser.add_argument(
        "--fixup",
        action="store_true",
        help="Patch an existing symptom_mapping.json: LLM-map the unmapped "
        "count>=2 phrases onto existing canonicals, then deterministically "
        "dedup phrases in multiple groups (winner = higher total count).",
    )
    parser.add_argument(
        "--pass2",
        action="store_true",
        help="Map every remaining unmapped phrase (mostly singletons) onto an "
        "existing canonical via chunked LLM calls. Phrases the model declines "
        "stay in the unmapped list as the genuine long tail.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Pass 2 chunk size (default: 500).",
    )
    args = parser.parse_args()

    if args.validate:
        sys.exit(validate_mapping())

    if args.fixup:
        load_dotenv()
        if not MAPPING_FILE.exists():
            print(f"No mapping at {MAPPING_FILE}. Run without --fixup first.",
                  file=sys.stderr)
            sys.exit(2)
        mapping = json.loads(MAPPING_FILE.read_text())
        counts = load_phrase_counts()
        client = anthropic.Anthropic()
        fixed = apply_fixups(client, mapping, counts)
        MAPPING_FILE.write_text(json.dumps(fixed, indent=2))
        print(f"\nWrote fixed mapping to {MAPPING_FILE}")
        return

    if args.pass2:
        load_dotenv()
        if not MAPPING_FILE.exists():
            print(f"No mapping at {MAPPING_FILE}. Run without --pass2 first.",
                  file=sys.stderr)
            sys.exit(2)
        mapping = json.loads(MAPPING_FILE.read_text())
        counts = load_phrase_counts()
        client = anthropic.Anthropic()
        updated = apply_pass2(client, mapping, counts, args.chunk_size)
        MAPPING_FILE.write_text(json.dumps(updated, indent=2))
        print(f"\nWrote updated mapping to {MAPPING_FILE}")
        return

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
