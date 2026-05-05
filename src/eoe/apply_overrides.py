"""Apply declarative overrides on top of the raw Stage-2 mapping.

Reads:
  mappings/symptom_mapping.raw.json  # ground-truth Stage-2 LLM output
  mappings/overrides.toml            # manual decisions, applied in order

Writes:
  mappings/symptom_mapping.json      # final mapping consumed by Stage 3

Run with:
  uv run python -m eoe.apply_overrides
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAPPINGS_DIR = ROOT / "mappings"
RAW_FILE = MAPPINGS_DIR / "symptom_mapping.raw.json"
OVERRIDES_FILE = MAPPINGS_DIR / "overrides.toml"
OUT_FILE = MAPPINGS_DIR / "symptom_mapping.json"


def _find(groups: list[dict], canonical: str) -> dict:
    for g in groups:
        if g["canonical"] == canonical:
            return g
    raise KeyError(f"canonical not found: {canonical!r}")


def _recompute_total(group: dict) -> None:
    group["total"] = sum(m["count"] for m in group["members"])
    group["members"].sort(key=lambda m: (-m["count"], m["phrase"]))


def op_merge(groups: list[dict], canonicals: list[str], into: str) -> None:
    if len(canonicals) < 2:
        raise ValueError(f"merge needs >=2 canonicals, got {canonicals!r}")
    sources = [_find(groups, c) for c in canonicals]
    merged_members: dict[str, int] = {}
    for src in sources:
        for mem in src["members"]:
            merged_members[mem["phrase"]] = merged_members.get(mem["phrase"], 0) + mem["count"]
    for src in sources:
        groups.remove(src)
    new_group = {
        "canonical": into,
        "total": 0,
        "members": [{"phrase": p, "count": c} for p, c in merged_members.items()],
    }
    _recompute_total(new_group)
    groups.append(new_group)


def op_rename(groups: list[dict], _from: str, to: str) -> None:
    g = _find(groups, _from)
    if any(other["canonical"] == to for other in groups if other is not g):
        raise ValueError(f"rename target already exists: {to!r}")
    g["canonical"] = to


def op_move(groups: list[dict], _from: str, to: str, phrases: list[str]) -> None:
    src = _find(groups, _from)
    dst = _find(groups, to)
    by_phrase = {m["phrase"]: m for m in src["members"]}
    missing = [p for p in phrases if p not in by_phrase]
    if missing:
        raise KeyError(f"phrases not found in {_from!r}: {missing}")
    for p in phrases:
        mem = by_phrase[p]
        src["members"].remove(mem)
        dst["members"].append(mem)
    _recompute_total(src)
    _recompute_total(dst)


OPS = {"merge": op_merge, "rename": op_rename, "move": op_move}


def apply(mapping: dict, operations: list[dict]) -> dict:
    groups = list(mapping["groups"])
    for i, raw_op in enumerate(operations, 1):
        op = dict(raw_op)
        name = op.pop("op", None)
        if name not in OPS:
            raise ValueError(f"operation {i}: unknown op {name!r} (expected one of {sorted(OPS)})")
        # tomllib parses bare `from` fine, but it's a Python keyword — accept it via dict pop
        if "from" in op:
            op["_from"] = op.pop("from")
        try:
            OPS[name](groups, **op)
        except (KeyError, ValueError, TypeError) as e:
            raise type(e)(f"operation {i} ({name}): {e}") from e
    groups.sort(key=lambda g: (-g["total"], g["canonical"]))
    return {**mapping, "groups": groups}


def main() -> int:
    raw = json.loads(RAW_FILE.read_text())
    overrides = tomllib.loads(OVERRIDES_FILE.read_text())
    operations = overrides.get("operations", [])
    print(f"Loaded raw mapping: {len(raw['groups'])} groups, {sum(len(g['members']) for g in raw['groups'])} members")
    print(f"Applying {len(operations)} override operation(s) from {OVERRIDES_FILE.name}")
    final = apply(raw, operations)
    OUT_FILE.write_text(json.dumps(final, indent=2) + "\n")
    print(f"Wrote {OUT_FILE.relative_to(ROOT)}: {len(final['groups'])} groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
