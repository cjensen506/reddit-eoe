"""Tolerant JSON extraction from Claude responses.

Haiku 4.5 occasionally wraps JSON in ```json``` fences or appends prose after
the closing brace despite explicit instructions, so we need a forgiving parser
for the extraction pipeline.
"""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def parse_model_json(text: str) -> dict | None:
    """Parse a JSON object from a model response.

    Returns the parsed dict, or None if no JSON object can be recovered.
    Handles: bare JSON, ```json fenced JSON, JSON followed by trailing prose.
    """
    text = text.strip()
    if not text:
        return None

    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            in_string = False
            escape = False
            for j in range(i, len(text)):
                c = text[j]
                if in_string:
                    if escape:
                        escape = False
                    elif c == "\\":
                        escape = True
                    elif c == '"':
                        in_string = False
                else:
                    if c == '"':
                        in_string = True
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            candidates.append(text[i : j + 1])
                            i = j + 1
                            break
            else:
                break
        else:
            i += 1

    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None
