"""Stage 1a: build the per-post symptom-extraction requests.

Two modes:

  uv run python -m eoe.extract_submit --limit 20 --no-batch
    → realtime, prints results inline. For prompt sanity-checking only.

  uv run python -m eoe.extract_submit [--model haiku|sonnet] [--force]
    → submits one Message Batch covering every eligible post and persists
      .data/batch_state.json so extract_collect.py can poll/download it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

from eoe.json_parsing import parse_model_json
from eoe.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_message

DATA_DIR = Path(".data")
INPUT_FILE = DATA_DIR / "r_EosinophilicE_posts.jsonl"
BATCH_STATE_FILE = DATA_DIR / "batch_state.json"
MIN_TEXT_CHARS = 50
MAX_TOKENS = 1024

MODEL_IDS = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
}


def iter_eligible_posts(path: Path):
    with path.open() as f:
        for line in f:
            post = json.loads(line)
            text = post.get("selftext") or ""
            if text in ("[removed]", "[deleted]", ""):
                continue
            title = post.get("title") or ""
            if len(title) + len(text) < MIN_TEXT_CHARS:
                continue
            yield post


def build_request(post: dict, model: str) -> Request:
    return Request(
        custom_id=post["id"],
        params=MessageCreateParamsNonStreaming(
            model=model,
            max_tokens=MAX_TOKENS,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": build_extraction_user_message(
                        post["title"], post["selftext"]
                    ),
                }
            ],
        ),
    )


def run_no_batch(client: anthropic.Anthropic, posts: list[dict], model: str) -> None:
    """Realtime sanity-check mode. For each post, prints the model's extraction
    followed by the full post body so a human reviewer can verify groundedness.
    """
    for i, post in enumerate(posts, 1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(posts)}] {post['id']}  —  {post['title']}")
        print("=" * 80)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": build_extraction_user_message(
                            post["title"], post["selftext"]
                        ),
                    }
                ],
            )
            raw = next(
                (b.text for b in response.content if b.type == "text"), ""
            ).strip()
            print("\n--- EXTRACTION ---")
            parsed = parse_model_json(raw)
            if parsed is not None:
                print(json.dumps(parsed, indent=2))
            else:
                print(f"[unparseable JSON — raw output]\n{raw}")
        except anthropic.APIError as e:
            print(f"\n--- EXTRACTION ---\n[error] {type(e).__name__}: {e}")

        print("\n--- POST BODY ---")
        print(post.get("selftext") or "")


def submit_batch(client: anthropic.Anthropic, posts: list[dict], model: str) -> None:
    if BATCH_STATE_FILE.exists():
        existing = json.loads(BATCH_STATE_FILE.read_text())
        print(
            f"Refusing to submit: a batch is already tracked in "
            f"{BATCH_STATE_FILE} (id={existing['batch_id']}, "
            f"model={existing['model']}, submitted_at={existing['submitted_at']}).\n"
            f"Re-run with --force to overwrite, or run extract_collect.py first.",
            file=sys.stderr,
        )
        sys.exit(2)

    requests = [build_request(p, model) for p in posts]
    print(f"Submitting batch of {len(requests):,} requests on {model}...")
    batch = client.messages.batches.create(requests=requests)

    state = {
        "batch_id": batch.id,
        "model": model,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "request_count": len(requests),
        "post_ids": [p["id"] for p in posts],
    }
    BATCH_STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"Batch submitted: {batch.id}")
    print(f"State persisted to {BATCH_STATE_FILE}")
    print(f"Next: uv run python -m eoe.extract_collect")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["haiku", "sonnet"],
        default="haiku",
        help="Extraction model. Default: haiku.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N eligible posts (use with --no-batch for sanity check).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Random sample N posts (from --limit) using this seed instead of the first N.",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Run realtime messages.create calls and print to stdout. No batch submitted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing batch_state.json.",
    )
    args = parser.parse_args()

    load_dotenv()
    client = anthropic.Anthropic()
    model_id = MODEL_IDS[args.model]

    posts = list(iter_eligible_posts(INPUT_FILE))
    if args.sample_seed is not None and args.limit is not None:
        rng = random.Random(args.sample_seed)
        posts = rng.sample(posts, args.limit)
        print(
            f"Eligible posts: {args.limit:,} (random sample, seed={args.sample_seed})  "
            f"(model={model_id})"
        )
    elif args.limit is not None:
        posts = posts[: args.limit]
        print(f"Eligible posts: {len(posts):,}  (model={model_id})")
    else:
        print(f"Eligible posts: {len(posts):,}  (model={model_id})")

    if args.no_batch:
        run_no_batch(client, posts, model_id)
        return

    if args.force and BATCH_STATE_FILE.exists():
        BATCH_STATE_FILE.unlink()
    submit_batch(client, posts, model_id)


if __name__ == "__main__":
    main()
