"""Stage 1b: poll the in-flight Message Batch and write extracted symptoms.

Reads .data/batch_state.json (written by extract_submit.py), waits for the
batch to finish, then streams results into:

  .data/symptoms_raw.jsonl     # one row per succeeded post
  .data/symptoms_errors.jsonl  # errored / expired / canceled rows

Resumable: if symptoms_raw.jsonl already exists, only post_ids not already
present are appended.

Usage:
  uv run python -m eoe.extract_collect              # poll → download → write
  uv run python -m eoe.extract_collect --status     # one-shot status, no wait
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from eoe.json_parsing import parse_model_json

DATA_DIR = Path(".data")
INPUT_FILE = DATA_DIR / "r_EosinophilicE_posts.jsonl"
BATCH_STATE_FILE = DATA_DIR / "batch_state.json"
RESULTS_FILE = DATA_DIR / "symptoms_raw.jsonl"
ERRORS_FILE = DATA_DIR / "symptoms_errors.jsonl"
POLL_SECONDS = 30


def load_post_index() -> dict[str, dict]:
    """Map post_id -> minimal post metadata (author, created_utc, permalink)."""
    index: dict[str, dict] = {}
    with INPUT_FILE.open() as f:
        for line in f:
            p = json.loads(line)
            index[p["id"]] = {
                "author": p.get("author"),
                "created_utc": p.get("created_utc"),
                "permalink": p.get("permalink"),
                "title": p.get("title"),
            }
    return index


def already_collected_ids() -> set[str]:
    if not RESULTS_FILE.exists():
        return set()
    seen: set[str] = set()
    with RESULTS_FILE.open() as f:
        for line in f:
            try:
                seen.add(json.loads(line)["post_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def fmt_counts(counts) -> str:
    return (
        f"processing={counts.processing} "
        f"succeeded={counts.succeeded} "
        f"errored={counts.errored} "
        f"canceled={counts.canceled} "
        f"expired={counts.expired}"
    )


def show_status(client: anthropic.Anthropic, batch_id: str) -> None:
    batch = client.messages.batches.retrieve(batch_id)
    print(f"batch_id:           {batch.id}")
    print(f"processing_status:  {batch.processing_status}")
    print(f"created_at:         {batch.created_at}")
    print(f"ended_at:           {batch.ended_at}")
    print(f"expires_at:         {batch.expires_at}")
    print(f"counts:             {fmt_counts(batch.request_counts)}")


def wait_for_batch(client: anthropic.Anthropic, batch_id: str):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        print(f"[{time.strftime('%H:%M:%S')}] {status}  {fmt_counts(batch.request_counts)}")
        if status == "ended":
            return batch
        time.sleep(POLL_SECONDS)


def collect_results(
    client: anthropic.Anthropic, batch_id: str, post_index: dict[str, dict]
) -> tuple[int, int, int]:
    """Stream results into symptoms_raw.jsonl / symptoms_errors.jsonl.
    Returns (written, skipped_already, error_count).
    """
    seen = already_collected_ids()
    written = 0
    skipped = 0
    errors = 0
    parse_failures = 0

    with RESULTS_FILE.open("a") as out, ERRORS_FILE.open("a") as err_out:
        for result in client.messages.batches.results(batch_id):
            post_id = result.custom_id
            meta = post_index.get(post_id, {})

            if post_id in seen:
                skipped += 1
                continue

            r = result.result
            if r.type == "succeeded":
                msg = r.message
                text = next(
                    (b.text for b in msg.content if b.type == "text"), ""
                ).strip()
                parsed = parse_model_json(text)
                if parsed is None or "symptoms" not in parsed:
                    parse_failures += 1
                    err_out.write(
                        json.dumps(
                            {
                                "post_id": post_id,
                                "kind": "parse_failure",
                                "raw": text,
                            }
                        )
                        + "\n"
                    )
                    continue
                out.write(
                    json.dumps(
                        {
                            "post_id": post_id,
                            "author": meta.get("author"),
                            "created_utc": meta.get("created_utc"),
                            "permalink": meta.get("permalink"),
                            "title": meta.get("title"),
                            "symptoms": parsed["symptoms"],
                        }
                    )
                    + "\n"
                )
                written += 1
            else:
                errors += 1
                err_record = {"post_id": post_id, "kind": r.type}
                if r.type == "errored" and getattr(r, "error", None):
                    err_record["error"] = {
                        "type": getattr(r.error, "type", None),
                        "message": getattr(r.error, "message", None),
                    }
                err_out.write(json.dumps(err_record) + "\n")

    print(
        f"\nDone. written={written} already_present={skipped} "
        f"errors={errors} parse_failures={parse_failures}"
    )
    print(f"  → {RESULTS_FILE}")
    if errors or parse_failures:
        print(f"  → {ERRORS_FILE}  (review failures)")
    return written, skipped, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print one-shot batch status and exit. Don't poll, don't download.",
    )
    args = parser.parse_args()

    load_dotenv()

    if not BATCH_STATE_FILE.exists():
        print(
            f"No batch state at {BATCH_STATE_FILE}. Run extract_submit.py first.",
            file=sys.stderr,
        )
        sys.exit(2)

    state = json.loads(BATCH_STATE_FILE.read_text())
    batch_id = state["batch_id"]
    print(
        f"Tracking batch {batch_id}  model={state['model']}  "
        f"submitted={state['submitted_at']}  requests={state['request_count']:,}"
    )

    client = anthropic.Anthropic()

    if args.status:
        show_status(client, batch_id)
        return

    batch = wait_for_batch(client, batch_id)
    print(f"\nBatch ended at {batch.ended_at}. Streaming results...")
    post_index = load_post_index()
    collect_results(client, batch_id, post_index)


if __name__ == "__main__":
    main()
