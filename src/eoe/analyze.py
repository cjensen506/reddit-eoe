"""Stage 3: aggregate canonical symptom counts and produce charts + quotes.

Reads:
  .data/symptoms_raw.jsonl       # one row per post, with raw extracted symptoms
  mappings/symptom_mapping.json  # raw phrase -> canonical mapping (tracked in git)

Writes:
  .data/symptom_counts.csv    # per-canonical post + author counts, side by side
  images/top_symptoms.png     # top-30 horizontal bar chart (tracked in git)
  images/wordcloud.png        # word cloud sized by post count (tracked in git)
  images/cooccurrence.png     # top-20 co-occurrence heatmap (tracked in git)
  .data/symptom_quotes.md     # 3-5 verbatim grounding quotes per canonical

Per-post dedup: a post that uses 4 phrases that all map to `dysphagia` counts
once for `dysphagia`. Author dedup: same idea across an author's posts.
[deleted]/None authors are excluded from the author counts (treated as the
unidentifiable tail rather than one big bucket that would inflate share).

Usage:
  uv run python -m eoe.analyze
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

DATA_DIR = Path(".data")
MAPPINGS_DIR = Path("mappings")
RAW_FILE = DATA_DIR / "symptoms_raw.jsonl"
MAPPING_FILE = MAPPINGS_DIR / "symptom_mapping.json"
COUNTS_CSV = DATA_DIR / "symptom_counts.csv"
QUOTES_FILE = DATA_DIR / "symptom_quotes.md"
IMAGES_DIR = Path("images")
TOP_CHART = IMAGES_DIR / "top_symptoms.png"
COOC_CHART = IMAGES_DIR / "cooccurrence.png"
WORDCLOUD_CHART = IMAGES_DIR / "wordcloud.png"

TOP_N_BAR = 30
TOP_N_HEATMAP = 20
QUOTES_PER_SYMPTOM = 5
QUOTE_MIN_LEN = 20
QUOTE_MAX_LEN = 240
QUOTE_SEED = 0


def load_mapping() -> tuple[dict[str, str], list[str], dict[str, list[str]]]:
    """Return (phrase->canonical, canonicals_in_order, canonical->top_member_phrases)."""
    m = json.loads(MAPPING_FILE.read_text())
    phrase_to_canon: dict[str, str] = {}
    canonicals: list[str] = []
    examples: dict[str, list[str]] = {}
    for g in m["groups"]:
        canonical = g["canonical"]
        canonicals.append(canonical)
        examples[canonical] = [m_["phrase"] for m_ in g["members"][:3]]
        for member in g["members"]:
            phrase_to_canon[member["phrase"]] = canonical
    return phrase_to_canon, canonicals, examples


def aggregate(phrase_to_canon: dict[str, str]):
    """One pass over symptoms_raw.jsonl. Build per-canonical post sets, author
    sets, co-occurrence counts, and a list of grounding quotes.
    """
    posts_per_canon: dict[str, set[str]] = defaultdict(set)
    authors_per_canon: dict[str, set[str]] = defaultdict(set)
    cooc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    quotes_per_canon: dict[str, list[dict]] = defaultdict(list)

    total_posts = 0
    total_authors: set[str] = set()
    symptomatic_posts = 0
    symptomatic_authors: set[str] = set()

    with RAW_FILE.open() as f:
        for line in f:
            row = json.loads(line)
            total_posts += 1
            author = row.get("author")
            real_author = author if author and author != "[deleted]" else None
            if real_author:
                total_authors.add(real_author)

            canon_in_post: set[str] = set()
            for s in row.get("symptoms", []):
                phrase = (s.get("phrase") or "").strip().lower()
                canonical = phrase_to_canon.get(phrase)
                if not canonical:
                    continue
                canon_in_post.add(canonical)
                quote = (s.get("quote") or "").strip()
                if quote:
                    quotes_per_canon[canonical].append(
                        {
                            "quote": quote,
                            "phrase": phrase,
                            "post_id": row["post_id"],
                            "permalink": row.get("permalink") or "",
                            "title": row.get("title") or "",
                        }
                    )

            if canon_in_post:
                symptomatic_posts += 1
                if real_author:
                    symptomatic_authors.add(real_author)

            for c in canon_in_post:
                posts_per_canon[c].add(row["post_id"])
                if real_author:
                    authors_per_canon[c].add(real_author)

            canon_list = list(canon_in_post)
            for i, a in enumerate(canon_list):
                for b in canon_list[i + 1 :]:
                    cooc[a][b] += 1
                    cooc[b][a] += 1

    return {
        "posts_per_canon": posts_per_canon,
        "authors_per_canon": authors_per_canon,
        "cooc": cooc,
        "quotes_per_canon": quotes_per_canon,
        "total_posts": total_posts,
        "total_authors": len(total_authors),
        "symptomatic_posts": symptomatic_posts,
        "symptomatic_authors": len(symptomatic_authors),
    }


def write_counts_csv(
    canonicals: list[str], examples: dict[str, list[str]], agg: dict
) -> list[dict]:
    sympt_posts = agg["symptomatic_posts"]
    sympt_authors = agg["symptomatic_authors"]
    rows: list[dict] = []
    for c in canonicals:
        pc = len(agg["posts_per_canon"].get(c, set()))
        ac = len(agg["authors_per_canon"].get(c, set()))
        rows.append(
            {
                "canonical": c,
                "post_count": pc,
                "post_share": pc / sympt_posts if sympt_posts else 0.0,
                "author_count": ac,
                "author_share": ac / sympt_authors if sympt_authors else 0.0,
                "example_phrases": "; ".join(examples.get(c, [])),
            }
        )
    rows.sort(key=lambda r: (-r["post_count"], r["canonical"]))

    with COUNTS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "canonical",
                "post_count",
                "post_share",
                "author_count",
                "author_share",
                "example_phrases",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    **r,
                    "post_share": f"{r['post_share']:.4f}",
                    "author_share": f"{r['author_share']:.4f}",
                }
            )
    print(f"  → {COUNTS_CSV}")
    return rows


def plot_top_bar(rows: list[dict], agg: dict) -> None:
    top = rows[:TOP_N_BAR][::-1]  # ascending so largest is at top
    canonicals = [r["canonical"] for r in top]
    counts = [r["post_count"] for r in top]

    fig, ax = plt.subplots(figsize=(11, max(8, TOP_N_BAR * 0.3)))
    bars = ax.barh(canonicals, counts, color="#3b6aa0")
    ax.set_xlabel("Posts mentioning the symptom")
    ax.set_title(
        f"Top {TOP_N_BAR} self-reported symptoms in r/EosinophilicE\n"
        f"(% of {agg['symptomatic_posts']:,} posts that report ≥1 symptom; "
        f"per-post deduped)"
    )
    pad = max(counts) * 0.01
    for bar, r in zip(bars, top):
        ax.text(
            bar.get_width() + pad,
            bar.get_y() + bar.get_height() / 2,
            f"{r['post_count']:,}  ({r['post_share']*100:.1f}%)",
            va="center",
            fontsize=8.5,
        )
    ax.set_xlim(0, max(counts) * 1.18)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(TOP_CHART, dpi=140)
    plt.close(fig)
    print(f"  → {TOP_CHART}")


def plot_wordcloud(rows: list[dict]) -> None:
    """Word cloud sized by per-canonical post count."""
    frequencies = {r["canonical"]: r["post_count"] for r in rows if r["post_count"] > 0}
    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        prefer_horizontal=0.9,
        collocations=False,
        relative_scaling=0.6,
        colormap="viridis",
        random_state=0,
    ).generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Self-reported symptoms in r/EosinophilicE (size ∝ posts)")
    fig.tight_layout()
    fig.savefig(WORDCLOUD_CHART, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {WORDCLOUD_CHART}")


def plot_cooccurrence(rows: list[dict], agg: dict) -> None:
    """Conditional-probability heatmap. Cell (row, col) reads:
    of posts mentioning ROW, what fraction also mention COL.
    Asymmetric. Diagonal masked so the color scale reflects only off-diagonals.
    """
    top = [r["canonical"] for r in rows[:TOP_N_HEATMAP]]
    n = len(top)
    matrix = np.full((n, n), np.nan, dtype=float)
    for i, a in enumerate(top):
        a_count = len(agg["posts_per_canon"].get(a, set()))
        if a_count == 0:
            continue
        for j, b in enumerate(top):
            if i == j:
                continue
            matrix[i, j] = agg["cooc"][a].get(b, 0) / a_count

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")  # diagonal NaNs

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(top, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(top, fontsize=9)
    ax.set_xlabel("…also mentions")
    ax.set_ylabel("Of posts mentioning…")
    ax.set_title(
        f"Symptom co-occurrence — P(column | row), top {TOP_N_HEATMAP}"
    )

    finite_max = np.nanmax(matrix) if np.any(np.isfinite(matrix)) else 0.0
    threshold = finite_max / 2 if finite_max else 0
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
            else:
                v = matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{v*100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if v > threshold else "black",
                )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, format=lambda x, _: f"{x*100:.0f}%")
    cbar.set_label("P(column mentioned | row mentioned)")
    fig.tight_layout()
    fig.savefig(COOC_CHART, dpi=140)
    plt.close(fig)
    print(f"  → {COOC_CHART}")


def write_quotes_md(rows: list[dict], agg: dict) -> None:
    rng = random.Random(QUOTE_SEED)
    sympt_posts = agg["symptomatic_posts"]
    sympt_authors = agg["symptomatic_authors"]
    with QUOTES_FILE.open("w") as f:
        f.write("# Self-reported EoE symptoms — verbatim quotes\n\n")
        f.write(
            f"For each canonical symptom (sorted by post count), {QUOTES_PER_SYMPTOM} "
            "randomly sampled quotes from r/EosinophilicE posts. The raw phrase "
            "shown is the lowercased extracted phrase that maps to this canonical. "
            f"Percentages are over the {sympt_posts:,} posts that report at least "
            f"one symptom (and {sympt_authors:,} distinct authors of those posts).\n\n"
        )
        for r in rows:
            canonical = r["canonical"]
            pc = r["post_count"]
            ac = r["author_count"]
            f.write(f"## {canonical}\n\n")
            f.write(
                f"_{pc:,} posts ({r['post_share']*100:.1f}%); "
                f"{ac:,} distinct authors ({r['author_share']*100:.1f}%)_\n\n"
            )
            quotes = agg["quotes_per_canon"].get(canonical, [])
            filtered = [
                q for q in quotes if QUOTE_MIN_LEN <= len(q["quote"]) <= QUOTE_MAX_LEN
            ]
            pool = filtered or quotes
            sample = (
                rng.sample(pool, QUOTES_PER_SYMPTOM)
                if len(pool) > QUOTES_PER_SYMPTOM
                else pool
            )
            if not sample:
                f.write("_(no quotes available)_\n\n")
                continue
            for q in sample:
                quote = q["quote"].replace("\n", " ").strip()
                permalink = q["permalink"]
                url = (
                    f"https://www.reddit.com{permalink}"
                    if permalink and permalink.startswith("/")
                    else permalink
                )
                f.write(f"- > {quote}\n")
                if url:
                    f.write(f"  — [{q['post_id']}]({url}) (raw: _{q['phrase']}_)\n\n")
                else:
                    f.write(f"  — `{q['post_id']}` (raw: _{q['phrase']}_)\n\n")
    print(f"  → {QUOTES_FILE}")


def main() -> None:
    if not RAW_FILE.exists():
        raise SystemExit(f"Missing {RAW_FILE} — run extract_collect.py first.")
    if not MAPPING_FILE.exists():
        raise SystemExit(f"Missing {MAPPING_FILE} — run cluster.py first.")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    phrase_to_canon, canonicals, examples = load_mapping()
    print(
        f"Loaded {len(phrase_to_canon):,} phrase mappings across "
        f"{len(canonicals)} canonical groups"
    )

    print("\nAggregating per-canonical post + author counts...")
    agg = aggregate(phrase_to_canon)
    print(
        f"  {agg['total_posts']:,} posts analyzed "
        f"({agg['symptomatic_posts']:,} report ≥1 symptom — used as denominator); "
        f"{agg['total_authors']:,} distinct authors "
        f"({agg['symptomatic_authors']:,} symptomatic — used as denominator)"
    )

    print("\nWriting outputs...")
    rows = write_counts_csv(canonicals, examples, agg)
    plot_top_bar(rows, agg)
    plot_wordcloud(rows)
    plot_cooccurrence(rows, agg)
    write_quotes_md(rows, agg)

    print("\nDone.")
    print(f"\nTop 10 by post count:")
    for r in rows[:10]:
        print(
            f"  {r['post_count']:>5}  ({r['post_share']*100:>5.1f}%)  "
            f"{r['author_count']:>4} authors  {r['canonical']}"
        )


if __name__ == "__main__":
    main()
