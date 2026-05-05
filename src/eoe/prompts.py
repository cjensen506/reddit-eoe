"""Prompts for the EoE symptom-analysis pipeline.

Kept in one module so the extraction system prompt is byte-stable across
batch requests (and would be cacheable if we weren't already using batch
pricing). Edit here, not in the per-stage scripts.
"""

EXTRACTION_SYSTEM_PROMPT = """\
You are extracting self-reported physical symptoms from posts on r/EosinophilicE \
(an Eosinophilic Esophagitis support subreddit).

You will be given the title and body of one Reddit post. Return a JSON object \
listing every symptom that a specific person — usually the author, sometimes a \
named family member — actually experiences and that the post associates with \
their EoE.

WHAT TO INCLUDE
- Physical symptoms the author says they (or a named relative) have: dysphagia, \
food impaction, heartburn, chest pain, regurgitation, throat tightness, vomiting, \
weight loss, abdominal pain, fatigue, etc. Use the author's own wording for the \
phrase.
- Current symptoms AND past symptoms (e.g. "before I started Dupixent I would \
choke daily"). Both count, as long as it's the author's actual experience.
- Symptoms experienced by a specific named individual the author is describing \
on their behalf — e.g. "my 6-year-old son chokes on rice", "my husband regurgitates \
food". These DO count.

WHAT TO EXCLUDE — these are common traps, read carefully:

1. Negation. If the author says they do NOT have a symptom, do NOT extract it. \
Watch for words like "don't", "no longer", "never", "used to but not anymore", \
"haven't had X since". Example: "I don't choke when I eat" → do not extract \
"choking".

2. Hypothetical / advice / educational framing. If the post is giving general \
advice about EoE, listing what EoE *can* cause, summarizing what foods *might* \
hurt, or otherwise not describing the author's lived experience — return an \
empty list. Example: "Carbonated beverages will hurt if you drink a lot" stated \
as a tip is NOT a symptom; "I get chest pain whenever I drink soda" IS.

3. Medication side effects. If the author attributes a symptom to a medication \
(Flovent, budesonide, PPIs, hyoscyamine, etc.) and frames it as a side effect or \
reaction to that drug, do NOT extract it. Example: "hyoscyamine relaxed my \
intestines and the pain went away" — the relief is from the drug, not an EoE \
symptom. Only extract the same symptom if the author separately frames it as \
part of their EoE itself.

4. Generic third-person mentions. Skip "people with EoE", "patients", "someone \
I read about", "you might experience". Anonymous/generic references don't count, \
even though specific named individuals (rule above) do.

5. Comorbidity lists. When the author lists multiple conditions in one breath \
("I have asthma, celiac, acid reflux, and EoE"), do NOT extract any of those \
other conditions as EoE symptoms unless the post separately ties that condition \
to their EoE experience.

6. Emotional states alone (anxiety, depression, frustration) unless tied to a \
physical EoE manifestation.

7. Test results, diagnoses, biopsy findings ("high eosinophil count", "rings on \
endoscopy"). Those are findings, not symptoms.

OUTPUT FORMAT
Return ONLY valid JSON, no prose, no markdown fences:
{
  "symptoms": [
    {"phrase": "<short normalized lowercase wording, e.g. 'food getting stuck'>",
     "quote": "<short verbatim span from the post that grounds the phrase, max ~120 chars>"}
  ]
}

If the author describes no qualifying symptoms — including hypothetical/advice \
posts and posts that are purely about diagnosis, dilation procedures, medication \
choices, or diet without symptom descriptions — return {"symptoms": []}.

Do not invent symptoms. Every entry must be supported by a quote you can copy \
verbatim from the post. Re-read each candidate against the EXCLUDE rules above \
before including it.
"""


def build_extraction_user_message(title: str, selftext: str) -> str:
    """Format a single post into the user-turn payload for extraction."""
    return f"TITLE: {title}\n\nBODY:\n{selftext}"


FIXUP_ASSIGNMENT_SYSTEM_PROMPT = """\
You are mapping a small set of leftover symptom phrases onto an existing list \
of canonical symptom groups for an analysis of self-reported symptoms in \
r/EosinophilicE posts.

You will be given:
1. A list of canonical symptom names that have already been established.
2. A list of unmapped phrases (with their occurrence counts).

For each unmapped phrase, return the single canonical name it best matches. \
Map liberally — these phrases are slight wording variations of common \
self-reported EoE symptoms (e.g. "food getting stuck" → "food impaction", \
"food sticking" → "food impaction", "throat soreness" → "sore throat"). \
Only return "none" if the phrase represents a genuinely different symptom \
not covered by any existing canonical.

Output format — return ONLY valid JSON, no prose, no markdown fences:
{
  "assignments": [
    {"phrase": "<phrase>", "canonical": "<canonical name or 'none'>"}
  ]
}

Every input phrase must appear exactly once in the assignments list.
"""


CLUSTERING_SYSTEM_PROMPT = """\
You are grouping free-text symptom phrases extracted from r/EosinophilicE posts \
into canonical symptom categories.

You will be given a JSON list of distinct symptom phrases with their occurrence \
counts. Return a JSON mapping that groups synonymous or near-synonymous phrases \
under a single canonical name.

Guidelines:
- The canonical name should be short, lowercase, and clinically recognizable \
where possible (e.g. "dysphagia", "food impaction", "heartburn", "chest pain", \
"regurgitation", "throat tightness", "vomiting", "weight loss"). When no \
clinical term fits, use the most common patient phrasing.
- Group obvious synonyms together: "burping" + "belching" + "burps" → "burping". \
"food getting stuck" + "food impaction" + "stuck food" → "food impaction".
- Be conservative about merging. If two phrases point at meaningfully different \
sensations (e.g. "heartburn" vs "chest pain"), keep them separate.
- Every input phrase must appear in exactly one group's `members` list.
- Sort groups roughly by total member count, biggest first.

Output format — return ONLY valid JSON:
{
  "groups": [
    {"canonical": "<canonical name>", "members": ["<phrase1>", "<phrase2>", ...]}
  ]
}
"""
