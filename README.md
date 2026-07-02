# Redrob Ranker — Intelligent Candidate Discovery

Ranks the 100,000-candidate pool for the **Senior AI Engineer** JD and writes a
validator-passing top-100 CSV.

> **Live sandbox:** [<PASTE_YOUR_STREAMLIT_CLOUD_URL_HERE>](https://redrob-candidate-ranker-yvzsbnzclzjhdfxghdtr7f.streamlit.app/)
> Runs this exact pipeline on a ≤100-candidate sample, end-to-end, in the browser.

**Core idea:** signal lives in what a candidate has *actually done* — their
`career_history` text — not in the gameable `skills` array or the noisy
`education` years. Scoring reads work text; the skills array is treated as a
*claim* checked against the work, and the gap between them is how keyword-
stuffers are caught.

## Reproduce the submission

**Single command (Stage 3 — runs in the 5-min window using committed artifacts):**

```bash
python -m src.rank --candidates data/candidates.jsonl --out submission.csv
```

**From scratch (offline pre-computation — may exceed 5 min, downloads the model):**

```bash
python -m venv venv && venv\Scripts\activate        # Windows
pip install -r requirements.txt
# place data/candidates.jsonl and data/job_description.docx

python -m src.precompute --candidates data/candidates.jsonl --jd data/job_description.docx
python -m src.rank        --candidates data/candidates.jsonl --out submission.csv
python validate_submission.py submission.csv
```

Helper modes (no re-embedding): `--features-only` (regenerate features after a
scoring change), `--bm25-only` (build just the BM25 index, ~30-60 s).

## How it works

Two phases, split at the 5-minute boundary:

- **Offline (`src/precompute.py`, unlimited):** embed each candidate's work text
  with a local BGE model, extract structured features, build a BM25 index over
  work text. Written to `artifacts/`.
- **Online (`src/rank.py`, the ONLY timed step — 5 min / 16 GB / CPU / no
  network):** RRF-fuse dense cosine + BM25 retrieval, apply structured JD-rubric
  scoring + a sentinel-aware behavioral multiplier + consistency penalties,
  write the top-100 CSV. Loads no model, makes no network calls.

Scoring: `final = (W_SEM*semantic + W_FIT*fit) * behavioral - honeypot_penalty`,
where `semantic` is the RRF retrieval score, `fit` is must-have hits in the
*work* (retrieval/ranking, vector DBs, eval frameworks) plus experience band and
geography minus disqualifier penalties (non-technical title, keyword stuffer,
off-domain, services-only, job-hopper), and behavioral folds in recruiter
response, recency, and notice period. Behavioral `-1` sentinels are treated as
*missing*, never as low scores. Reasoning is generated from the same features
that drive the score, so it cites real facts without hallucination.

## Folder structure

```
redrob-ranker/
├── data/candidates.jsonl        <- the pool (gitignored; graders supply their own)
├── data/job_description.docx    <- the JD
├── eda/                          standalone EDA + the rule-validation harness (03_honeypots.py)
├── eda_out/                      committed EDA findings
├── src/
│   ├── config.py                 thresholds frozen from EDA + the JD rubric
│   ├── features.py               candidate -> features (work-text first, sentinel-aware)
│   ├── scoring.py                fit + behavioral + reasoning (the logic we own)
│   ├── precompute.py             OFFLINE: embeddings + features + BM25 -> artifacts/
│   └── rank.py                   ONLINE: load artifacts, RRF + score, write CSV
├── artifacts/                    emb.npy + features.jsonl + bm25.pkl (COMMITTED via Git LFS)
├── app.py                        hosted-sandbox app (Streamlit); sample bundled
├── validate_submission.py        the OFFICIAL validator (vendored, unmodified)
├── submission_metadata.yaml      portal metadata mirror
├── requirements.txt
└── README.md
```

Artifacts are committed (via Git LFS, since `emb.npy` is ~150 MB) so the single
reproduce command works on a fresh clone without a 20-minute re-embed. The raw
465 MB `candidates.jsonl` is gitignored — graders provide their own copy.

## What gets submitted (submission_spec §10)

1. **CSV** — `submission.csv`, top-100, validator-clean.
2. **Portal metadata** — mirrored in `submission_metadata.yaml`.
3. **This repo** — single reproduce command above; artifacts committed; deps pinned.
4. **Sandbox link** — the Streamlit app above (≤100 sample, end-to-end, ≤5 min).
5. **AI-tools declaration** — in `submission_metadata.yaml`.