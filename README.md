# Redrob Ranker — Intelligent Candidate Discovery

Ranks the 100,000-candidate pool for the **Senior AI Engineer** JD and writes a validator-passing top-100 CSV.

**Core idea:** signal lives in what a candidate has *actually done* — their
`career_history` text — not in the gameable `skills` array or the noisy
`education` years. Everything below follows from that: scoring reads work text,
the skills array is treated as a *claim* to be checked against the work, and the
gap between the two is how keyword-stuffers are caught.

## Two-phase design (split at the 5-minute boundary)

- **Offline — `src/precompute.py`** (unlimited time, may download the embedding
  model): builds per-candidate embeddings, a feature table, and a BM25 index
  over work text. Writes to `artifacts/`.
- **Online — `src/rank.py`** (the ONLY step on the 5-min / 16 GB / CPU /
  no-network clock): loads the artifacts, fuses dense + BM25 retrieval with RRF,
  applies structured JD-rubric scoring + a behavioral multiplier + consistency
  penalties, and writes the top-100 CSV. It loads **no model and touches no
  network**.

## How a candidate is scored

1. **Hybrid retrieval (semantic component).** Dense cosine (BGE embedding of
   work text vs the JD) is RRF-fused with BM25 over the same work text. BM25
   catches exact-terminology matches; the embedding catches plain-language fits.
   The BM25 query is the JD's must-have + nice-to-have terms — not a hand-tuned
   keyword soup, and never the candidate's skills array.
2. **Structured fit.** Must-have hits *in the work* (retrieval/ranking/recsys,
   vector DBs, eval frameworks), experience band (JD sweet spot 6-9 yrs),
   geography - minus disqualifier penalties (non-technical title, keyword
   stuffer, off-domain CV/speech-only, services-only career, job-hopper).
3. **Behavioral multiplier** (`0.55-1.15`). Recruiter response rate, recency,
   open-to-work, notice period. Sentinel-aware: `-1` means *missing*, never
   *bad*.
4. **Consistency penalty.** Honeypot-style profile contradictions are penalized
   (buried), never deleted.

`final = (W_SEM*semantic + W_FIT*fit) * behavioral - honeypot_penalty`

## Folder structure

```
redrob-ranker/
├── data/
│   ├── candidates.jsonl         <- the pool (gitignored; graders supply their own)
│   └── job_description.md / .docx
├── eda/                          standalone EDA (streams the file, no installs)
│   ├── 01_overview.py  02_signals.py
│   ├── 03_honeypots.py           <- the rule-validation harness (key diagnostic)
│   ├── 04_keyword_stuffers.py  05_cosine_probe.py
├── eda_out/                      EDA findings (committed, from a real run)
├── src/
│   ├── config.py                 thresholds frozen from EDA + the JD rubric
│   ├── features.py               candidate -> features (work-text first, sentinel-aware)
│   ├── scoring.py                fit + behavioral + reasoning (the logic we own)
│   ├── precompute.py             OFFLINE: embeddings + features + BM25 -> artifacts/
│   └── rank.py                   ONLINE: load artifacts, RRF + score, write CSV
├── artifacts/                    emb.npy + features.jsonl + bm25.pkl (COMMITTED — see note)
├── deploy/                       hosted-sandbox app (Streamlit / HF Spaces) + sample
├── validate_submission.py        the OFFICIAL validator (vendored, unmodified)
├── submission_metadata.yaml      portal metadata mirror (fill in team identity)
├── requirements.txt
└── README.md
```

**Artifacts are committed on purpose** so the single reproduce command works on a fresh clone without a 20-minute re-embed. `emb.npy` is ~150 MB — use Git LFS 
(`git lfs track "artifacts/*.npy" "artifacts/*.pkl"`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
# place: data/candidates.jsonl  and  data/job_description.docx
```

## Reproduce the submission

**Single command (Stage 3 — runs in the 5-min window using committed artifacts):**

```bash
python -m src.rank --candidates data/candidates.jsonl --out submission.csv
```

**From scratch (offline pre-computation — may exceed 5 min, downloads the model):**

```bash
# full offline build: embeddings + features + BM25 index
python -m src.precompute --candidates data/candidates.jsonl --jd data/job_description.docx
# then the timed ranking step
python -m src.rank --candidates data/candidates.jsonl --out submission.csv
python validate_submission.py submission.csv
```

Helper modes (no re-embedding):
- `--features-only` — regenerate `features.jsonl` after changing feature/scoring logic.
- `--bm25-only` — build just the BM25 index (~30-60 s).

## Sandbox / demo

A hosted sandbox that runs this exact pipeline on a <=100-candidate sample lives
in `deploy/` (Streamlit + HuggingFace Spaces apps, a 100-candidate sample, and
step-by-step deploy guides). See `deploy/STREAMLIT_DEPLOY.md`.

## What gets submitted (per submission_spec section 10)

1. **CSV** — top-100 ranking (`candidate_id,rank,score,reasoning`), validator-clean.
2. **Portal metadata** — mirrored in `submission_metadata.yaml`.
3. **This repo** — single reproduce command above; artifacts committed; deps pinned.
4. **Sandbox link** — the hosted `deploy/` app (<=100 sample, end-to-end, <=5 min).
5. **AI-tools declaration** — in `submission_metadata.yaml`.

Run `python validate_submission.py submission.csv` before every upload (3 max).