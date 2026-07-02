"""
05 — Cosine sanity probe (OPTIONAL; needs `pip install sentence-transformers numpy python-docx`).

Your regression test for the keyword-matching disease. It embeds the JD against two representations of each candidate:

  naive_blob = headline + summary + SKILL NAMES   (the tempting cheap one;
                                                   includes the aspirational                                                    summary the stuffers abuse)
  work_text  = job titles + descriptions ONLY     (what they actually DID;                                                    no summary)

It compares genuine strong candidates against the REAL keyword-stuffers found by 04 (loaded from eda_out/04_keyword_stuffers.csv). 
The pattern to look for: stuffers ride high under naive_blob and SINK under work_text. 
That contrast is the whole argument for not ranking on embeddings.

Setup before running:
  - put the JD at data/job_description.docx
  - run 04_keyword_stuffers.py first so eda_out/04_keyword_stuffers.csv exists
"""
import csv
import os
import sys

from common import iter_candidates, parse_args, ai_proxy_score, ROOT

N_PER_SIDE = 10   # 10 real vs 10 stuffers -> stable enough to screenshot


def naive_blob(c):
    # The trap: skill names + the aspirational summary.
    p = c["profile"]
    skills = " ".join(s["name"] for s in c.get("skills", []))
    return f"{p['headline']} {p['summary']} {skills}".lower()


def work_text(c):
    # The fix: titles + descriptions only. NO summary (the stuffer's weapon).
    return " ".join((j.get("title", "") + " " + j.get("description", ""))
                    for j in c.get("career_history", [])).lower()


def load_stuffer_ids():
    path = os.path.join(ROOT, "eda_out", "04_keyword_stuffers.csv")
    if not os.path.exists(path):
        sys.exit("Run 04_keyword_stuffers.py first (eda_out/04_keyword_stuffers.csv missing).")
    with open(path, encoding="utf-8") as f:
        return {row["candidate_id"] for row in csv.DictReader(f)}


def main():
    args = parse_args()

    # JD straight from the docx, anchored to repo root (works from any cwd) 
    jd_path = os.path.join(ROOT, "data", "job_description.docx")
    if not os.path.exists(jd_path):
        sys.exit(f"Put the JD at {jd_path} first.")
    from docx import Document
    jd = "\n".join(p.text for p in Document(jd_path).paragraphs)

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np  # noqa: F401  (used implicitly by encode output)
    except ImportError:
        sys.exit("pip install sentence-transformers numpy")

    stuffer_ids = load_stuffer_ids()

    # Scan a slice of the pool, then pick the strongest real candidates and the first genuine stuffers (from 04) we find.
    pool = []
    for i, c in enumerate(iter_candidates(args.data)):
        pool.append(c)
        if i >= 30000:
            break

    real = sorted([c for c in pool if c["candidate_id"] not in stuffer_ids],
                  key=ai_proxy_score, reverse=True)[:N_PER_SIDE]
    stuffers = [c for c in pool if c["candidate_id"] in stuffer_ids][:N_PER_SIDE]
    probe = [("REAL", c) for c in real] + [("STUFFER", c) for c in stuffers]

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    jd_emb = model.encode(jd, normalize_embeddings=True)

    def rank_by(textfn):
        embs = model.encode([textfn(c)[:2000] for _, c in probe],
                            normalize_embeddings=True)
        sims = embs @ jd_emb
        order = sorted(range(len(probe)), key=lambda i: -sims[i])
        return order, sims

    def show(title, textfn):
        print(f"\n=== {title} ===")
        order, sims = rank_by(textfn)
        for r, i in enumerate(order, 1):
            tag, c = probe[i]
            print(f"  {r:2d}. {sims[i]:.3f} [{tag:7s}] "
                  f"{c['candidate_id']} | {c['profile']['current_title']}")
        return order

    show("naive_blob (headline + summary + SKILL NAMES) — the trap", naive_blob)
    order2 = show("work_text (titles + descriptions, NO summary) — the fix", work_text)

    # quick read: how many stuffers landed in the top half of the FIX column?
    half = len(probe) // 2
    stuffers_top = sum(1 for i in order2[:half] if probe[i][0] == "STUFFER")
    print(f"\nIn the work_text (fix) top {half}: {stuffers_top} stuffers.")
    print("Lower is better. The point of the probe: cosine alone never fully "
          "separates them — which is why the ranker reads career structure, "
          "not just similarity.")


if __name__ == "__main__":
    main()