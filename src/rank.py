import argparse
import csv
import json
import os

import numpy as np

from . import config as C
from . import scoring

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def load_artifacts():
    need = ["jd.npy", "emb.npy", "features.jsonl"]
    missing = [n for n in need if not os.path.exists(os.path.join(ART, n))]
    if missing:
        raise SystemExit(f"Missing {missing} in artifacts/. Run src/precompute.py first.")
    jd = np.load(os.path.join(ART, "jd.npy"))
    emb = np.load(os.path.join(ART, "emb.npy"))
    feats = []
    with open(os.path.join(ART, "features.jsonl"), encoding="utf-8") as f:
        for line in f:
            feats.append(json.loads(line))
    return jd, emb, feats


def _ranks_desc(scores):
    # 1-based rank position per index, by descending score.
    order = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _hybrid_retrieval(sims, feats, k=60):
    # RRF-fuse dense cosine with BM25 over work text. Returns a [0,1] score per candidate. Falls back to normalized cosine if bm25.pkl is absent.
    bm25_path = os.path.join(ART, "bm25.pkl")
    if not os.path.exists(bm25_path):
        # no BM25 index -> just min-max the cosine so the scale matches
        s = sims - sims.min()
        return s / (s.max() or 1.0)

    import pickle
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    # BM25 query = the JD's must-have + nice-to-have terms (principled, not a hand-tuned keyword soup). Multi-word terms are split into tokens.
    query = " ".join(C.MUST_HAVE_TERMS + C.NICE_TO_HAVE_TERMS).split()
    bm25_scores = np.asarray(bm25.get_scores(query), dtype=np.float64)

    # Defensive: BM25 doc order must match feats/emb order (file order). If the index was built on a different candidate count, fall back to cosine.
    if len(bm25_scores) != len(sims):
        s = sims - sims.min()
        return s / (s.max() or 1.0)

    dense_rank = _ranks_desc(sims)
    bm25_rank = _ranks_desc(bm25_scores)
    rrf = 1.0 / (k + dense_rank) + 1.0 / (k + bm25_rank)
    return rrf / rrf.max()      # normalize to [0,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="kept for the canonical reproduce command; rank reads artifacts/")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    jd, emb, feats = load_artifacts()

    # Dense retrieval: cosine of JD vs every candidate (one matmul).
    sims = emb @ jd   # vectors normalized -> cosine

    # Hybrid retrieval: fuse dense cosine with BM25-over-work-text via RRF.
    # BM25 catches exact-terminology matches the embedding's semantics smears; the embedding catches plain-language fits BM25 misses. RRF merges by rank.
    # Falls back to pure cosine if the BM25 index wasn't built.
    retrieval = _hybrid_retrieval(sims, feats)

    scored = []
    for i, row in enumerate(feats):
        sc = scoring.final_score(
            semantic=float(retrieval[i]),     # fused retrieval score, [0,1]
            fi=row["fit"],
            b=row["beh"],
            honeypot_n=len(row["honeypots"]),
        )
        scored.append((row["candidate_id"], sc, row))

    # Rank: score DESC, then candidate_id ASC. The id-ascending secondary key is exactly what the validator requires on score ties — so equal scores always come out id-ascending and the submission is valid by construction.
    scored.sort(key=lambda x: (-x[1], x[0]))
    top = scored[:100]

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, sc, row) in enumerate(top, 1):
            reason = scoring.reasoning(row["fit"], row["beh"], row["honeypots"], rank)
            w.writerow([cid, rank, f"{sc:.6f}", reason])

    print(f"wrote {args.out} (top 100). Validate with:")
    print(f"  python validate_submission.py {args.out}")

    # Regression guard: did any known keyword-stuffer reach the top 100? eda/04 found the stuffers (>=4 AI skills, zero AI in actual work). If ANY of them rank here, the keyword-matching disease has crept back in. This is the check that holds us honest instead of trusting a gut feeling.
    stuffer_csv = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "eda_out", "04_keyword_stuffers.csv")
    if os.path.exists(stuffer_csv):
        import csv as _csv
        with open(stuffer_csv, encoding="utf-8") as sf:
            stuffer_ids = {row["candidate_id"] for row in _csv.DictReader(sf)}
        top_ids = {cid for cid, _, _ in top}
        leaked = sorted(top_ids & stuffer_ids)
        if leaked:
            print(f"\n  !!! REGRESSION: {len(leaked)} known stuffer(s) in the top 100: "
                  f"{leaked[:10]}{' ...' if len(leaked) > 10 else ''}")
            print("  Keyword matching has crept back. Investigate before submitting.")
        else:
            print("\n  regression check PASSED: 0 known stuffers in the top 100.")
    else:
        print("\n  (run eda/04 to enable the stuffer regression check)")


if __name__ == "__main__":
    main()