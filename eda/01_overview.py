"""
01 — Pool overview.
Answers: who is in this pool, and which fields are even reliably populated?
Outputs eda_out/01_overview.json
"""
import json
import os
import statistics as st
from collections import Counter

from common import iter_candidates, parse_args, pct, OUT_DIR


def main():
    args = parse_args()
    n = 0
    ctry, ind, size = Counter(), Counter(), Counter()
    yoe, compl = [], []
    empty = {"education": 0, "certifications": 0, "languages": 0,
             "skill_assessment_scores": 0}

    for c in iter_candidates(args.data):
        n += 1
        p, s = c["profile"], c["redrob_signals"]
        ctry[p["country"]] += 1
        ind[p["current_industry"]] += 1
        size[p["current_company_size"]] += 1
        yoe.append(p["years_of_experience"])
        compl.append(s["profile_completeness_score"])
        if not c.get("education"):
            empty["education"] += 1
        if not c.get("certifications"):
            empty["certifications"] += 1
        if not c.get("languages"):
            empty["languages"] += 1
        if not s.get("skill_assessment_scores"):
            empty["skill_assessment_scores"] += 1

    yoe.sort()
    compl.sort()
    report = {
        "n": n,
        "geography": {
            "india_pct": round(100 * ctry["India"] / n, 1),
            "top_countries": ctry.most_common(8),
        },
        "yoe": {"p10": pct(yoe, .1), "median": st.median(yoe),
                "p90": pct(yoe, .9), "min": yoe[0], "max": yoe[-1]},
        "industry_pct": [(k, round(100 * v / n, 1)) for k, v in ind.most_common()],
        "company_size": dict(size),
        "field_emptiness_pct": {k: round(100 * v / n, 1) for k, v in empty.items()},
        "profile_completeness": {"p10": pct(compl, .1),
                                 "median": st.median(compl), "p90": pct(compl, .9)},
    }
    out = os.path.join(OUT_DIR, "01_overview.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"\n-> wrote {out}")


if __name__ == "__main__":
    main()
