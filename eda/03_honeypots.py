"""
03 — Honeypots, WITH a validation harness.

The lesson that produced this file: a rule that flags 14% of the pool as "impossible" is almost always detecting a DATA ARTIFACT, not fabrication.
The only way to tell is to ask: does the rule fire MORE on weak candidates than on strong ones? 
A real impossibility rule spares good candidates. 
A noise rule fires on everyone equally.

So every rule below is scored on:
  - total flagged
  - fire-rate among LOW-quality candidates vs HIGH-quality (career-text proxy)
  - how many genuine AI/ML-titled candidates it would wrongly hit
and gets an automatic verdict: KEEP / REJECT-NOISE / REVIEW.

`degree_inversion` is included ON PURPOSE as a known-bad rule so you can see the harness reject it. Do not ship it.

Outputs eda_out/03_honeypots.json
"""
import json
import os
from datetime import date

from common import (iter_candidates, parse_args, ai_proxy_score,
                    is_ai_titled, OUT_DIR)

TODAY = date.today()


def _d(s):
    y, m, d = map(int, s.split("-"))
    return date(y, m, d)


# Candidate rules: each returns True if the profile is "impossible"
def r_expert_zero_evidence(c):
    z = [s for s in c.get("skills", [])
         if s.get("proficiency") in ("expert", "advanced")
         and s.get("duration_months", 0) == 0]
    return len(z) >= 5


def r_duration_gt_span(c):
    for j in c.get("career_history", []):
        sd = _d(j["start_date"])
        ed = _d(j["end_date"]) if j["end_date"] else TODAY
        if j["duration_months"] - (ed - sd).days / 30.4 > 14:
            return True
    return False


def r_job_longer_than_career(c):
    ch = c.get("career_history", [])
    if not ch:
        return False
    yoe = c["profile"]["years_of_experience"]
    return max(j["duration_months"] for j in ch) / 12 - yoe > 3


def r_career_exceeds_yoe(c):
    yoe = c["profile"]["years_of_experience"]
    cm = sum(j["duration_months"] for j in c.get("career_history", [])) / 12
    return cm - yoe > 5


def r_degree_inversion(c):
    # KNOWN-BAD. Education years are noise in this dataset. Harness should REJECT.
    LEVEL = {"Ph.D": 4, "M.E.": 3, "M.S.": 3, "M.Sc": 3, "M.Tech": 3,
             "B.Tech": 2, "B.E.": 2, "B.Sc": 2}
    es = [(LEVEL.get(e.get("degree"), 0), e.get("end_year") or 0)
          for e in c.get("education", []) if e.get("degree") in LEVEL]
    for i in range(len(es)):
        for j in range(len(es)):
            if es[i][0] > es[j][0] and es[i][1] < es[j][1]:
                return True
    return False


RULES = {
    "expert_zero_evidence": r_expert_zero_evidence,
    "duration_gt_span": r_duration_gt_span,
    "job_longer_than_career": r_job_longer_than_career,
    "career_exceeds_yoe": r_career_exceeds_yoe,
    "degree_inversion__KNOWN_BAD": r_degree_inversion,
}


def main():
    args = parse_args()
    proxies = []
    rows = []  # (proxy_score, is_ai_titled, {rule_name: bool})
    for c in iter_candidates(args.data):
        proxies.append(ai_proxy_score(c))
        rows.append((proxies[-1], is_ai_titled(c),
                     {name: fn(c) for name, fn in RULES.items()}))
    n = len(rows)

    sp = sorted(proxies)
    q1, q3 = sp[n // 4], sp[3 * n // 4]
    n_low = sum(1 for p in proxies if p <= q1)
    n_high = sum(1 for p in proxies if p >= q3)

    report = {"n": n, "proxy_quartiles": {"q1": q1, "q3": q3}, "rules": {}}
    for name in RULES:
        total = sum(1 for (_, _, d) in rows if d[name])
        fire_low = sum(1 for (p, _, d) in rows if d[name] and p <= q1)
        fire_high = sum(1 for (p, _, d) in rows if d[name] and p >= q3)
        ai_hit = sum(1 for (_, ai, d) in rows if d[name] and ai)
        rate_low = fire_low / max(n_low, 1)
        rate_high = fire_high / max(n_high, 1)
        ratio = (rate_low / rate_high) if rate_high > 0 else float("inf")

        # The quality-ratio is only trustworthy when enough candidates fire.
        # A rule firing on a handful (e.g. expert_zero_evidence) can't be split into quartiles meaningfully -- inspect those few by hand instead.
        if total == 0:
            verdict = "EMPTY"
        elif total < 30:
            verdict = ("TOO FEW TO JUDGE STATISTICALLY (n<30) — "
                       "inspect the flagged profiles by hand")
        elif ratio == float("inf") or ratio >= 3:
            verdict = "KEEP — fires on weak, spares strong"
        elif 0.7 <= ratio <= 1.5:
            verdict = "REJECT — NOISE (fires equally on strong & weak)"
        else:
            verdict = "REVIEW — weak/ambiguous signal"

        report["rules"][name] = {
            "total_flagged": total,
            "pct_of_pool": round(100 * total / n, 2),
            "fire_rate_low_quality": round(rate_low, 4),
            "fire_rate_high_quality": round(rate_high, 4),
            "low_over_high_ratio": ("inf" if ratio == float("inf") else round(ratio, 2)),
            "genuine_ai_candidates_wrongly_flagged": ai_hit,
            "verdict": verdict,
        }

    out = os.path.join(OUT_DIR, "03_honeypots.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print("\nHOW TO READ THIS:")
    print("  KEEP   -> safe to use as a honeypot PENALTY (not a delete).")
    print("  REJECT -> the rule is a data artifact; using it deletes real candidates.")
    print("  'genuine_ai_candidates_wrongly_flagged' is the body count of a bad rule.")
    print(f"\n-> wrote {out}")


if __name__ == "__main__":
    main()
