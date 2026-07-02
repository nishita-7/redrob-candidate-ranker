"""
02 — Behavioral signals.
Audits the -1 sentinels (github / offer history) and emits PERCENTILES for every continuous signal. 
Those percentiles are what you build the behavioral multiplier from later -- never hardcode thresholds, read them off real distributions.
Outputs eda_out/02_signals.json
"""
import json
import os
from collections import Counter
from datetime import date

from common import iter_candidates, parse_args, pct, OUT_DIR

TODAY = date.today()


def days_since(s):
    y, m, d = map(int, s.split("-"))
    return (TODAY - date(y, m, d)).days


CONT = ["recruiter_response_rate", "avg_response_time_hours",
        "profile_completeness_score", "connection_count", "endorsements_received",
        "profile_views_received_30d", "applications_submitted_30d",
        "search_appearance_30d", "saved_by_recruiters_30d",
        "interview_completion_rate", "notice_period_days"]

BOOLS = ["open_to_work_flag", "willing_to_relocate", "verified_email",
         "verified_phone", "linkedin_connected"]


def summ(a):
    if not a:
        return None
    a = sorted(a)
    return {"p10": pct(a, .1), "p25": pct(a, .25), "median": pct(a, .5),
            "p75": pct(a, .75), "p90": pct(a, .9)}


def main():
    args = parse_args()
    n = 0
    vals = {k: [] for k in CONT}
    inactive, gh_vals, oar_vals = [], [], []
    gh_missing = oar_missing = 0
    bool_true = {b: 0 for b in BOOLS}
    workmode = Counter()

    for c in iter_candidates(args.data):
        n += 1
        s = c["redrob_signals"]
        for k in CONT:
            vals[k].append(s[k])
        inactive.append(days_since(s["last_active_date"]))
        if s["github_activity_score"] == -1:
            gh_missing += 1
        else:
            gh_vals.append(s["github_activity_score"])
        if s["offer_acceptance_rate"] == -1:
            oar_missing += 1
        else:
            oar_vals.append(s["offer_acceptance_rate"])
        for b in BOOLS:
            if s[b]:
                bool_true[b] += 1
        workmode[s["preferred_work_mode"]] += 1

    report = {
        "n": n,
        "sentinels_TREAT_AS_MISSING": {
            "github_activity_score_missing_pct": round(100 * gh_missing / n, 1),
            "offer_acceptance_rate_missing_pct": round(100 * oar_missing / n, 1),
        },
        "github_activity_score_when_present": summ(gh_vals),
        "offer_acceptance_rate_when_present": summ(oar_vals),
        "days_inactive": {**summ(inactive),
                          "over_180d_pct": round(100 * sum(1 for x in inactive if x > 180) / n, 1)},
        "continuous_percentiles": {k: summ(v) for k, v in vals.items()},
        "bool_true_pct": {k: round(100 * v / n, 1) for k, v in bool_true.items()},
        "preferred_work_mode": dict(workmode),
    }
    out = os.path.join(OUT_DIR, "02_signals.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"\n-> wrote {out}")
    print("REMINDER: -1 means MISSING. Impute or flag it; never score it as a low value.")


if __name__ == "__main__":
    main()
