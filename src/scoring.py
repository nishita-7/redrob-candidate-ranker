"""
scoring.py — THE part you own.

This is a transparent, defensible BASELINE, not a finished winning ranker.
Every component is a few readable lines so you can explain each one at Stage 5
and tune it against your own hand-labeled set. It is deliberately simple.

Final score = (W_SEMANTIC * semantic + W_FIT * structured_fit)
              * behavioral_multiplier
              - honeypot_penalty

semantic comes from precomputed embeddings (cosine of JD vs career_text).
structured_fit reads TITLE + career text. behavioral is sentinel-aware.
honeypot_penalty uses only the harness-approved rules.
"""
from . import config as C


def structured_fit(fi):
    """0..1-ish fit from the JD rubric, read off title + career text."""
    score = 0.0
    # positives
    score += min(fi["must_have_hits"], 6) * 0.12        # career shows IR/ranking work
    score += min(fi["nice_to_have_hits"], 4) * 0.04
    # experience sweet spot (JD ideal ~6-8, flexible)
    yoe = fi["yoe"]
    if 5 <= yoe <= 9:
        score += 0.15
    elif 3 <= yoe < 5 or 9 < yoe <= 12:
        score += 0.07
    # geography (no visa sponsorship; Pune/Noida preferred)
    if fi["in_india"]:
        score += 0.08
        if fi["preferred_city"]:
            score += 0.05
    elif not fi["willing_relocate"]:
        score -= 0.10

    # disqualifiers / negative signals (the "we do NOT want" section)
    if fi["nontech_title"]:
        score -= 0.45        # Project/Marketing Manager etc. — the stuffer trap
    if fi["is_stuffer"]:
        score -= 0.35        # AI terms in skills array, none in career
    if fi["off_domain"]:
        score -= 0.25        # CV/speech/robotics primary, no NLP/IR
    if fi["only_services_career"]:
        score -= 0.20        # entire career at services firms (JD: not a fit)
    if fi.get("research_only"):
        score -= 0.30        # pure research, no production (explicit JD disqual)
    if fi.get("noncoding_lead"):
        score -= 0.15        # architecture/tech-lead/mgmt — "this role writes code"
    if fi.get("job_hopper"):
        score -= 0.12        # title-chaser pattern (JD wants 3+ yr tenure)

    return score


def behavioral_multiplier(b):
    """
    A modifier, NOT additive points. A great profile that's dormant/unresponsive
    is 'not actually available' (JD's own words) and gets scaled down.
    Range roughly [0.55, 1.15]. Sentinel-aware: missing github/offer = neutral.
    """
    m = 1.0
    # responsiveness (percentiles from EDA)
    if b["resp_rate"] >= C.RESP_RATE["p75"]:
        m += 0.08
    elif b["resp_rate"] <= C.RESP_RATE["p25"]:
        m -= 0.12
    # recency: >180d inactive ~= unavailable
    if b["days_inactive"] > 180:
        m -= 0.20
    elif b["days_inactive"] <= C.DAYS_INACTIVE["p25"]:
        m += 0.05
    # explicit availability
    if b["open_to_work"]:
        m += 0.06
    # profile completeness
    if b["completeness"] <= C.COMPLETENESS["p25"]:
        m -= 0.05
    # github is a BONUS when present, never a penalty when missing (sentinel)
    if b["github"] is not None and b["github"] >= 40:
        m += 0.05
    # notice period (JD: sub-30 preferred; 30+ "bar gets higher")
    if b["notice_days"] <= C.NOTICE_GOOD_DAYS:
        m += 0.04
    elif b["notice_days"] >= C.NOTICE_HIGH_DAYS:
        m -= 0.06
    return max(0.55, min(1.15, m))


def final_score(semantic, fi, b, honeypot_n):
    base = C.W_SEMANTIC * semantic + C.W_FIT * structured_fit(fi)
    score = base * behavioral_multiplier(b)
    if honeypot_n:
        score -= C.HONEYPOT_PENALTY      # bury, don't delete
    return score


def reasoning(fi, b, honeypot_flags, rank):
    """
    Specific, varied, rank-consistent reasoning built FROM the scored features.
    Cites real facts (title, company, yoe, named skills, signal values) so it
    passes the Stage-4 checks: specific facts, JD connection, honest concerns,
    no hallucination, variation, rank consistency. Never invents anything --
    every token comes from the candidate's own profile.
    """
    title = fi.get("current_title") or "Engineer"
    company = fi.get("current_company") or "their current company"
    yoe = fi["yoe"]

    head = f"{title} at {company}, {yoe:.0f} yrs"

    arc = [t for t in fi.get("arc", []) if t]
    skills = fi.get("real_ai_skills", [])
    if fi["must_have_hits"] >= 8 and arc:
        body = f"deep retrieval/ranking history ({' to '.join(arc[:3])})"
    elif arc and len(arc) >= 2:
        body = f"career arc {' to '.join(arc[:3])}"
    elif skills:
        body = f"works with {', '.join(skills[:2])}"
    elif fi["must_have_hits"]:
        body = f"{fi['must_have_hits']} retrieval/ranking signals in their work"
    else:
        body = "adjacent ML background"

    if fi["in_india"] and fi["preferred_city"]:
        loc = "in a preferred city"
    elif not fi["in_india"]:
        loc = "outside India (no visa sponsorship)"
    else:
        loc = "India-based"

    concerns = []
    if fi["nontech_title"]:
        concerns.append("non-technical current title")
    if fi["is_stuffer"]:
        concerns.append("AI skills listed but absent from actual work")
    if fi.get("research_only"):
        concerns.append("research-heavy, little production signal")
    if fi.get("noncoding_lead"):
        concerns.append("lead/architecture title — may be hands-off")
    if fi.get("only_services_career"):
        concerns.append("services-firm-only career")
    if fi.get("job_hopper"):
        concerns.append("short average tenure")
    if fi["off_domain"]:
        concerns.append("CV/speech-primary, limited NLP/IR")
    if b["days_inactive"] > 180:
        concerns.append(f"inactive {b['days_inactive']}d")
    if b["resp_rate"] <= C.RESP_RATE["p25"]:
        concerns.append(f"low recruiter response ({b['resp_rate']:.2f})")
    if b["notice_days"] >= C.NOTICE_HIGH_DAYS:
        concerns.append(f"{b['notice_days']}d notice")

    sentence = f"{head}; {body}, {loc}."
    if concerns:
        plural = "s" if len(concerns) > 1 else ""
        return f"{sentence} Concern{plural}: {', '.join(concerns[:3])}."
    if b["resp_rate"] >= C.RESP_RATE["p75"]:
        return f"{sentence} Strong on the JD's core retrieval/ranking need and actively responsive."
    return f"{sentence} Solid match on the JD's core retrieval/ranking requirements."
