"""
features.py — turn a raw candidate dict into the signals the ranker uses.

The whole file obeys one rule: read career_history TEXT and TITLE first.
The skills array is used only as weak corroboration and to detect stuffing.
Education years are ignored entirely (the EDA proved they're noise).
"""
from datetime import date

from . import config as C

TODAY = date.today()


def _d(s):
    y, m, d = map(int, s.split("-"))
    return date(y, m, d)


def work_text(c):
    """What the candidate actually DID: job titles + descriptions. NO summary.
    This is the trustworthy text. The 05 probe proved that embedding this
    (instead of summary-inclusive text) cleanly separates real engineers from
    stuffers, because stuffers pad the summary, not the work descriptions."""
    parts = []
    for j in c.get("career_history", []):
        parts.append(j.get("title", "") or "")
        parts.append(j.get("description", "") or "")
    return " ".join(parts).lower()


def summary_text(c):
    """What the candidate CLAIMS: summary + headline. Aspirational, gameable.
    Used ONLY to detect the claim-vs-work gap, never as positive evidence."""
    p = c.get("profile", {})
    return (f"{p.get('summary', '') or ''} {p.get('headline', '') or ''}").lower()


def _term_hits(text, terms):
    return sum(text.count(t) for t in terms)


# Honeypot rules (only the ones that survived the harness) 
def honeypot_flags(c):
    flags = []
    skills = c.get("skills", [])
    if sum(1 for s in skills if s.get("proficiency") in ("expert", "advanced")
           and s.get("duration_months", 0) == 0) >= 5:
        flags.append("expert_zero_evidence")
    ch = c.get("career_history", [])
    yoe = c["profile"]["years_of_experience"]
    for j in ch:
        sd = _d(j["start_date"])
        ed = _d(j["end_date"]) if j["end_date"] else TODAY
        if j["duration_months"] - (ed - sd).days / 30.4 > 14:
            flags.append("duration_gt_span")
            break
    if ch and max(j["duration_months"] for j in ch) / 12 - yoe > 3:
        flags.append("job_longer_than_career")
    if sum(j["duration_months"] for j in ch) / 12 - yoe > 5:
        flags.append("career_exceeds_yoe")
    return flags


# Behavioral signals, sentinel-aware 
def behavioral(c):
    s = c["redrob_signals"]
    days_inactive = (TODAY - _d(s["last_active_date"])).days
    return {
        "resp_rate": s["recruiter_response_rate"],
        "days_inactive": days_inactive,
        "open_to_work": bool(s["open_to_work_flag"]),
        "completeness": s["profile_completeness_score"],
        "notice_days": s["notice_period_days"],
        # sentinel-aware: None means "no data", handled as neutral, never as low
        "github": (None if s["github_activity_score"] == C.SENTINEL
                   else s["github_activity_score"]),
        "offer_accept": (None if s["offer_acceptance_rate"] == C.SENTINEL
                         else s["offer_acceptance_rate"]),
    }


# Structured fit inputs (work text + title, never skills array) 
def fit_inputs(c):
    p = c["profile"]
    title = (p.get("current_title", "") or "").lower()
    work = work_text(c)                 # what they DID (no summary)
    loc = (p.get("location", "") or "").lower()
    country = (p.get("country", "") or "").lower()

    # "only ever services" = every company in the career is a services firm.
    comp_list = [j.get("company", "").lower() for j in c.get("career_history", [])]
    only_services = bool(comp_list) and all(
        any(sc in comp for sc in C.SERVICES_COMPANIES) for comp in comp_list
    )

    # AI signal is counted from WORK ONLY. The skills array and the summary are where stuffing lives, so neither earns positive credit here.
    skill_names = " ".join(s["name"].lower() for s in c.get("skills", []))
    skills_ai = _term_hits(skill_names, C.MUST_HAVE_TERMS)
    work_ai = _term_hits(work, C.MUST_HAVE_TERMS)

    # Stuffer signature (matches eda/04): >=4 AI terms in the skills array but
    # ZERO AI in the actual work. NOTE: we deliberately do NOT trigger on the summary -- a real AI engineer naturally describes AI in their summary, so using summary-claims here wrongly flagged 42 genuine ML engineers.
    is_stuffer = skills_ai >= 4 and work_ai == 0

    ch = c.get("career_history", [])
    titles_all = " ".join((j.get("title", "") or "").lower() for j in ch)
    insts = " ".join((e.get("institution", "") or "").lower()
                     for e in c.get("education", []))

    # JD disqualifier: pure research, no production. Research-flavored career + academic affiliation + NO production language anywhere in the work.
    research_only = (
        any(t in titles_all for t in C.RESEARCH_TERMS)
        and not any(t in work for t in C.PRODUCTION_TERMS)
        and (any(t in insts for t in C.RESEARCH_INSTITUTIONS)
             or any(t in (p.get("current_industry", "") or "").lower()
                    for t in ["research", "academ"]))
    )

    # JD disqualifier: moved out of coding (architecture / tech-lead / mgmt).
    # Conservative: only the CURRENT title, and only pure non-coding titles.
    noncoding_lead = any(t in title for t in C.NONCODING_TITLE_TERMS)

    # JD disqualifier: title-chaser / job-hopper.
    durs = [j.get("duration_months", 0) for j in ch]
    job_hopper = (len(durs) >= C.JOB_HOP_MIN_JOBS
                  and (sum(durs) / len(durs)) < C.JOB_HOP_MAX_AVG_MONTHS)

    # Fields used only to build specific, varied reasoning strings 
    arc = [j.get("title", "") for j in ch[:3] if j.get("title")]
    real_ai_skills = [s["name"] for s in c.get("skills", [])
                      if any(t in s["name"].lower() for t in C.MUST_HAVE_TERMS)]

    return {
        "title": title,
        "current_title": p.get("current_title", ""),     # original case, for reasoning
        "current_company": p.get("current_company", ""),
        "industry": p.get("current_industry", ""),
        "arc": arc,                                       # first 1-3 career titles
        "real_ai_skills": real_ai_skills[:3],
        "must_have_hits": work_ai,                 # from WORK, not skills/summary
        "nice_to_have_hits": _term_hits(work, C.NICE_TO_HAVE_TERMS),
        "nontech_title": any(t in title for t in C.NONTECH_TITLE_TERMS),
        "off_domain": (any(t in work for t in C.OFF_DOMAIN_TERMS)
                       and work_ai == 0),
        "only_services_career": only_services,
        "research_only": research_only,
        "noncoding_lead": noncoding_lead,
        "job_hopper": job_hopper,
        "in_india": country == "india",
        "preferred_city": any(ci in loc for ci in C.PREFERRED_CITIES),
        "willing_relocate": bool(c["redrob_signals"]["willing_to_relocate"]),
        "yoe": p.get("years_of_experience", 0),
        "is_stuffer": is_stuffer,
    }
