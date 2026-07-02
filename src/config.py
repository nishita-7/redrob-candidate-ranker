"""
config.py — the bridge between EDA and the ranker.

Every constant here is a FINDING from eda_out/, not a guess. When you re-run the
EDA on your machine and the numbers shift, update them here. The ranker reads
this file; it never hardcodes thresholds inline.

You will be asked to defend these choices at Stage 5. Make sure you can say,
for each number, "this came from the data, here's the percentile."
"""

# sentinels (eda_out/02_signals.json) 
# -1 means MISSING, not a low score. ~65% / ~60% of the pool. NEVER score as low.
SENTINEL = -1
GITHUB_MISSING_PCT = 64.6
OFFER_MISSING_PCT = 59.6

# Behavioral percentiles (eda_out/02_signals.json) 
# Used to build the behavioral MULTIPLIER off real distributions, not arbitrary cutoffs. p25/median/p75 of the pool.
RESP_RATE = {"p25": 0.25, "median": 0.44, "p75": 0.62}
DAYS_INACTIVE = {"p25": 83, "median": 136, "p75": 193}   # 30% are >180d ("not available")
COMPLETENESS = {"p25": 42.2, "median": 56.8, "p75": 71.6}

# Pool shape (eda_out/01_overview.json) 
INDIA_PCT = 75.1
YOE = {"p10": 2.2, "median": 6.8, "p90": 13.0}
# certifications (75% empty) and skill_assessment_scores (76% empty) are too sparse to rely on as features. Do not build core logic on them.
SPARSE_FIELDS = ("certifications", "skill_assessment_scores")

# Honeypot rules that SURVIVED the harness (eda_out/03_honeypots.json) 
# These fire on weak candidates, spare strong ones, and hit ZERO genuine ML candidates. degree_inversion is intentionally EXCLUDED — the harness proved it is noise (fires equally on good/bad, would have deleted 86 real engineers).
# Honeypots get a PENALTY, never a delete.
HONEYPOT_RULES = (
    "expert_zero_evidence",
    "duration_gt_span",
    "job_longer_than_career",
    "career_exceeds_yoe",
)
HONEYPOT_PENALTY = 1.0   # large enough to bury, applied to the final score

# JD rubric (parsed from job_description.md, NOT embedded as a blob) 
# Embeddings can't represent the "we do NOT want" section. These are executable.
# Source: the Senior AI Engineer JD. Tune wording to your own reading of it.
MUST_HAVE_TERMS = [
    "retrieval", "ranking", "ranker", "recommendation", "recsys", "embedding",
    "vector", "semantic", "information retrieval", "search relevance",
    "learning to rank", "relevance", "matching", "personaliz", "nlp",
    # vector DBs / hybrid search infra (JD must-have #2)
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "hybrid search", "vector database", "vector db", "bm25",
    # ranking-eval frameworks (JD must-have #4)
    "ndcg", "mrr", "mean reciprocal", "map@", "a/b test", "ab test",
    "offline evaluation", "eval framework", "evaluation framework",
]
NICE_TO_HAVE_TERMS = [
    "lora", "qlora", "peft", "fine-tun", "xgboost", "distributed",
    "inference optimization", "hr-tech", "marketplace", "open source",
]
# Disqualifiers / negative signals (read from TITLE + career text):
NONTECH_TITLE_TERMS = [
    "manager", "sales", "marketing", "customer support", "operations",
    "recruiter", "accountant", "content writer", "designer", "executive",
    "hr ",
]
OFF_DOMAIN_TERMS = ["computer vision", "image classification", "speech",
                    "robotics"]   # only disqualifying WITHOUT nlp/ir exposure
SERVICES_COMPANIES = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "mindtree", "ltimindtree", "hcl", "tech mahindra",
]

# Pure-research signal (JD disqualifier: research-only, no production).
RESEARCH_TERMS = ["research scientist", "researcher", "postdoc", "phd student",
                  "research fellow", "research assistant", "academic"]
PRODUCTION_TERMS = ["production", "deployed", "shipped", "launched", "in prod",
                    "serving", "real users", "at scale", "live"]
RESEARCH_INSTITUTIONS = ["university", "institute of technology", "iit ", "iisc",
                         "research lab", "laboratory", "labs"]

# Moved-out-of-coding signal (JD disqualifier: no production code in 18 months).
# Kept conservative so hands-on "Staff/Principal ENGINEER" is NOT caught.
NONCODING_TITLE_TERMS = ["architect", "tech lead", "engineering manager",
                         "director", "vp ", "head of", "principal architect"]

# Title-chaser / job-hopper (JD: wants 3+ yr tenure, not 1.5-yr hops).
JOB_HOP_MAX_AVG_MONTHS = 18      # avg tenure below this, across >=3 jobs
JOB_HOP_MIN_JOBS = 3

# Notice period (JD: sub-30 preferred; 30+ "bar gets higher").
NOTICE_GOOD_DAYS = 30
NOTICE_HIGH_DAYS = 90

# Geography (no visa sponsorship; Pune/Noida preferred)
PREFERRED_CITIES = ["pune", "noida", "hyderabad", "mumbai", "delhi",
                    "bengaluru", "bangalore", "gurgaon", "gurugram"]

# Scoring weights (THE thing you tune and must defend) 
W_SEMANTIC = 0.45    # cosine(JD, career_text)
W_FIT = 0.55         # structured rubric match (title + career)
# Behavioral multiplier ranges roughly [0.55, 1.15]; honeypot penalty is flat.