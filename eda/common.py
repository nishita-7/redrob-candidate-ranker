"""
Shared helpers for the EDA package.

Design rule that drives everything here: SIGNAL LIVES IN career_history TEXT, not in the skills array (gameable) or education years (noise in this dataset).
The AI-fit proxy below is built ONLY from career text + summary, on purpose.
"""
import json
import os
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "candidates.jsonl")
OUT_DIR = os.path.join(ROOT, "eda_out")
os.makedirs(OUT_DIR, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH,
                    help="path to candidates.jsonl (default: data/candidates.jsonl)")
    return ap.parse_args()


def iter_candidates(path):
    # Stream one candidate per line. Never loads the whole file into memory.
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


# AI/IR fit proxy: coarse, transparent, CAREER-TEXT-ONLY 
# This is NOT the ranker. It exists only so the honeypot harness can ask: "does this rule fire more on weak candidates than strong ones?"
AI_TERMS = [
    "retrieval", "ranking", "ranker", "recommend", "recsys", "embedding",
    "semantic", "vector", "information retrieval", "learning to rank",
    "relevance", "personaliz", "search relevance", "nlp", "llm", "transformer",
    "matching", "fine-tun", "bm25", "elasticsearch", "faiss",
]


def career_text(c):
    parts = []
    for j in c.get("career_history", []):
        parts.append(j.get("title", "") or "")
        parts.append(j.get("description", "") or "")
    p = c.get("profile", {})
    parts.append(p.get("summary", "") or "")
    parts.append(p.get("headline", "") or "")
    return " ".join(parts).lower()


def ai_proxy_score(c):
    # Count of AI/IR terms in CAREER TEXT (not the skills array).
    t = career_text(c)
    return sum(t.count(term) for term in AI_TERMS)


AI_TITLE_TERMS = [
    "ai engineer", "ml engineer", "machine learning", "data scient", "nlp",
    "retrieval", "recommendation", "search engineer", "applied ml",
    "applied scientist", "research engineer",
]


def is_ai_titled(c):
    t = (c.get("profile", {}).get("current_title", "") or "").lower()
    return any(x in t for x in AI_TITLE_TERMS)
