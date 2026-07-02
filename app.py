"""
streamlit_app.py — Streamlit Community Cloud sandbox for the Redrob ranker.

Runs the REAL pipeline (the same src/ code that produces the graded submission)
on a small bundled sample, end to end:
  precompute (embed sample + features + BM25)  ->  rank (hybrid RRF)  ->  top 100

The full-pool run uses precomputed artifacts under the 5-min/CPU/no-network
limit; here we run precompute live on a 100-candidate sample so it finishes in
seconds and judges can see the whole system work.
"""
import os
import subprocess
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))


def _find(name):
    """Look for a bundled file at the repo root or in deploy/."""
    for cand in (os.path.join(ROOT, name), os.path.join(ROOT, "deploy", name)):
        if os.path.exists(cand):
            return cand
    return os.path.join(ROOT, name)


SAMPLE = _find("sample_candidates.jsonl")
JD = _find("job_description.md")
OUT = os.path.join(ROOT, "submission_sample.csv")


def _count(path):
    return sum(1 for line in open(path, encoding="utf-8") if line.strip())


@st.cache_data(show_spinner=False)
def run_ranker():
    """Run the real precompute + rank pipeline on the sample. Cached so it only
    runs once per session."""
    env = dict(os.environ, PYTHONPATH=ROOT)
    subprocess.run(
        [sys.executable, "-m", "src.precompute", "--candidates", SAMPLE, "--jd", JD],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=600,
    )
    subprocess.run(
        [sys.executable, "-m", "src.rank", "--candidates", SAMPLE, "--out", OUT],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=300,
    )
    return pd.read_csv(OUT)


st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯", layout="wide")

st.title("Redrob Intelligent Candidate Discovery — live sandbox")
st.markdown(
    "Ranks candidates for the **Senior AI Engineer** JD by reading career "
    "history (not the gameable skills array), with hybrid BM25 + dense "
    "retrieval, structured JD-rubric scoring, behavioral signals, and "
    "profile-consistency checks.\n\n"
    "This is the same `src/` code that produces the full 100k submission — only "
    "the input size differs (so the embedding step finishes in seconds)."
)

if st.button("Run ranker on sample", type="primary"):
    with st.spinner("Embedding sample, building features + BM25, ranking… "
                    "(first run downloads the model, ~1 min)"):
        try:
            df = run_ranker()
        except subprocess.CalledProcessError as e:
            st.error("Pipeline error:")
            st.code(e.stderr[-2000:])
            st.stop()
        except subprocess.TimeoutExpired:
            st.error("Timed out — the model download can be slow on first run. "
                     "Reload and try again.")
            st.stop()

    st.success(f"Ranked the top {len(df)} from a {_count(SAMPLE)}-candidate sample.")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False),
                       file_name="submission_sample.csv", mime="text/csv")