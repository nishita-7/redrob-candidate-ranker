"""
precompute.py — OFFLINE phase. No time/GPU/network limits apply here.

RESUMABLE: embeds in chunks and writes to disk as it goes. If it stops for any
reason, just run the same command again -- it resumes from the last saved chunk
instead of starting over. Memory stays flat regardless of pool size.

Produces:
  artifacts/jd.npy            JD embedding
  artifacts/emb.npy           candidate embeddings (written at the end)
  artifacts/features.jsonl    per-candidate features (appended as it goes)
  artifacts/_emb.dat          working memmap (safe to delete after success)

Run once (re-run to resume if interrupted):
  python -m src.precompute --candidates data/candidates.jsonl --jd data/job_description.docx

Only rank.py is on the 5-minute clock; this step may take as long as it needs.
"""
import argparse
import json
import os

import numpy as np

from . import features as F

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
EMB_MODEL = "BAAI/bge-small-en-v1.5"   # small, CPU-friendly, good for retrieval


def read_jd(path):
    """Read the JD from .docx or plain text (.md/.txt)."""
    if path.lower().endswith(".docx"):
        from docx import Document
        return "\n".join(p.text for p in Document(path).paragraphs)
    return open(path, encoding="utf-8").read()


def build_bm25(candidates_path, log=print):
    """Build a BM25 index over WORK TEXT (titles + descriptions, NOT the skills array). 
    Doc order == candidate file order, so it aligns by index with emb.npy and features.jsonl. 
    Saves artifacts/bm25.pkl. Pure-Python, no model; the IDF build takes a few minutes on 100K."""
    import pickle
    from rank_bm25 import BM25Okapi
    log("tokenizing work text for BM25 ...")
    corpus = []
    n = 0
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            corpus.append(F.work_text(c).split())   # work text only
            n += 1
            if n % 20000 == 0:
                log(f"  tokenized {n}")
    log(f"building BM25Okapi over {n} docs (a few minutes) ...")
    bm25 = BM25Okapi(corpus)
    with open(os.path.join(ART, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"wrote artifacts/bm25.pkl (vocab {len(bm25.idf)} terms)")


def count_lines(path):
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def embed_and_save(candidates_path, dim, encode_fn, chunk=2000, log=print,
                   _stop_after_chunks=None):
    # Stream candidates, embed in chunks, write each chunk to its own .npy. 
    # Resumable and portable (no memmap). Concatenates to emb.npy at the end.
    import glob
    os.makedirs(ART, exist_ok=True)
    cdir = os.path.join(ART, "_chunks")
    os.makedirs(cdir, exist_ok=True)
    feat_path = os.path.join(ART, "features.jsonl")
    emb_path = os.path.join(ART, "emb.npy")

    total = count_lines(candidates_path)
    done = count_lines(feat_path)

    # Resume only from a clean chunk boundary; truncate any partial feature tail
    if done % chunk != 0 and done < total:
        done = (done // chunk) * chunk
    if os.path.exists(feat_path):
        lines = open(feat_path, encoding="utf-8").readlines()
        if len(lines) != done:
            with open(feat_path, "w", encoding="utf-8") as fx:
                fx.writelines(lines[:done])

    if os.path.exists(emb_path) and done >= total:
        log(f"already complete: {total} candidates. Delete artifacts/ to redo.")
        return total
    if done:
        log(f"resuming from {done}/{total}")

    fout = open(feat_path, "a", encoding="utf-8")
    buf_idx, buf_text, buf_feat = [], [], []

    def flush_chunk(start):
        vecs = encode_fn(buf_text)
        np.save(os.path.join(cdir, f"emb_{start:08d}.npy"), vecs)  # emb to disk FIRST
        for r in buf_feat:
            fout.write(json.dumps(r) + "\n")
        fout.flush()                                                # then features
        return len(buf_idx)

    processed = done
    chunk_start = done
    flushed_this_run = 0
    i = -1
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            i += 1
            if i < done:
                continue
            c = json.loads(line)
            buf_idx.append(i)
            buf_text.append(F.work_text(c)[:2000])
            buf_feat.append({
                "candidate_id": c["candidate_id"],
                "fit": F.fit_inputs(c),
                "beh": F.behavioral(c),
                "honeypots": F.honeypot_flags(c),
            })
            if len(buf_idx) >= chunk:
                n = flush_chunk(chunk_start)
                processed += n
                chunk_start += n
                log(f"  embedded {processed}/{total}")
                buf_idx, buf_text, buf_feat = [], [], []
                flushed_this_run += 1
                if _stop_after_chunks and flushed_this_run >= _stop_after_chunks:
                    fout.close()
                    raise RuntimeError("simulated crash (test hook)")

    if buf_idx:
        processed += flush_chunk(chunk_start)
        log(f"  embedded {processed}/{total}")
    fout.close()

    parts = sorted(glob.glob(os.path.join(cdir, "emb_*.npy")))
    arrs = [np.load(p) for p in parts]
    np.save(emb_path, np.concatenate(arrs, axis=0))
    for p in parts:
        os.remove(p)
    os.rmdir(cdir)
    log(f"wrote {total} candidates to {ART}")
    log("artifacts: jd.npy, emb.npy, features.jsonl")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--jd", required=True)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--features-only", action="store_true",
                    help="regenerate features.jsonl from candidates WITHOUT "
                         "re-embedding (reuse existing emb.npy). Fast. Use after "
                         "changing features.py / scoring inputs.")
    ap.add_argument("--bm25-only", action="store_true",
                    help="build ONLY the BM25 index (artifacts/bm25.pkl) over "
                         "work text, without re-embedding. Run once to enable "
                         "hybrid RRF retrieval in rank.py.")
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)

    if args.bm25_only:
        build_bm25(args.candidates)
        return

    if args.features_only:
        feat_path = os.path.join(ART, "features.jsonl")
        n = 0
        with open(feat_path, "w", encoding="utf-8") as out:
            with open(args.candidates, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    c = json.loads(line)
                    out.write(json.dumps({
                        "candidate_id": c["candidate_id"],
                        "fit": F.fit_inputs(c),
                        "beh": F.behavioral(c),
                        "honeypots": F.honeypot_flags(c),
                    }) + "\n")
                    n += 1
                    if n % 10000 == 0:
                        print(f"  features {n}", end="\r")
        print(f"\nrewrote features for {n} candidates (emb.npy untouched).")
        return

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL)

    jd = read_jd(args.jd)
    jd_emb = model.encode(jd, normalize_embeddings=True).astype("float32")
    np.save(os.path.join(ART, "jd.npy"), jd_emb)
    dim = int(jd_emb.shape[0])

    def encode_fn(texts):
        return model.encode(texts, normalize_embeddings=True,
                            batch_size=64).astype("float32")

    embed_and_save(args.candidates, dim, encode_fn, chunk=args.chunk)
    build_bm25(args.candidates)


if __name__ == "__main__":
    main()