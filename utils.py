# This has been an evolving file and not all functions are used in the final code.

from __future__ import annotations

from typing import List, Tuple, Optional, Union
import logging
import os
from pathlib import Path

import numpy as np

try:
    # Optional: available in many environments
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore

from config import CHUNK_SIZE, MODEL_PATH_EMBEDDING



def get_embedder(model_path: Optional[str] = None) -> SentenceTransformer:
    
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required for embedding.")
    return SentenceTransformer(model_path or MODEL_PATH_EMBEDDING)


def read_document(path: str) -> str:
    
    path = str(path)
    if path.lower().endswith(".pdf") and PdfReader is not None:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def split_into_chunks(text: str, size: int = CHUNK_SIZE) -> List[str]:
    
    return [
        text[i : i + size].strip()
        for i in range(0, len(text), size)
        if len(text[i : i + size].strip()) > 20
    ]


def chunk_text_file(text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
    
    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step

    return chunks


def show_top_k_chunks(
    chunks: List[str],
    sims: List[float],
    k: int = 10,
    preview_len: int = 200,
) -> None:
    
    logging.info("")  # blank line for readability
    logging.info("Top-k chunks by cosine similarity:")
    top_k = min(k, len(chunks), len(sims))
    for i, (chunk, sim) in enumerate(zip(chunks[:top_k], sims[:top_k]), start=1):
        preview = (chunk or "").replace("\n", " ")
        logging.info("%2d. sim=%.3f  %s", i, float(sim), preview[:preview_len])


def display_aem(aem_obj, preview_len: int = 120) -> None:
    
    for i, chunk in enumerate(getattr(aem_obj, "get_aem")() or [], start=1):
        preview = (chunk or "").replace("\n", " ")
        logging.info("%d. %s", i, preview[:preview_len])


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def select_topk_chunks(
    question: str,
    chunks: List[str],
    embedder: SentenceTransformer,
    top_k: int,
) -> Tuple[List[str], List[float]]:
    """Cosine-only selector: return top_k most relevant chunks to *question*."""
    if not chunks or top_k <= 0:
        return [], []

    embs = embedder.encode([question] + chunks, normalize_embeddings=True)
    q_emb, c_embs = embs[0], embs[1:]
    sims = np.dot(c_embs, q_emb)  # cosine similarity because vectors are normalized
    idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in idx], sims[idx].tolist()


def select_diverse_topk_chunks(
    question: str,
    chunks: List[str],
    embedder: SentenceTransformer,
    top_k: int = 10,
    alpha: float = 0.7,
) -> Tuple[List[str], List[float], List[float], List[float]]:
    
    if not chunks or top_k <= 0:
        return [], [], [], []

    embs = embedder.encode([question] + chunks, normalize_embeddings=True)
    q_emb, c_embs = embs[0], embs[1:]
    cos_q = np.dot(c_embs, q_emb)  # relevance (-1..1) when normalized

    selected: List[int] = []
    tot: List[float] = []
    cos_part: List[float] = []
    dist_part: List[float] = []

    remain: set[int] = set(range(len(chunks)))
    mean_vec: Optional[np.ndarray] = None

    while remain and len(selected) < min(top_k, len(chunks)):
        if mean_vec is None:
            # first pick: best by relevance
            best = max(remain, key=lambda i: cos_q[i])
            e_dist = 0.0
            total = float(cos_q[best])
        else:
            # Euclidean distance to current memory mean (~0..2)
            cos_m = np.dot(c_embs[list(remain)], mean_vec)
            e_dist_all = np.sqrt(2.0 - 2.0 * cos_m)  # convert cosine to Euclidean
            comb = alpha * cos_q[list(remain)] + (1.0 - alpha) * e_dist_all #cos_m
            loc = int(np.argmax(comb))
            best = list(remain)[loc]
            e_dist = float(e_dist_all[loc])
            total = float(comb[loc])

        selected.append(best)
        tot.append(total)
        cos_part.append(float(cos_q[best]))
        dist_part.append(float(e_dist))
        remain.remove(best)

        # update mean memory vector (keep unit length)
        if mean_vec is None:
            mean_vec = c_embs[best].copy()
        else:
            mean_vec = np.vstack([mean_vec, c_embs[best]]).mean(axis=0)
            mean_vec /= np.linalg.norm(mean_vec)

    return [chunks[i] for i in selected], tot, cos_part, dist_part
