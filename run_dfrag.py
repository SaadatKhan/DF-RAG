import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from config import MODEL_PATH_EMBEDDING
from evaluator import LLMJudge
from planner import get_plan 
from prompts import format_prompt
from sharedmodel import global_model
from utils import select_diverse_topk_chunks, select_topk_chunks



DATASET_NAME = "2wikimqa"
TOP_K = 5
MAX_SAMPLES = 200
CHUNK_SIZE = 100
DATA_PATH = Path(
    f"/projects/klybarge/HPV_Information_Extraction/Llama/Saadat/dfr/data/longbench/{DATASET_NAME}.jsonl"
)
RESULTS_DIR = Path("results")
LOGS_DIR = Path("logs")
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILENAME = LOGS_DIR / f"execution_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILENAME), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def log_print(*args: object) -> None:
    logger.info("[DFRAG] " + " ".join(str(a) for a in args))



def chunked_by_word(text: str, chunk_size: int = 100, overlap: int = 0) -> List[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", str(text).lower())


def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _tokenize(prediction)
    gt_tokens = _tokenize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)



def generate_answer(prompt: str, max_tokens: int = 50, temperature: float = 0.01) -> str:
    return global_model.generate_response(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    ).strip()

def run() -> None:
    judge = LLMJudge()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    embedder = SentenceTransformer(MODEL_PATH_EMBEDDING)

    # aggregates
    total_df_rouge = 0.0
    total_topk_rouge = 0.0
    total_df_f1 = 0.0
    total_topk_f1 = 0.0
    df_exact = 0
    topk_exact = 0
    sum_lambda = 0.0
    total_context_words = 0

    rows: List[Dict[str, Any]] = []
    count = 0

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            q = (data.get("input") or "").strip()
            context = (data.get("context") or "").strip()
            answers = data.get("answers") or []
            gt = (answers[0] if answers else "").strip()

            if not q or not context or not gt:
                continue

            count += 1
            total_context_words += len(context.split())

            log_print("=" * 60)
            log_print(f"Row: {count}")
            log_print("Question:", q)
            log_print("Context words:", len(context.split()))
            log_print("Ground truth:", gt)

            # chunk context
            chunks = chunked_by_word(context, chunk_size=CHUNK_SIZE, overlap=0)
            if not chunks:
                continue

            # baseline top-k retrieval
            cos_chunks_topk, _ = select_topk_chunks(q, chunks, embedder, TOP_K)

            # diverse retrieval sweep (alpha ~ lambda)
            chunks_by_lambda: Dict[float, List[str]] = {}
            for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                div_chunks, *_ = select_diverse_topk_chunks(q, chunks, embedder, TOP_K, alpha=alpha)
                chunks_by_lambda[alpha] = div_chunks

            chunks_by_lambda[1.0] = cos_chunks_topk

            plan_lines = get_plan(q)
            log_print("Plan:", plan_lines)

            result = judge.find_best_lambda(plan_lines, chunks_by_lambda)
            best_lambda = float(result["best_lambda"])
            plausible = result.get("plausible_lambdas", [])
            response_text = (result.get("response_text") or "").strip()

            log_print("Plausible lambdas:", plausible)
            log_print("Best lambda:", best_lambda)

            pool: List[str] = []
            for ch in chunks_by_lambda.get(best_lambda, []):
                if ch not in pool:
                    pool.append(ch)

            reranked_chunks, _ = select_topk_chunks(q, pool, embedder, TOP_K)

            if response_text and response_text not in reranked_chunks:
                reranked_chunks.append(response_text)

            df_prompt = format_prompt(q, reranked_chunks)
            df_answer = generate_answer(df_prompt, max_tokens=50)

            topk_prompt = format_prompt(q, cos_chunks_topk)
            topk_answer = generate_answer(topk_prompt, max_tokens=50)

            log_print("DF-RAG answer:", df_answer)
            log_print("TopK answer:", topk_answer)

            # scoring
            gt_low = gt.lower().strip()
            df_low = df_answer.lower().strip()
            topk_low = topk_answer.lower().strip()

            df_rouge = scorer.score(gt_low, df_low)["rougeL"].fmeasure
            topk_rouge = scorer.score(gt_low, topk_low)["rougeL"].fmeasure

            df_f1 = calculate_f1_score(df_answer, gt)
            topk_f1 = calculate_f1_score(topk_answer, gt)

            total_df_rouge += df_rouge
            total_topk_rouge += topk_rouge
            total_df_f1 += df_f1
            total_topk_f1 += topk_f1
            sum_lambda += best_lambda

            if gt_low == df_low:
                df_exact += 1
            if gt_low == topk_low:
                topk_exact += 1

            rows.append(
                {
                    "question": q,
                    "plausible_lambdas": plausible,
                    "ground_truth": gt,
                    "df_rag_answer": df_answer,
                    "topk_answer": topk_answer,
                    "best_lambda": best_lambda,
                    "df_rougeL": df_rouge,
                    "topk_rougeL": topk_rouge,
                    "df_f1": df_f1,
                    "topk_f1": topk_f1,
                }
            )

            log_print(f"Running Avg DF-RAG RougeL: {total_df_rouge / count:.4f}")
            log_print(f"Running Avg TopK RougeL:   {total_topk_rouge / count:.4f}")
            log_print(f"Running Avg DF-RAG F1:     {total_df_f1 / count:.4f}")
            log_print(f"Running Avg TopK F1:       {total_topk_f1 / count:.4f}")

            if count >= MAX_SAMPLES:
                break

    if count == 0:
        log_print("No samples processed — check dataset formatting/path.")
        return

    # final metrics
    avg_df_rouge = total_df_rouge / count
    avg_topk_rouge = total_topk_rouge / count
    df_acc = df_exact / count
    topk_acc = topk_exact / count
    avg_lambda = sum_lambda / count
    avg_context_words = total_context_words / count

    log_print("\n--- Final Results ---")
    log_print(f"Samples:             {count}")
    log_print(f"Avg context words:   {avg_context_words:.1f}")
    log_print(f"Avg best lambda:     {avg_lambda:.3f}")
    log_print(f"DF-RAG ROUGE-L:      {avg_df_rouge:.4f}")
    log_print(f"TopK ROUGE-L:        {avg_topk_rouge:.4f}")
    log_print(f"DF-RAG EM Accuracy:  {df_acc:.4f} ({df_acc:.2%})")
    log_print(f"TopK EM Accuracy:    {topk_acc:.4f} ({topk_acc:.2%})")
    log_print("---------------------")

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / f"{DATASET_NAME}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_optimal_lambda.csv"
    df.to_csv(out_csv, index=False)
    log_print("Saved results CSV to:", out_csv)
    logger.info("Saved results CSV to %s", out_csv)


if __name__ == "__main__":
    run()
