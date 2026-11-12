"""Evaluate the Dice & Dough RAG service with two Ollama chat models.

Runs 12 questions (8 in-scope including 3 cross-document, 2 paraphrased,
2 out-of-scope) three times per model after warm-up. Saves eval_results.md.
"""

from __future__ import annotations

import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import rag  # noqa: E402
import vector_store  # noqa: E402
from rag import REFUSAL  # noqa: E402

MODELS = [m.strip() for m in os.getenv("EVAL_MODELS", "qwen3.5:4b,qwen3:1.7b").split(",") if m.strip()]
TOP_K = int(os.getenv("EVAL_TOP_K", "5"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.44"))
RUNS = 3
RESULTS_FILE = PROJECT_ROOT / "eval_results.md"

# Each fact is a list of acceptable substrings; all fact groups must match.
QUESTIONS = [
    {
        "id": "I1",
        "type": "in-scope",
        "question": "ค่าเข้าร้านวันธรรมดาคนละกี่บาท",
        "facts": [["60"]],
    },
    {
        "id": "I2",
        "type": "in-scope",
        "question": "สมาชิก Gold ได้ส่วนลดค่าเข้าเท่าไหร่ และได้บัตรผ่านฟรีเดือนละกี่ใบ",
        "facts": [["15"], ["2"]],
    },
    {
        "id": "I3",
        "type": "in-scope",
        "question": "มีเกมอะไรบ้างที่เหมาะกับมือใหม่ 4 คน",
        "facts": [["codenames", "carcassonne", "azul", "dixit", "splendor"]],
    },
    {
        "id": "I4",
        "type": "in-scope",
        "question": "ที่จอดรถฟรีกี่ชั่วโมง และต้องใช้จ่ายครบเท่าไหร่",
        "facts": [["2"], ["300"]],
    },
    {
        "id": "I5",
        "type": "in-scope",
        "question": "จองโต๊ะกลุ่ม 8 คนต้องทำอย่างไร และมัดจำเท่าไหร่",
        "facts": [["400"]],
    },
    {
        "id": "C1",
        "type": "cross-document",
        "question": "จัดปาร์ตี้วันเกิด 8 คน ต้องจองห้องไหน ล่วงหน้ากี่วัน และห้องจุได้สูงสุดกี่คน",
        "facts": [["dragon"], ["7"], ["10"]],
    },
    {
        "id": "C2",
        "type": "cross-document",
        "question": "นักเรียนที่เป็นสมาชิก Silver ควรใช้ส่วนลดไหน และใช้ร่วมกับ Happy Hour ได้ไหม",
        "facts": [["20"], ["happy hour"]],
    },
    {
        "id": "C3",
        "type": "cross-document",
        "question": "นำเค้กวันเกิดจากภายนอกเข้ามาได้ไหม ต้องจ่ายเท่าไหร่",
        "facts": [["ได้"], ["100"]],
    },
    {
        "id": "P1",
        "type": "paraphrase",
        "question": "เข้าเล่นกี่บาทอ่ะ",
        "facts": [["60"]],
    },
    {
        "id": "P2",
        "type": "paraphrase",
        "question": "ร้านหยุดวันไหนบ้างอ่ะ",
        "facts": [["ทุกวัน", "ไม่หยุด"]],
    },
    {
        "id": "O1",
        "type": "out-of-scope",
        "question": "ราคาหุ้นวันนี้เท่าไหร่",
        "facts": [],
    },
    {
        "id": "O2",
        "type": "out-of-scope",
        "question": "เมนูพิซซ่าของร้านอื่นเป็นยังไง",
        "facts": [],
    },
]

OUT_OF_SCOPE_IDS = {item["id"] for item in QUESTIONS if item["type"] == "out-of-scope"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace(",", "")).strip()


def facts_match(answer: str, fact_groups: list[list[str]]) -> bool:
    normalized = normalize(answer)
    return all(any(normalize(option) in normalized for option in group) for group in fact_groups)


def answer_once(question: str, model: str) -> dict:
    started = time.perf_counter()
    hits = vector_store.search_multi(question, top_k=TOP_K)
    best_score = hits[0]["similarity"] if hits else 0.0
    if not hits or best_score < MIN_SIMILARITY:
        return {
            "answer": REFUSAL,
            "sources": [],
            "best_score": best_score,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "threshold_refusal": True,
        }

    cutoff = max(MIN_SIMILARITY, best_score * 0.65)
    used = [hit for hit in hits if hit["similarity"] >= cutoff] or hits[:1]
    result = rag.generate_answer(question, used, model=model)
    return {
        "answer": result["answer"],
        "sources": used,
        "best_score": best_score,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "threshold_refusal": False,
    }


def median(values: list[float]) -> float:
    return round(statistics.median(values)) if values else 0


def run_model(model: str) -> dict:
    print(f"\n===== MODEL: {model} =====", flush=True)
    warm = rag.warmup(model=model)
    print(f"warm-up: ok={warm.get('ok')} latency={warm.get('latency_ms')} ms", flush=True)

    records = []
    for item in QUESTIONS:
        runs = []
        for run_index in range(RUNS):
            outcome = answer_once(item["question"], model)
            if item["type"] == "out-of-scope":
                correct = REFUSAL in outcome["answer"]
            else:
                correct = facts_match(outcome["answer"], item["facts"])
            runs.append({**outcome, "correct": correct})
            print(
                f"  {item['id']} run {run_index + 1}: correct={correct} "
                f"{outcome['latency_ms']} ms",
                flush=True,
            )

        correct_runs = sum(1 for run in runs if run["correct"])
        record = {
            **item,
            "runs": runs,
            "correct_runs": correct_runs,
            "majority_correct": correct_runs >= 2,
            "median_latency": median([run["latency_ms"] for run in runs]),
            "first_answer": runs[0]["answer"],
            "best_score": runs[0]["best_score"],
            "threshold_refusals": sum(1 for run in runs if run["threshold_refusal"]),
        }
        records.append(record)

    answerable = [r for r in records if r["id"] not in OUT_OF_SCOPE_IDS]
    out_scope = [r for r in records if r["id"] in OUT_OF_SCOPE_IDS]
    return {
        "model": model,
        "records": records,
        "warmup": warm,
        "accuracy": sum(1 for r in answerable if r["majority_correct"]),
        "answerable_total": len(answerable),
        "refusal_correct": sum(1 for r in out_scope if r["majority_correct"]),
        "refusal_total": len(out_scope),
        "median_latency": median([r["median_latency"] for r in answerable]),
    }


def snippet(text: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def build_markdown(results: list[dict]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append("# Dice & Dough RAG — Evaluation Results")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Embedding model: `embeddinggemma` (query prefix: `task: search result | query: `)")
    lines.append(f"- Retrieval: ChromaDB cosine, multi-query split on `และ`, top_k={TOP_K}, MIN_SIMILARITY={MIN_SIMILARITY}")
    lines.append(f"- Chat models: {', '.join('`' + r['model'] + '`' for r in results)}")
    lines.append(f"- Runs: {RUNS} per question after one warm-up request; answer correctness by majority (2/3)")
    lines.append("- Answerable questions: 8 in-scope (3 cross-document) + 2 paraphrased = 10")
    lines.append("- Out-of-scope questions: 2 (stock price, another shop's pizza menu)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | Answer accuracy (10) | Refusal correctness (2) | Median latency, answerable (ms) |")
    lines.append("|---|---|---|---|")
    for result in results:
        acc = result["accuracy"]
        total = result["answerable_total"]
        ref = result["refusal_correct"]
        ref_total = result["refusal_total"]
        lines.append(
            f"| `{result['model']}` | {acc}/{total} ({acc / total * 100:.0f}%) | "
            f"{ref}/{ref_total} | {result['median_latency']:,} |"
        )
    lines.append("")
    lines.append("Median latency includes retrieval + generation for answerable questions only; "
                 "out-of-scope threshold refusals are nearly instant and are excluded to keep the "
                 "comparison fair.")
    lines.append("")

    lines.append("## Failures (honest)")
    lines.append("")
    any_failure = False
    for result in results:
        failures = [r for r in result["records"] if not r["majority_correct"]]
        lines.append(f"### `{result['model']}`")
        lines.append("")
        if not failures:
            lines.append("No majority failures.")
        else:
            any_failure = True
            for record in failures:
                if record["type"] == "out-of-scope":
                    expected = "must refuse"
                else:
                    expected = " / ".join(" | ".join(group) for group in record["facts"])
                lines.append(
                    f"- **{record['id']}** ({record['type']}) — expected: `{expected}`; "
                    f"correct runs: {record['correct_runs']}/{RUNS}"
                )
                lines.append(f"  - Q: {record['question']}")
                lines.append(f"  - A: {snippet(record['first_answer'])}")
        lines.append("")
    if not any_failure:
        lines.append("")
        lines.append("All questions passed by majority.")
        lines.append("")

    lines.append("## Per-question detail")
    lines.append("")
    for result in results:
        lines.append(f"### `{result['model']}`")
        lines.append("")
        lines.append("| ID | Type | Best sim | Correct runs | Median ms | Answer snippet |")
        lines.append("|---|---|---|---|---|---|")
        for record in result["records"]:
            lines.append(
                f"| {record['id']} | {record['type']} | {record['best_score']:.3f} | "
                f"{record['correct_runs']}/{RUNS} | {record['median_latency']:,} | "
                f"{snippet(record['first_answer'], 180).replace('|', '\\|')} |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- The out-of-scope pizza question is deliberately near-domain: it passes the similarity "
                 "threshold, so refusal depends on the system prompt. The stock-price question is rejected "
                 "by the threshold before the LLM is called.")
    lines.append("- Thai has no word spaces; chunking is paragraph/section based with ~400-character windows "
                 "and 60-character overlap, so retrieval behavior can differ from whitespace-based systems.")
    lines.append("- On this laptop (RTX 3050 Laptop, 4 GB VRAM) `qwen3.5:4b` runs partly on CPU, which limits "
                 "latency. See README for the `ollama ps` reading.")
    lines.append("- Correctness is a substring check of the expected key facts, not a human/LLM judge; "
                 "a fluent but factually wrong answer fails.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    print("Indexing knowledge base...", flush=True)
    counts = vector_store.index_directory()
    print(f"Indexed {sum(counts.values())} chunks from {len(counts)} documents.", flush=True)

    results = [run_model(model) for model in MODELS]
    markdown = build_markdown(results)
    RESULTS_FILE.write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
