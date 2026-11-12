"""RAG prompt construction and Ollama chat calls for Dice & Dough."""

from __future__ import annotations

import os
import re
import time
from typing import Any

import ollama

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:4b")
REFUSAL = "ขออภัย ไม่มีข้อมูลนี้ในฐานความรู้ของร้าน"

SYSTEM_PROMPT = """คุณคือผู้ช่วยของร้านบอร์ดเกมคาเฟ่ "ไดซ์แอนด์โดว์" (Dice & Dough Board Game Cafe) ในกรุงเทพมหานคร

กฎการตอบ:
1. ตอบเป็นภาษาไทยเท่านั้น
2. ใช้ข้อมูลจาก CONTEXT ที่ให้มาเท่านั้น ห้ามใช้ความรู้ภายนอก ห้ามเดา และห้ามแต่งตัวเลข ราคา เวลา หรือเงื่อนไข
3. ตอบตรงคำถาม กระชับ ชัดเจน ถ้าถามราคาให้ระบุชื่อรายการและราคาให้ครบ ถ้าถามค่าเข้าและมีทั้งอัตรารายชั่วโมงกับบัตร All-Day Pass ให้ตอบทั้งสองแบบโดยแยกให้ชัด
4. อ้างชื่อไฟล์ต้นทางในวงเล็บท้ายคำตอบ เช่น (pricing.txt) และอ้างเฉพาะไฟล์ที่นำข้อมูลมาตอบจริง ห้ามอ้างไฟล์ที่ไม่ได้ใช้
5. ถ้า CONTEXT ไม่มีข้อมูลที่ตอบคำถามได้ ให้ตอบข้อความนี้เท่านั้น: ขออภัย ไม่มีข้อมูลนี้ในฐานความรู้ของร้าน แต่ถ้า CONTEXT มีข้อมูลที่ตอบได้ ห้ามใช้ข้อความปฏิเสธนี้
6. ถ้าคำถามต้องใช้ข้อมูลจากหลายไฟล์ ให้รวบรวมและอ้างอิงทุกไฟล์ที่ใช้
7. ระวังความสับสนระหว่างส่วนลดนักเรียน (pricing.txt) กับส่วนลดสมาชิก (membership.txt) ซึ่งมีเงื่อนไขการใช้ร่วมกับ Happy Hour ต่างกัน ให้ตอบเฉพาะส่วนลดที่คำถามถามถึง และห้ามนำเงื่อนไขของส่วนลดสมาชิกไปใช้กับส่วนลดนักเรียน
8. ตอบเป็นข้อความธรรมดา ไม่ต้องใช้เครื่องหมาย markdown เช่น ** หรือ #
9. ถ้าคำถามถามถึงร้านอื่น ผลิตภัณฑ์ของร้านอื่น หรือข้อมูลที่ไม่เกี่ยวกับร้านไดซ์แอนด์โดว์โดยตรง ให้ตอบข้อความปฏิเสธตามข้อ 5 แม้ CONTEXT จะมีคำคล้ายกัน ห้ามตอบในนามร้านอื่นหรือเปรียบเทียบกับร้านอื่น
"""

_client: ollama.Client | None = None
_think_supported: bool | None = None


def get_client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=os.getenv("OLLAMA_HOST") or None)
    return _client


def strip_think(text: str) -> str:
    """Remove Qwen-style reasoning blocks from model output."""
    if not text:
        return ""
    text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)
    if "<｜end▁of▁thinking｜>" in text:
        text = text.rsplit("<｜end▁of▁thinking｜>", 1)[-1]
    if "<think" in text.lower():
        text = re.split(r"<think\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
    return text.strip()


def build_context(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] ไฟล์: {chunk['source']} | chunk {chunk['chunk']} | "
            f"ความคล้าย: {chunk['similarity']:.2f}\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def _call_chat(messages: list[dict[str, str]], model: str, think: bool | None, num_predict: int):
    global _think_supported
    client = get_client()
    options = {"temperature": 0.2, "num_predict": num_predict, "seed": 42}
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "options": options}

    if think is not None and _think_supported is not False:
        kwargs["think"] = think

    try:
        response = client.chat(**kwargs)
        if think is not None:
            _think_supported = True
        return response
    except Exception:
        if "think" not in kwargs:
            raise
        # Older Ollama versions reject the `think` option; retry without it.
        _think_supported = False
        kwargs.pop("think", None)
        return client.chat(**kwargs)


def generate_answer(
    question: str,
    chunks: list[dict[str, Any]],
    model: str | None = None,
) -> dict[str, Any]:
    """Generate a grounded Thai answer from retrieved chunks."""
    model = model or CHAT_MODEL
    if not chunks:
        return {"answer": REFUSAL, "latency_ms": 0, "model": model, "refused": True}

    user_prompt = (
        f"CONTEXT:\n{build_context(chunks)}\n\n"
        f"คำถาม: {question}\n\n"
        "ตอบคำถามโดยใช้เฉพาะ CONTEXT ข้างต้น และอ้างชื่อไฟล์ต้นทาง"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    start = time.perf_counter()
    response = _call_chat(messages, model=model, think=False, num_predict=500)
    latency_ms = round((time.perf_counter() - start) * 1000)
    raw = (response.get("message") or {}).get("content", "")
    answer = strip_think(raw)
    # The model sometimes appends a citation to the refusal sentence. Force the
    # exact refusal string required by the service contract.
    if not answer or REFUSAL in answer:
        answer = REFUSAL
        refused = True
    else:
        refused = False
    return {
        "answer": answer,
        "latency_ms": latency_ms,
        "model": model,
        "refused": refused,
    }


def warmup(model: str | None = None) -> dict[str, Any]:
    """Send a tiny request so the first real question is not a cold start."""
    model = model or CHAT_MODEL
    messages = [{"role": "user", "content": "ตอบคำเดียวว่า: พร้อม"}]
    start = time.perf_counter()
    try:
        response = _call_chat(messages, model=model, think=False, num_predict=16)
        answer = strip_think((response.get("message") or {}).get("content", ""))
        return {
            "ok": True,
            "model": model,
            "latency_ms": round((time.perf_counter() - start) * 1000),
            "answer": answer,
        }
    except Exception as exc:  # pragma: no cover - depends on local Ollama
        return {
            "ok": False,
            "model": model,
            "latency_ms": round((time.perf_counter() - start) * 1000),
            "error": str(exc),
        }
