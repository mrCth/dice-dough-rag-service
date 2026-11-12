"""FastAPI application for the Dice & Dough Board Game Cafe RAG service."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

import rag
import vector_store
from embeddings import EMBEDDING_MODEL

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "questions.log"

CHAT_MODEL = rag.CHAT_MODEL
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.44"))  # calibrated, see README
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
APP_VERSION = "1.0.0"

_log_lock = threading.Lock()


class Source(BaseModel):
    source: str = Field(..., description="ชื่อไฟล์ต้นทาง", examples=["pricing.txt"])
    chunk: int = Field(..., description="ลำดับ chunk ในไฟล์", examples=[0])
    text: str = Field(..., description="ข้อความ chunk ที่ถูกดึงมา")
    similarity: float = Field(..., description="ความคล้ายแบบ cosine (0-1)", examples=[0.62])


class SearchRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="คำถามภาษาไทย",
        examples=["ค่าเข้าร้านวันธรรมดาเท่าไหร่"],
    )
    top_k: int = Field(
        DEFAULT_TOP_K,
        ge=1,
        le=10,
        description="จำนวน chunk ที่ต้องการดึง",
        examples=[4],
    )


class SearchResponse(BaseModel):
    question: str
    results: list[Source]
    embedding_model: str = EMBEDDING_MODEL


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="คำถามภาษาไทยเกี่ยวกับร้าน",
        examples=["นักศึกษาได้ส่วนลดเท่าไหร่ และใช้ร่วมกับ Happy Hour ได้ไหม"],
    )
    top_k: int = Field(
        DEFAULT_TOP_K,
        ge=1,
        le=10,
        description="จำนวน chunk ที่ใช้เป็น context",
        examples=[4],
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    model: str
    latency_ms: int
    refused: bool


class DocumentRequest(BaseModel):
    name: str = Field(
        ...,
        min_length=5,
        max_length=70,
        description="ชื่อไฟล์ภาษาอังกฤษ ลงท้ายด้วย .txt เช่น promotions.txt",
        examples=["promotions.txt"],
    )
    text: str = Field(
        ...,
        min_length=20,
        max_length=20000,
        description="เนื้อหาเอกสารภาษาไทย",
        examples=[
            "โปรโมชันเดือนตุลาคม: ลูกค้าที่มาเล่นวันจันทร์ถึงพฤหัสบดี "
            "หลัง 20:00 น. รับส่วนลดค่าเข้า 10 เปอร์เซ็นต์"
        ],
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if Path(value).name != value or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}\.txt", value):
            raise ValueError(
                "ชื่อไฟล์ต้องเป็นภาษาอังกฤษ ตัวเลข ขีดกลาง หรือขีดล่าง "
                "และลงท้ายด้วย .txt เท่านั้น"
            )
        return value


class DocumentResponse(BaseModel):
    name: str
    chunks: int
    documents: int
    total_chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    chat_model: str
    embedding_model: str
    documents: int
    chunks: int
    min_similarity: float
    warmup_ok: bool


class ModelsResponse(BaseModel):
    chat_model: str
    embedding_model: str
    default_top_k: int
    min_similarity: float
    refusal_message: str


def _log_question(payload: dict[str, Any]) -> None:
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **payload}
    with _log_lock:
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.warmup = {"ok": False, "error": "not started"}
    try:
        indexed = vector_store.index_directory()
        app.state.indexed = indexed
        print(f"[startup] indexed {sum(indexed.values())} chunks from {len(indexed)} files")
    except Exception as exc:  # pragma: no cover - depends on local environment
        app.state.indexed = {}
        print(f"[startup] indexing failed: {exc}")

    warm = rag.warmup()
    app.state.warmup = warm
    if warm.get("ok"):
        print(f"[startup] chat warm-up ok in {warm['latency_ms']} ms ({warm['model']})")
    else:
        print(f"[startup] chat warm-up failed: {warm.get('error')}")
    yield


app = FastAPI(
    title="Dice & Dough Board Game Cafe — RAG Service",
    description=(
        "บริการตอบคำถามจากฐานความรู้ของร้านบอร์ดเกมคาเฟ่ **ไดซ์แอนด์โดว์** "
        "(Dice & Dough Board Game Cafe) กรุงเทพมหานคร\n\n"
        "ระบบใช้ Retrieval-Augmented Generation (RAG) บน Ollama + ChromaDB "
        "โดยค้นคืนเฉพาะเอกสารของร้านและตอบพร้อมอ้างอิงชื่อไฟล์ต้นทาง "
        "หากไม่มีข้อมูลในฐานความรู้ ระบบจะตอบว่า "
        "\"ขออภัย ไม่มีข้อมูลนี้ในฐานความรู้ของร้าน\"\n\n"
        "**ตัวอย่างคำถาม**: ค่าเข้าร้านเท่าไหร่ / นักศึกษาได้ส่วนลดไหม / "
        "จองโต๊ะกลุ่ม 8 คนต้องทำอย่างไร / เกมไหนเหมาะกับผู้เริ่มต้น"
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "System", "description": "สถานะและโมเดลที่ใช้งาน"},
        {"name": "Retrieval", "description": "ค้นคืนเอกสารจาก ChromaDB"},
        {"name": "RAG", "description": "ถามตอบด้วย RAG พร้อมแหล่งอ้างอิง"},
        {"name": "Documents", "description": "เพิ่มหรืออัปเดตเอกสารในฐานความรู้"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html", media_type="text/html")


@app.get("/health", response_model=HealthResponse, tags=["System"], summary="ตรวจสอบสถานะระบบ")
def health() -> HealthResponse:
    counts = vector_store.stats()
    warm = getattr(app.state, "warmup", {})
    return HealthResponse(
        status="ok",
        chat_model=CHAT_MODEL,
        embedding_model=EMBEDDING_MODEL,
        documents=counts["documents"],
        chunks=counts["chunks"],
        min_similarity=MIN_SIMILARITY,
        warmup_ok=bool(warm.get("ok")),
    )


@app.get("/models", response_model=ModelsResponse, tags=["System"], summary="โมเดลที่ตั้งค่าไว้")
def models() -> ModelsResponse:
    return ModelsResponse(
        chat_model=CHAT_MODEL,
        embedding_model=EMBEDDING_MODEL,
        default_top_k=DEFAULT_TOP_K,
        min_similarity=MIN_SIMILARITY,
        refusal_message=rag.REFUSAL,
    )


@app.post("/search", response_model=SearchResponse, tags=["Retrieval"], summary="ค้นคืนเอกสาร")
def search(request: SearchRequest) -> SearchResponse:
    try:
        hits = vector_store.search(request.question, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ค้นคืนเอกสารไม่สำเร็จ: {exc}") from exc
    return SearchResponse(question=request.question, results=hits)


@app.post("/ask", response_model=AskResponse, tags=["RAG"], summary="ถามตอบจากฐานความรู้")
def ask(request: AskRequest) -> AskResponse:
    started = time.perf_counter()
    try:
        hits = vector_store.search_multi(request.question, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ค้นคืนเอกสารไม่สำเร็จ: {exc}") from exc

    best_score = hits[0]["similarity"] if hits else 0.0
    if not hits or best_score < MIN_SIMILARITY:
        latency_ms = round((time.perf_counter() - started) * 1000)
        _log_question(
            {
                "endpoint": "ask",
                "question": request.question,
                "top_k": request.top_k,
                "best_score": best_score,
                "refused": True,
                "model": CHAT_MODEL,
                "latency_ms": latency_ms,
            }
        )
        return AskResponse(
            question=request.question,
            answer=rag.REFUSAL,
            sources=[],
            model=CHAT_MODEL,
            latency_ms=latency_ms,
            refused=True,
        )

    # Drop weak chunks so they do not become hallucinated citations.
    cutoff = max(MIN_SIMILARITY, best_score * 0.65)
    used_hits = [hit for hit in hits if hit["similarity"] >= cutoff] or hits[:1]

    try:
        result = rag.generate_answer(request.question, used_hits, model=CHAT_MODEL)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"ไม่สามารถเรียกโมเดล {CHAT_MODEL} ได้ กรุณาตรวจสอบ Ollama: {exc}",
        ) from exc

    latency_ms = round((time.perf_counter() - started) * 1000)
    refused = bool(result["refused"])
    _log_question(
        {
            "endpoint": "ask",
            "question": request.question,
            "top_k": request.top_k,
            "best_score": best_score,
            "refused": refused,
            "model": CHAT_MODEL,
            "latency_ms": latency_ms,
        }
    )
    return AskResponse(
        question=request.question,
        answer=result["answer"],
        sources=[] if refused else used_hits,
        model=result["model"],
        latency_ms=latency_ms,
        refused=refused,
    )


@app.post("/documents", response_model=DocumentResponse, tags=["Documents"], summary="เพิ่มเอกสาร")
def add_document(request: DocumentRequest) -> DocumentResponse:
    path = (vector_store.KNOWLEDGE_DIR / request.name).resolve()
    knowledge_root = vector_store.KNOWLEDGE_DIR.resolve()
    if path.parent != knowledge_root:
        raise HTTPException(status_code=422, detail="ชื่อไฟล์ไม่ปลอดภัย")

    try:
        path.write_text(request.text, encoding="utf-8")
        chunks = vector_store.index_text(request.text, source=request.name)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"เพิ่มเอกสารไม่สำเร็จ: {exc}") from exc

    counts = vector_store.stats()
    return DocumentResponse(
        name=request.name,
        chunks=chunks,
        documents=counts["documents"],
        total_chunks=counts["chunks"],
        message=f"เพิ่มเอกสาร {request.name} สำเร็จ {chunks} chunks",
    )
