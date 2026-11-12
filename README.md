# Dice & Dough Board Game Cafe — RAG Service

บริการถามตอบด้วย **Retrieval-Augmented Generation (RAG)** สำหรับร้านบอร์ดเกมคาเฟ่สมมติ
**“ไดซ์แอนด์โดว์” (Dice & Dough Board Game Cafe)** กรุงเทพมหานคร
งานนี้เป็นส่วนหนึ่งของรายวิชา DIT323 Artificial Intelligence as a Service

ระบบตอบคำถามภาษาไทยจากเอกสารของร้านเท่านั้น พร้อมอ้างอิงชื่อไฟล์และ chunk ต้นทาง
ถ้าฐานความรู้ไม่มีคำตอบ ระบบจะตอบว่า

> ขออภัย ไม่มีข้อมูลนี้ในฐานความรู้ของร้าน

---

## 1. ระบบทำอะไร

- ฐานความรู้ภาษาไทย 8 เอกสาร (ค่าบริการ, การจอง, เมนู, บอร์ดเกม, สมาชิก, กฎร้าน, กิจกรรม, ที่ตั้ง)
- ค้นคืนด้วย **ChromaDB** + embedding หลายภาษา **`embeddinggemma`**
- สร้างคำตอบด้วย **`qwen3.5:4b`** ผ่าน **Ollama** (`think=False`)
- **FastAPI** เปิด API และเสิร์ฟเว็บไคลเอนต์ที่ `/` ในคำสั่งเดียว
- ตอบพร้อม **แหล่งอ้างอิง**: ชื่อไฟล์, chunk, ความคล้าย (%) และข้อความ chunk
- มี **relevance threshold** (`MIN_SIMILARITY`) เพื่อปฏิเสธคำถามนอกขอบเขตก่อนเรียก LLM
- มีสคริปต์เปรียบเทียบกับโมเดลระดับคลาส **`qwen3:1.7b`** และบันทึกผลใน `eval_results.md`

## 2. สถาปัตยกรรม

```text
index.html (Bootstrap 5, Thai UI)
        │  fetch /ask, /search, /documents, /health
        ▼
main.py (FastAPI, def handlers, CORS, questions.log)
        │
        ├── POST /search ──► vector_store.search()
        │                        │ embed_query()  (embeddinggemma + query prefix)
        │                        ▼
        │                   ChromaDB (chroma_db/, cosine)
        │
        └── POST /ask ─────► vector_store.search_multi()
                                 │  full question + clauses split on "และ"
                                 │  de-duplicate, sort by similarity
                                 ▼
                            threshold check (MIN_SIMILARITY)
                                 │ pass
                                 ▼
                            rag.generate_answer()
                                 │ system prompt + context + question
                                 ▼
                            Ollama chat (qwen3.5:4b, think=False, temp 0.2)
                                 │ strip <｜begin▁of▁thinking｜> blocks, force exact refusal
                                 ▼
                            answer + sources + latency_ms

startup: knowledge/*.txt ──► split_text() ──► embeddings ──► Chroma
         + one tiny chat request (warm-up, so the first question is not a cold start)
```

### การแบ่งเอกสาร (`split_text`)

ภาษาไทยไม่มีช่องว่างระหว่างคำ จึง **ไม่ใช้ whitespace splitting**:

1. แบ่งตามย่อหน้า/หัวข้อด้วยบรรทัดว่าง (`\n\s*\n`)
2. รวมย่อหน้าเข้าด้วยกันจนใกล้ **400 ตัวอักษร**
3. ย่อหน้าที่มีเนื้อหายาวเกิน 400 ตัวอักษร ตัดเป็นหน้าต่าง ๆ ละ ~400 ตัวอักษร **overlap ~60 ตัวอักษร**
4. รวม chunk หัวข้อที่สั้นกว่า 80 ตัวอักษรเข้ากับเนื้อหาถัดไป เพื่อไม่ให้หัวข้อกิน retrieval slot เปล่า ๆ
5. เก็บ metadata `source` (ชื่อไฟล์) และ `chunk` (ลำดับ) ทุก chunk

ผลลัพธ์ปัจจุบัน: **8 เอกสาร / 50 chunks**

### การค้นคืนและการปฏิเสธ

- Query ถูก normalize คำพูดกันเองเล็กน้อย (`กี่บาท → ราคาเท่าไหร่`, `เข้าเล่น → ค่าเข้าร้าน`, `เท่าไร → เท่าไหร่`, `ยังไง → อย่างไร`, ตัด `อ่ะ/อะ/ค่ะ/ครับ` ท้ายประโยค)
- Embedding query ใช้ prompt ของ EmbeddingGemma: `task: search result | query: ...`
- คำถามที่มี `และ` จะถูกค้นทั้งประโยคเต็มและแต่ละอนุประโยค แล้ว merge ผล (multi-query retrieval) เพื่อไม่ให้คำถามสองส่วนแย่ง slot กัน
- `MIN_SIMILARITY=0.44` (calibrate แล้ว): คำถามในขอบเขตที่ต่ำที่สุดได้ **0.456** (`ค่าเค้กตัดเท่าไหร่`) ส่วนคำถามนอกขอบเขตทั่วไปได้ต่ำกว่า **0.39**
- ข้อยกเว้นที่ตั้งใจ: `เมนูพิซซ่าของร้านอื่น` ได้ similarity สูงถึง **0.4866** เพราะคำว่า “พิซซ่า/เมนู” อยู่ในเอกสารจริง กรณีนี้จึงผ่าน threshold และให้ system prompt เป็นผู้ปฏิเสธ (ทดสอบแล้วปฏิเสธถูกต้อง)
- ถ้า best similarity ต่ำกว่า threshold ระบบ **ไม่เรียก LLM** และคืน refusal พร้อม `sources: []`
- หลังผ่าน threshold จะตัด chunk ที่อ่อนกว่า `max(MIN_SIMILARITY, best × 0.65)` ออกก่อนสร้าง context เพื่อลด citation ที่ไม่เกี่ยวข้อง

## 3. โมเดลที่ใช้ (Ollama tags)

| บทบาท | Exact Ollama tag | ขนาด | หมายเหตุ |
|---|---|---|---|
| Chat (default) | **`qwen3.5:4b`** | 3.3 GB | ใหญ่กว่าโมเดลคลาส `qwen3:1.7b`; ใช้เป็นค่าเริ่มต้น |
| Embedding | **`embeddinggemma:latest`** | 621 MB, 768 dims | หลายภาษา; ทดสอบภาษาไทยแล้วดึงประโยคเรื่อง “ค่าเข้าร้าน” ถูกต้อง |
| Baseline เปรียบเทียบ | **`qwen3:1.7b`** | 1.4 GB | โมเดลระดับคลาส ใช้เฉพาะ `eval.py` |

### ทำไมเป็น `qwen3.5:4b` ไม่ใช่ 8B

โจทย์กำหนดให้ใช้ Qwen 3.5 8B ที่ติดตั้งอยู่แล้ว แต่ `ollama list` บนเครื่องนี้พบเพียง
`qwen3.5:0.8b` (เล็กกว่าโมเดลคลาส) ไม่มี 8B ติดตั้งจริง
จึงสอบถามและได้รับเลือกให้ใช้ **`qwen3.5:4b`** ซึ่งใหญ่กว่า `qwen3:1.7b`
และไม่ดึง `qwen3:8b/14b` ตามข้อกำหนด

> หมายเหตุ: เครื่องนี้มี RTX 3050 Laptop GPU **4 GB VRAM** ทำให้ `qwen3.5:4b`
> รันแบบ **52% GPU / 48% CPU** (`ollama ps`) คำตอบใช้เวลาประมาณ 5–17 วินาที
> ส่วน `embeddinggemma` และ `qwen3:1.7b` รัน **100% GPU**

### Thinking model handling

- เรียก `ollama.chat(..., think=False)`
- ถ้าเวอร์ชัน Ollama ไม่รองรับ `think` จะ retry โดยไม่ส่งพารามิเตอร์นี้
- ตัด `<think>...</think>` (และ tag ปิดที่ค้าง) ออกจาก `message.content`
- บังคับข้อความปฏิเสธให้ตรงทุกตัวอักษรถ้าโมเดลเติม citation ต่อท้าย

ทดสอบเทียบ `think=True` กับ `think=False` บน `qwen3.5:4b` ด้วยคำถามเดียวกัน (C2):

| think | latency (2 รอบ) | `message.content` | ผล |
|---|---|---|---|
| `False` | 5,580 / 10,395 ms | 148 ตัวอักษร | ได้คำตอบครบ |
| `True` | 41,465 / 48,956 ms | 0 ตัวอักษร | token ถูกใช้ไปกับ reasoning จนไม่มีคำตอบ |

จึงเลือก `think=False`: เร็วกว่า ~4.7–7.4 เท่าและได้คำตอบที่ใช้งานได้จริง

## 4. การติดตั้ง

ต้องมี Python 3.12 และ Ollama ที่ทำงานอยู่

```powershell
# ในโฟลเดอร์โปรเจกต์
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

ollama pull embeddinggemma
ollama pull qwen3.5:4b
ollama pull qwen3:1.7b   # สำหรับ eval.py เท่านั้น
```

## 5. การรัน

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

เปิด

- เว็บไคลเอนต์: <http://127.0.0.1:8000/>
- Swagger UI: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health>

ครั้งแรกจะใช้เวลาสักครู่เพื่อ index เอกสารและ warm-up โมเดล
ทุกคำถามจะถูกบันทึกใน `questions.log` (JSON Lines, UTF-8)

### Environment variables

| ตัวแปร | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `CHAT_MODEL` | `qwen3.5:4b` | โมเดลแชท |
| `EMBEDDING_MODEL` | `embeddinggemma` | โมเดล embedding |
| `MIN_SIMILARITY` | `0.44` | threshold ปฏิเสธ |
| `DEFAULT_TOP_K` | `5` | จำนวน chunk เริ่มต้น (1–10) |
| `KNOWLEDGE_DIR` | `./knowledge` | โฟลเดอร์เอกสาร |
| `CHROMA_DIR` | `./chroma_db` | ที่เก็บ ChromaDB |
| `COLLECTION_NAME` | `dicedough_knowledge` | ชื่อ collection |
| `OLLAMA_HOST` | ค่าเริ่มต้นของ Ollama | เช่น `http://127.0.0.1:11434` |
| `EMBEDDING_QUERY_PREFIX` | `task: search result \| query: ` | query prompt |

## 6. API

| Method | Path | คำอธิบาย |
|---|---|---|
| GET | `/` | เว็บไคลเอนต์ภาษาไทย |
| GET | `/health` | สถานะ + โมเดล + จำนวนเอกสาร/chunks |
| GET | `/models` | โมเดลและค่าที่ตั้งไว้ |
| POST | `/search` | ค้นคืน chunk (ไม่เรียก LLM) |
| POST | `/ask` | RAG Q&A + sources + latency |
| POST | `/documents` | เพิ่ม/อัปเดตเอกสาร `.txt` (delete chunks เก่าก่อน re-index) |

### ตัวอย่าง curl (ผลจริงจากเครื่องนี้)

> บน Windows PowerShell ให้ใช้ `curl.exe` เพื่อเลี่ยง alias ของ `Invoke-WebRequest`

**GET /health**

```bash
curl.exe http://127.0.0.1:8000/health
```

```json
{"status":"ok","chat_model":"qwen3.5:4b","embedding_model":"embeddinggemma","documents":8,"chunks":50,"min_similarity":0.44,"warmup_ok":true}
```

**POST /search**

```bash
curl.exe -X POST http://127.0.0.1:8000/search ^
  -H "Content-Type: application/json; charset=utf-8" ^
  -d "{\"question\":\"ค่าเข้าร้านวันธรรมดาเท่าไหร่\",\"top_k\":5}"
```

ผลจริง (ตัดมา 2 รายการแรก):

```json
{
  "question": "ค่าเข้าร้านวันธรรมดาเท่าไหร่",
  "results": [
    {"source":"pricing.txt","chunk":0,"similarity":0.5634,
     "text":"ค่าบริการและโปรโมชันของร้านไดซ์แอนด์โดว์\n\nอัตราค่าเข้าร้าน\n..."},
    {"source":"events.txt","chunk":4,"similarity":0.5554,
     "text":"การเหมาร้านและความจุ\nร้านมีที่นั่งทั้งหมด 70 ที่ ..."}
  ],
  "embedding_model": "embeddinggemma"
}
```

**POST /ask (in-scope)**

```bash
curl.exe -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json; charset=utf-8" ^
  -d "{\"question\":\"ค่าเข้าร้านวันธรรมดาเท่าไหร่\",\"top_k\":5}"
```

ผลจริง:

```json
{
  "question": "ค่าเข้าร้านวันธรรมดาเท่าไหร่",
  "answer": "ค่าเข้าร้านวันธรรมดาคิดแบบรายชั่วโมงคือ 60 บาทต่อคน (วันธรรมดาชั่วโมงละ 60 บาท) หรือซื้อบัตร All-Day Pass ราคา 349 บาทต่อคนสำหรับวันธรรมดา (pricing.txt)",
  "sources": [
    {"source":"pricing.txt","chunk":0,"similarity":0.5634,"text":"..."},
    {"source":"pricing.txt","chunk":1,"similarity":0.5362,"text":"..."}
  ],
  "model": "qwen3.5:4b",
  "latency_ms": 16597,
  "refused": false
}
```

**POST /ask (out-of-scope)**

```bash
curl.exe -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json; charset=utf-8" ^
  -d "{\"question\":\"ราคาหุ้นวันนี้เท่าไหร่\"}"
```

```json
{
  "question": "ราคาหุ้นวันนี้เท่าไหร่",
  "answer": "ขออภัย ไม่มีข้อมูลนี้ในฐานความรู้ของร้าน",
  "sources": [],
  "model": "qwen3.5:4b",
  "latency_ms": 112,
  "refused": true
}
```

**POST /documents**

```bash
curl.exe -X POST http://127.0.0.1:8000/documents ^
  -H "Content-Type: application/json; charset=utf-8" ^
  -d "{\"name\":\"promotions.txt\",\"text\":\"โปรโมชันเดือนตุลาคม ลูกค้าที่มาเล่นวันจันทร์ถึงพฤหัสบดีหลัง 20:00 น. รับส่วนลดค่าเข้า 10 เปอร์เซ็นต์\"}"
```

```json
{
  "name": "promotions.txt",
  "chunks": 1,
  "documents": 9,
  "total_chunks": 51,
  "message": "เพิ่มเอกสาร promotions.txt สำเร็จ 1 chunks"
}
```

## 7. Evaluation

รัน:

```powershell
.\.venv\Scripts\python.exe eval.py
```

- คำถาม 12 ข้อ: in-scope 8 ข้อ (3 ข้อต้องใช้สองเอกสาร), paraphrase 2 ข้อ, out-of-scope 2 ข้อ
- โมเดล: `qwen3.5:4b` และ `qwen3:1.7b`
- แต่ละข้อรัน 3 ครั้งหลัง warm-up; วัด median latency และตัดสินความถูกต้องแบบ majority (2/3)
- ตรวจ key fact ด้วย substring (ไม่ใช้ LLM judge)
- ผลละเอียด + ตาราง + failure list อยู่ใน **`eval_results.md`**

ตารางสรุป (ดูตัวเลขล่าสุดใน `eval_results.md`):

<!-- EVAL_TABLE_START -->
| Model | Answer accuracy (10) | Refusal correctness (2) | Median latency, answerable (ms) |
|---|---|---|---|
| **`qwen3.5:4b`** (default) | **10/10 (100%)** | **2/2** | 6,613 |
| `qwen3:1.7b` (baseline) | 10/10 (100%) | 2/2 | 1,614 |

ผลจริงจากการรันล่าสุด (`eval_results.md`, 2026-10-25 22:57) ทั้งสองโมเดลตอบถูกทุกข้อในชุดทดสอบนี้
ความแตกต่างหลักคือ **latency**: `qwen3.5:4b` ช้ากว่า ~4 เท่าเพราะรันข้าม CPU/GPU (VRAM 4 GB)
แต่ให้คำตอบละเอียดและอ้างอิงหลายเอกสารได้ดีกว่าในการใช้งานจริง คำถามชุดนี้มีเพียง 10 ข้อ
จึงยังไม่ใช่หลักฐานว่าคุณภาพเท่ากันทุกกรณี
<!-- EVAL_TABLE_END -->

## 8. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

ผลจริง: **9 passed** ครอบคลุม `/health`, `/search` (ไฟล์ถูกต้องสำหรับคำถามไทย),
`/ask` (answer + sources), out-of-scope refusal, validation 422, path traversal `../x.txt` 422,
`/models`, และการเสิร์ฟ `index.html` ที่ `/`

## 9. โครงสร้างโปรเจกต์

```text
Rag Service/
├── knowledge/             # 8 เอกสารไทย (pricing, reservation, menu, games,
│                          # membership, rules, events, location)
├── chroma_db/             # ChromaDB persistent store (สร้างอัตโนมัติ)
├── embeddings.py          # Ollama embeddings + Thai query normalization
├── vector_store.py        # chunking (Thai-safe), index, search, search_multi
├── rag.py                 # system prompt, context, think=False + fallback, refusal
├── main.py                # FastAPI app, endpoints, threshold, logging, warm-up
├── index.html             # Bootstrap 5 Thai chat UI
├── tests/test_api.py      # pytest + TestClient
├── eval.py                # 2-model evaluation -> eval_results.md
├── eval_results.md        # ผล evaluation ล่าสุด
├── requirements.txt
├── questions.log          # JSONL ของคำถามที่เรียก /ask (สร้างเมื่อใช้งาน)
├── server.out.log / server.err.log  # log ตอนรัน background (ถ้ามี)
└── README.md
```

## 10. ข้อจำกัดและสิ่งที่ควรรู้

- **ไม่ได้ใช้ 8B จริง** เพราะเครื่องไม่มีโมเดล 8B ติดตั้ง; ใช้ `qwen3.5:4b` ตามที่ผู้ใช้เลือก
  (ยังใหญ่กว่า `qwen3:1.7b` ของคลาส)
- **VRAM 4 GB**: `qwen3.5:4b` รัน 52% GPU / 48% CPU ทำให้ตอบช้ากว่าโมเดลเล็ก
  ถ้าปิดแอปที่ใช้ GPU อื่นหรือใช้การ์ดที่ VRAM มากกว่า อาจได้ GPU มากขึ้น
- คำถาม near-domain out-of-scope (เช่น พิซซ่าของร้านอื่น) ผ่าน similarity threshold
  การปฏิเสธจึงขึ้นกับ system prompt ไม่ใช่ threshold เพียงอย่างเดียว
- คำตอบจากโมเดล 4B อาจมีข้อมูลเกินคำถามหรือ citation ผิดไฟล์เล็กน้อย;
  evaluation ใช้ substring จึงอาจตัดสิน “ถูก” แม้ถ้อยคำไม่สวย
- ภาษาไทยไม่มีช่องว่าง คำที่สะกดต่างกันเล็กน้อยอาจทำให้ similarity ลดลง
- ไม่มีระบบ authentication; `/documents` เขียนไฟล์ลง `knowledge/` ได้
  จึงควรใช้ในเครื่อง/local demo เท่านั้น
- `questions.log` เก็บคำถามและ metadata แต่ไม่เก็บคำตอบเต็ม (ดู answer ได้จาก response)
- ต้องมี Ollama รันอยู่; ถ้า Ollama ล่ม `/ask` จะตอบ 503 พร้อมคำอธิบาย
