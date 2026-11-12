# Dice & Dough RAG — Evaluation Results

Generated: 2026-10-25 22:57

## Setup

- Embedding model: `embeddinggemma` (query prefix: `task: search result | query: `)
- Retrieval: ChromaDB cosine, multi-query split on `และ`, top_k=5, MIN_SIMILARITY=0.44
- Chat models: `qwen3.5:4b`, `qwen3:1.7b`
- Runs: 3 per question after one warm-up request; answer correctness by majority (2/3)
- Answerable questions: 8 in-scope (3 cross-document) + 2 paraphrased = 10
- Out-of-scope questions: 2 (stock price, another shop's pizza menu)

## Summary

| Model | Answer accuracy (10) | Refusal correctness (2) | Median latency, answerable (ms) |
|---|---|---|---|
| `qwen3.5:4b` | 10/10 (100%) | 2/2 | 6,613 |
| `qwen3:1.7b` | 10/10 (100%) | 2/2 | 1,614 |

Median latency includes retrieval + generation for answerable questions only; out-of-scope threshold refusals are nearly instant and are excluded to keep the comparison fair.

## Failures (honest)

### `qwen3.5:4b`

No majority failures.

### `qwen3:1.7b`

No majority failures.


All questions passed by majority.

## Per-question detail

### `qwen3.5:4b`

| ID | Type | Best sim | Correct runs | Median ms | Answer snippet |
|---|---|---|---|---|---|
| I1 | in-scope | 0.590 | 3/3 | 5,186 | ค่าเข้าร้านวันธรรมดาคิดเป็นรายชั่วโมงต่อคน ชั่วโมงละ 60 บาท (pricing.txt) และมีบัตร All-Day Pass ราคา 349 บาทสำหรับวันธรรมดา (pricing.txt) |
| I2 | in-scope | 0.649 | 3/3 | 4,236 | สมาชิก Gold ได้รับส่วนลดค่าเข้า 15 เปอร์เซ็นต์ และได้รับบัตรผ่านฟรี 2 ใบต่อเดือน (membership.txt) |
| I3 | in-scope | 0.612 | 3/3 | 7,554 | สำหรับกลุ่ม 4 คนที่เพิ่งเริ่มต้นเล่นบอร์ดเกม แนะนำให้ไปเลือกเกมจากชั้น Beginner Shelf ได้แก่ Codenames, Dixit, Azul, Splendor, Carcassonne, Ticket to Ride, Love Letter, UNO, Jenga… |
| I4 | in-scope | 0.564 | 3/3 | 4,714 | ที่จอดรถลูกค้าสามารถจอดได้ฟรี 2 ชั่วโมงเมื่อใช้จ่ายในร้านครบ 300 บาท โดยให้พนักงานประทับตราที่บัตรจอดรถ (location.txt) |
| I5 | in-scope | 0.575 | 3/3 | 9,307 | การจองโต๊ะสำหรับกลุ่ม 8 คน ต้องจองโต๊ะติดกัน 2 โต๊ะและชำระมัดจำ 400 บาท (อ้างอิงจาก reservation.txt chunk 3) หากจองทุกวันศุกร์ เสาร์ อาทิตย์ หรือกลุ่มเกิน 5 คน มัดจำจะเพิ่มเป็น 20… |
| C1 | cross-document | 0.636 | 3/3 | 9,528 | จัดปาร์ตี้วันเกิด 8 คน ควรจองห้องส่วนตัว Dragon Room ซึ่งรองรับได้สูงสุด 10 คน ต้องจองล่วงหน้าอย่างน้อย 7 วัน (ข้อมูลจาก events.txt) และค่าเช่าห้องนี้คิดตามอัตรา 500 บาทต่อชั่วโมง… |
| C2 | cross-document | 0.840 | 3/3 | 4,779 | นักเรียนที่เป็นสมาชิก Silver ควรใช้ส่วนลดนักเรียน 20 เปอร์เซ็นต์ ส่วนลดนี้จะใช้ร่วมกับช่วง Happy Hour ได้ (ข้อมูลจาก membership.txt และ pricing.txt) |
| C3 | cross-document | 0.582 | 3/3 | 5,672 | นำเค้กวันเกิดจากภายนอกเข้ามาได้ แต่ต้องชำระค่าเค้กตัด 100 บาท (menu.txt) หรือ (rules.txt) |
| P1 | paraphrase | 0.493 | 3/3 | 9,186 | ราคาเข้าร้านคิดแบบรายชั่วโมงต่อคน โดยวันธรรมดา (จันทร์-ศุกร์) ชั่วโมงละ 60 บาทต่อคน ส่วนวันเสาร์ วันอาทิตย์ หรือวันหยุดนักขัตฤกษ์ ชั่วโมงละ 80 บาทต่อคน ชั่วโมงแรกคิดเต็มชั่วโมง หล… |
| P2 | paraphrase | 0.545 | 3/3 | 8,993 | ร้านเปิดทำการทุกวัน ไม่มีวันหยุดประจำปี แต่มีกำหนดพิเศษในช่วงสงกรานต์วันที่ 13 ถึง 15 เมษายน เปิดเวลา 12:00 ถึง 18:00 น. และวันขึ้นปีใหม่วันที่ 1 มกราคม เปิดเวลา 12:00 ถึง 22:00 น… |
| O1 | out-of-scope | 0.178 | 3/3 | 25 | ขออภัย ไม่มีข้อมูลนี้ในฐานความรู้ของร้าน |
| O2 | out-of-scope | 0.487 | 3/3 | 3,044 | ขออภัย ไม่มีข้อมูลนี้ในฐานความรู้ของร้าน |

### `qwen3:1.7b`

| ID | Type | Best sim | Correct runs | Median ms | Answer snippet |
|---|---|---|---|---|---|
| I1 | in-scope | 0.590 | 3/3 | 352 | ค่าเข้าร้านวันธรรมดาคนละ 60 บาท (จากไฟล์: pricing.txt) |
| I2 | in-scope | 0.649 | 3/3 | 911 | จากไฟล์: membership.txt \| chunk 2 สมาชิก Gold ได้ส่วนลดค่าเข้า 15 เปอร์เซ็นต์ และได้บัตรผ่านฟรี 2 ใบต่อเดือน (อ้างอิงจากไฟล์: membership.txt \| chunk 2) |
| I3 | in-scope | 0.612 | 3/3 | 2,160 | จาก CONTEXT ที่ให้มา เกมที่เหมาะกับมือใหม่ 4 คน ได้แก่: - Codenames (2-8 คน 15-20 นาที) - Dixit (3-6 คน 30 นาที) - Azul (2-4 คน 30-45 นาที) - Splendor (2-4 คน 30 นาที) - Carcasson… |
| I4 | in-scope | 0.564 | 3/3 | 1,898 | จากไฟล์: location.txt \| chunk 4 \| ความคล้าย: 0.56 ที่จอดรถฟรีให้โดยไม่ต้องใช้จ่าย 2 ชั่วโมง หลังจากนั้นคิดชั่วโมงละ 20 บาท ซึ่งต้องใช้จ่ายครบ 300 บาท เพื่อให้ได้ที่จอดรถฟรี (อ้างอ… |
| I5 | in-scope | 0.575 | 3/3 | 1,743 | จากไฟล์: reservation.txt \| chunk 3 \| ความคล้าย: 0.57 จองโต๊ะกลุ่ม 8 คน ต้องใช้แพ็กเกจกิจกรรมหรือเหมาร้านตามรายละเอียดในเอกสาร events.txt แต่หากไม่มีข้อมูลใน events.txt สามารถจองได… |
| C1 | cross-document | 0.636 | 3/3 | 1,156 | จาก CONTEXT สามารถตอบคำถามได้ดังนี้: - ห้องที่ต้องจอง: ห้องส่วนตัว Dragon Room (จาก [2]) - ล่วงหน้า: ต้องจองล่วงหน้าอย่างน้อย 7 วัน (จาก [1]) - ห้องจุได้สูงสุด: 10 คน (จาก [2]) (อ… |
| C2 | cross-document | 0.840 | 3/3 | 1,182 | นักเรียนที่เป็นสมาชิก Silver ควรใช้ส่วนลดนักเรียน 20 เปอรเซ็นต์ ซึ่งมากกว่าส่วนลดสมาชิก Silver 10 เปอรเซ็นต์ และสามารถใช้ร่วมกับ Happy Hour ได้ (ตามข้อมูลในไฟล์: pricing.txt \| chu… |
| C3 | cross-document | 0.582 | 3/3 | 1,485 | จาก CONTEXT ที่ให้มา ร้านไม่อนุญาตให้นำเค้กจากภายนอกเข้ามาเสียค่าเค้กตัด 100 บาท ดังนั้น ไม่สามารถนำเค้กจากภายนอกเข้ามาได้ ซึ่งต้องจ่ายค่าเค้กตัด 100 บาท หากต้องการนำเค้กจากภายนอก… |
| P1 | paraphrase | 0.493 | 3/3 | 2,260 | ราคาค่าเข้าร้านทั้งหมดรวมภาษีมูลค่าเพิ่ม 7 เปอร์เซ็นต์ สามารถคำนวณได้ตามจำนวนชั่วโมงที่เล่น โดยมีอัตราดังนี้: - วันธรรมดา (วันจันทร์ถึงวันศุกร์): ชั่วโมงละ 60 บาท - วันเสาร์ วันอา… |
| P2 | paraphrase | 0.545 | 3/3 | 3,137 | ร้านเปิดทำการทุกวัน ไม่มีวันหยุดประจำปี ดังนั้น ร้านไม่มีวันหยุดวันใด โดยเฉพาะในช่วงสงกรานต์วันที่ 13 ถึง 15 เมษายน และวันขึ้นปีใหม่วันที่ 1 มกราคม ซึ่งเปิดทำการตามปกติ แต่หากมีกำ… |
| O1 | out-of-scope | 0.178 | 3/3 | 21 | ขออภัย ไม่มีข้อมูลนี้ในฐานความรู้ของร้าน |
| O2 | out-of-scope | 0.487 | 3/3 | 595 | ขออภัย ไม่มีข้อมูลนี้ในฐานความรู้ของร้าน |

## Notes

- The out-of-scope pizza question is deliberately near-domain: it passes the similarity threshold, so refusal depends on the system prompt. The stock-price question is rejected by the threshold before the LLM is called.
- Thai has no word spaces; chunking is paragraph/section based with ~400-character windows and 60-character overlap, so retrieval behavior can differ from whitespace-based systems.
- On this laptop (RTX 3050 Laptop, 4 GB VRAM) `qwen3.5:4b` runs partly on CPU, which limits latency. See README for the `ollama ps` reading.
- Correctness is a substring check of the expected key facts, not a human/LLM judge; a fluent but factually wrong answer fails.
