# SecureDoc Core

PDF 문서 기반 멀티모달 RAG(Retrieval-Augmented Generation) 시스템

---

## 시스템 개요

SecureDoc Core는 PDF 문서를 업로드하면 텍스트/이미지/표를 파싱하고, 하이브리드 검색과 LLM을 결합하여 질문에 대한 정확한 답변과 근거(citation)를 제공하는 엔드투엔드 RAG 파이프라인입니다.

---

## 아키텍처

### 1. 전체 시스템 구성도

![System Architecture](docs/archi1.png)

| 구성 요소 | 역할 | 상세 |
|---|---|---|
| **FastAPI Server** | API 게이트웨이 | `/upload`, `/query`, `/documents` 엔드포인트 (port 8000) |
| **LangGraph Agent** | 오케스트레이션 | `analyze_query` → `generate_answer` → `verify` 3단계 에이전트 |
| **Hybrid Retrieval** | 검색 엔진 | BGE-M3 Embedder + Dense/Sparse Search + RRF Fusion + Reranker |
| **VLM (GPU 0)** | 이미지 캡셔닝 | Qwen2-VL-7B-AWQ, vLLM port 8001 |
| **LLM (GPU 1)** | 추론 및 검증 | Qwen2.5-14B-AWQ, vLLM port 8002 |
| **Langfuse** | 관측성(Observability) | 트레이스 수집 (port 3000) |

### 2. 질의 처리 시퀀스

![Query Sequence](docs/archi2.png)

**처리 흐름:**

1. **질문 분류** — LLM(GPU 1)이 질문 타입을 `simple` / `multi_hop` / `visual`로 분류
2. **하이브리드 검색** — Dense + Sparse + RRF + Reranker를 통해 top-5 청크 검색
3. **이미지 캡셔닝** (조건부) — figure 청크가 포함된 경우 VLM(GPU 0)에서 이미지 → 텍스트 변환
4. **답변 생성** — 검색 컨텍스트 + 비주얼 컨텍스트를 결합하여 LLM(GPU 1)에서 답변 + citation 생성
5. **Faithfulness 검증** — 답변의 faithful 스코어가 0.7 이상인지 체크
6. **재검색 루프** — 검증 실패 시 re-retrieve 후 재생성
7. **최종 응답** — answer + citations를 사용자에게 반환

### 3. 파이프라인 요약

![Pipeline Summary](docs/archi3.png)

```
USER (질문+PDF) → FastAPI (POST /query) → Retrieval (Hybrid+Rerank, top-5)
    → GPU 0: VLM (Qwen2-VL, 이미지→텍스트)
    → GPU 1: LLM (Qwen2.5, 답변+검증)
    → Answer + Citation + Score + Trace
```

### 4. 문서 인제스트 파이프라인

![Ingest Pipeline](docs/archi4.png)

**PDF 업로드 후 처리 단계:**

| 단계 | 처리 | 도구 |
|---|---|---|
| **PARSE (추출)** | 텍스트+이미지 추출 / 레이아웃 분석 / 표 구조 추출 | PyMuPDF, Surya OCR, pdfplumber |
| **CHUNK (분할)** | Sliding Window (size=500, overlap=50) + 메타데이터(page, bbox, type, path) | - |
| **EMBED (변환)** | Dense 임베딩(1024-dim) + 한국어 형태소 분석 | BGE-M3, Kiwi |
| **STORE (저장)** | 벡터 DB / 역색인 / 이미지 파일 | ChromaDB, BM25, File System |

---

## 기술 스택

- **API**: FastAPI
- **Agent**: LangGraph
- **LLM Serving**: vLLM (Qwen2.5-14B-AWQ, Qwen2-VL-7B-AWQ)
- **Vector DB**: ChromaDB
- **Sparse Search**: BM25 (Inverted Index)
- **Embedding**: BGE-M3 (1024-dim Dense)
- **Korean NLP**: Kiwi (형태소 분석)
- **PDF Parsing**: PyMuPDF, Surya OCR, pdfplumber
- **Observability**: Langfuse
- **GPU**: 2x A4000 (GPU 0: VLM, GPU 1: LLM)
