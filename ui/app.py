"""SecureDoc Core Streamlit UI - PDF 문서 Q&A 인터페이스"""

import streamlit as st
import requests

API_BASE = st.sidebar.text_input("API 서버 주소", value="http://localhost:8000")

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="SecureDoc Core",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 커스텀 CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background: #f8f9fa;
        border-left: 4px solid #4361ee;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .score-high { background: #d4edda; color: #155724; }
    .score-mid  { background: #fff3cd; color: #856404; }
    .score-low  { background: #f8d7da; color: #721c24; }
    .type-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        background: #e2e3f1;
        color: #3d405b;
    }
    .doc-item {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
    }
    .stChatMessage { max-width: 100% !important; }
</style>
""", unsafe_allow_html=True)


# ── 세션 상태 초기화 ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 SecureDoc Core")
    st.caption("PDF 기반 멀티모달 RAG 시스템")
    st.divider()

    # PDF 업로드
    st.markdown("### 문서 업로드")
    uploaded_file = st.file_uploader(
        "PDF 파일 선택", type=["pdf"], label_visibility="collapsed"
    )
    if uploaded_file and st.button("📤 업로드", use_container_width=True):
        with st.spinner("문서 인덱싱 중..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                resp = requests.post(f"{API_BASE}/upload", files=files, timeout=300)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"✅ 업로드 완료!\n\n"
                        f"- 문서 ID: `{data['doc_id']}`\n"
                        f"- 페이지: {data['page_count']}p\n"
                        f"- 청크: {data['chunk_count']}개"
                    )
                else:
                    st.error(f"업로드 실패: {resp.text}")
            except requests.ConnectionError:
                st.error("API 서버에 연결할 수 없습니다.")
            except Exception as e:
                st.error(f"오류: {e}")

    st.divider()

    # 업로드된 문서 목록
    st.markdown("### 업로드된 문서")
    try:
        resp = requests.get(f"{API_BASE}/documents", timeout=10)
        if resp.status_code == 200:
            documents = resp.json()
            if documents:
                selected_doc = st.selectbox(
                    "문서 선택 (질의 범위 지정)",
                    options=["전체 문서"] + [f"{d['filename']} ({d['doc_id']})" for d in documents],
                )
                for doc in documents:
                    st.markdown(
                        f'<div class="doc-item">'
                        f'📎 <strong>{doc["filename"]}</strong><br>'
                        f'<small>ID: {doc["doc_id"]} | '
                        f'{doc["page_count"]}p | '
                        f'{doc["chunk_count"]}청크</small></div>',
                        unsafe_allow_html=True,
                    )
            else:
                selected_doc = "전체 문서"
                st.info("업로드된 문서가 없습니다.")
        else:
            selected_doc = "전체 문서"
            st.warning("문서 목록을 불러올 수 없습니다.")
    except requests.ConnectionError:
        selected_doc = "전체 문서"
        st.warning("API 서버에 연결할 수 없습니다.")
    except Exception:
        selected_doc = "전체 문서"

    # 선택된 문서 ID 추출
    doc_id_filter = None
    if selected_doc != "전체 문서" and "(" in selected_doc:
        doc_id_filter = selected_doc.split("(")[-1].rstrip(")")


# ── 메인 영역 ────────────────────────────────────────────────
st.markdown('<p class="main-header">SecureDoc Core</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    "PDF 문서를 업로드하고 자연어로 질문하세요. "
    "하이브리드 검색과 LLM이 정확한 답변과 근거를 제공합니다."
    "</p>",
    unsafe_allow_html=True,
)

# 대화 히스토리 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg.get("extra"):
            st.markdown(msg["extra"], unsafe_allow_html=True)

# 질문 입력
if question := st.chat_input("PDF 문서에 대해 질문하세요..."):
    # 사용자 메시지 표시 및 저장
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # API 호출
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                payload = {"question": question}
                if doc_id_filter:
                    payload["doc_id"] = doc_id_filter

                resp = requests.post(
                    f"{API_BASE}/query", json=payload, timeout=120
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "답변을 생성할 수 없습니다.")
                    score = data.get("faithfulness_score", 0)
                    qtype = data.get("query_type", "simple")
                    citations = data.get("citations", [])

                    # 답변 표시
                    st.markdown(answer)

                    # 메타 정보
                    score_class = "score-high" if score >= 0.7 else ("score-mid" if score >= 0.4 else "score-low")
                    type_labels = {"simple": "단순 질문", "multi_hop": "복합 질문", "visual": "시각 질문"}
                    meta_html = (
                        f'<span class="type-badge">{type_labels.get(qtype, qtype)}</span> '
                        f'<span class="score-badge {score_class}">'
                        f"신뢰도: {score:.0%}</span>"
                    )
                    st.markdown(meta_html, unsafe_allow_html=True)
                    st.progress(min(score, 1.0))

                    # Citation 표시
                    extra_html = ""
                    if citations:
                        with st.expander(f"📚 근거 ({len(citations)}개)", expanded=False):
                            for i, cite in enumerate(citations, 1):
                                cite_html = (
                                    f'<div class="citation-card">'
                                    f"<strong>#{i}</strong> "
                                    f'<small>p.{cite.get("page_num", "?")} | '
                                    f'관련도: {cite.get("relevance_score", 0):.0%}</small><br>'
                                    f'{cite.get("content_preview", "")}'
                                    f"</div>"
                                )
                                st.markdown(cite_html, unsafe_allow_html=True)
                                extra_html += cite_html

                    # 세션에 저장
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "extra": meta_html,
                    })

                else:
                    error_msg = f"오류 발생: {resp.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

            except requests.ConnectionError:
                err = "API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요."
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as e:
                err = f"오류 발생: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
