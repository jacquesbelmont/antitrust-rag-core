"""
Hierarchical Legal RAG — Test UI
Run: streamlit run ui.py
"""
import time

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Legal RAG",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Hierarchical Legal RAG")
st.caption("Upload legal documents and ask questions in natural language.")

tab_upload, tab_search = st.tabs(["📄 Upload Document", "🔍 Search"])


# ── Upload tab ────────────────────────────────────────────────────────────────

with tab_upload:
    st.subheader("Upload a document for ingestion")

    input_mode = st.radio("Input mode", ["PDF file", "Paste text"], horizontal=True)

    title = st.text_input("Title (optional)")
    source = st.text_input("Source / reference (optional)")

    if input_mode == "PDF file":
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
        ready = uploaded_file is not None
    else:
        raw_text = st.text_area("Document text", height=300, placeholder="Paste the full document text here…")
        ready = bool(raw_text.strip())

    if st.button("Upload & ingest", disabled=not ready, type="primary"):
        with st.spinner("Uploading…"):
            try:
                if input_mode == "PDF file":
                    resp = requests.post(
                        f"{API_BASE}/v1/documents/upload-pdf",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        data={"title": title or "", "source": source or ""},
                        timeout=30,
                    )
                else:
                    resp = requests.post(
                        f"{API_BASE}/v1/documents/upload",
                        json={"title": title or None, "source": source or None, "text": raw_text},
                        timeout=30,
                    )
            except requests.ConnectionError:
                st.error("Cannot reach the API at http://localhost:8000 — is it running?")
                st.stop()

        if resp.status_code == 202:
            data = resp.json()
            task_id = data["task_id"]
            document_id = data["document_id"]

            st.success(f"Queued! **task_id**: `{task_id}`")
            col1, col2 = st.columns(2)
            col1.metric("Document ID", document_id[:8] + "…")
            col2.metric("Task ID", task_id[:8] + "…")

            # Poll for completion
            status_placeholder = st.empty()
            progress = st.progress(0)
            for attempt in range(60):  # poll up to ~60 s
                time.sleep(1)
                progress.progress(min(attempt / 60, 0.95))
                try:
                    job_resp = requests.get(f"{API_BASE}/v1/jobs/{task_id}", timeout=10)
                    job = job_resp.json()
                except Exception:
                    continue

                job_status = job.get("status", "unknown")
                status_placeholder.info(f"Status: **{job_status}**")

                if job_status == "SUCCESS":
                    progress.progress(1.0)
                    result = job.get("result") or {}
                    chunks = result.get("chunks_created", "?")
                    status_placeholder.success(f"✅ Ingestion complete — **{chunks}** chunks created and indexed.")
                    break
                elif job_status == "FAILURE":
                    progress.progress(1.0)
                    status_placeholder.error(f"❌ Ingestion failed: {job.get('error', 'unknown error')}")
                    break
            else:
                status_placeholder.warning("⏳ Still processing — check back later or watch worker logs.")
        else:
            st.error(f"API returned {resp.status_code}: {resp.text}")


# ── Search tab ────────────────────────────────────────────────────────────────

with tab_search:
    st.subheader("Ask a question about the legal corpus")

    query = st.text_area(
        "Your question",
        height=100,
        placeholder="e.g. Quelles sont les conditions pour obtenir une indemnisation selon l'article 3 ?",
    )
    top_k = st.slider("Context chunks to retrieve", min_value=1, max_value=20, value=5)

    if st.button("Search", disabled=not query.strip(), type="primary"):
        with st.spinner("Running RAG pipeline…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/v1/search/",
                    json={"query": query, "top_k": top_k},
                    timeout=120,
                )
            except requests.ConnectionError:
                st.error("Cannot reach the API at http://localhost:8000 — is it running?")
                st.stop()

        if resp.status_code == 200:
            data = resp.json()

            st.markdown("### Answer")
            st.markdown(data["answer"])

            # Evaluation metrics
            evaluation = data.get("evaluation") or {}
            if evaluation:
                with st.expander("Evaluation metrics", expanded=False):
                    cols = st.columns(len(evaluation))
                    for col, (k, v) in zip(cols, evaluation.items()):
                        col.metric(k, f"{v:.2f}" if isinstance(v, float) else v)

            # Context chunks
            context = data.get("context") or []
            st.markdown(f"### Context ({len(context)} chunks)")
            for i, chunk in enumerate(context, 1):
                hierarchy = " › ".join(chunk.get("hierarchy_path") or [])
                score = chunk.get("score", 0)
                with st.expander(f"Chunk {i}  |  {hierarchy or 'No hierarchy'}  |  score {score:.3f}"):
                    st.code(chunk.get("text", ""), language=None)

        elif resp.status_code == 422:
            st.warning(f"Query rejected: {resp.json().get('detail', 'invalid query')}")
        elif resp.status_code == 503:
            st.error("Search service unavailable — Weaviate or Ollama may not be running.")
        else:
            st.error(f"API returned {resp.status_code}: {resp.text}")
