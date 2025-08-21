
import os
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import streamlit as st

from src.helper import init_pinecone, get_embedding_model, get_chat_model
from src.pdf_utils import process_coursebook_pdf, process_patient_pdf, load_ingested_books
from src.rag import build_retrievers, build_rag_chain, ask

# ================================
# 0) Page Setup & Utilities
# ================================

APP_TITLE = "🏥 First Aid Medical Chatbot"
COURSE_DIR = "data/medical_course"
PATIENT_DIR = "data/patient_data"
COURSE_NAMESPACE = "Medical_Course"

def ensure_dirs():
    os.makedirs(COURSE_DIR, exist_ok=True)
    os.makedirs(PATIENT_DIR, exist_ok=True)

ensure_dirs()

st.set_page_config(page_title=APP_TITLE, layout="wide")

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 7rem; }
.chat-bubble { padding: 0.85rem 1rem; border-radius: 14px; margin: 0.35rem 0; }
.user { background: #e8f9ee; border: 1px solid #bde7cd; text-align: right; }
.assistant { background: #f2f2f2; border: 1px solid #e0e0e0; }
.right { display: flex; justify-content: flex-end; }
.left { display: flex; justify-content: flex-start; }
span.pill { background:#fff3cd; border:1px solid #ffe8a1; border-radius: 999px; padding: 0.1rem 0.5rem; }
.fixed-input {
  position: fixed; bottom: 0; left: 0; right: 0;
  background: white; border-top: 1px solid #eee;
  padding: 0.6rem 1rem; z-index: 999;
}
.fixed-inner { max-width: 1200px; margin: 0 auto; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ================================
# 1) Initialize Backend & Session
# ================================

@st.cache_resource(show_spinner=False)
def _bootstrap_backends() -> Tuple[object, str, object, object]:
    pc, index_name = init_pinecone()
    embedding = get_embedding_model()
    llm = get_chat_model()
    return pc, index_name, embedding, llm

pc, index_name, embedding, llm = _bootstrap_backends()

if "current_patient" not in st.session_state:
    st.session_state.current_patient = "None"
if "chat_history" not in st.session_state:
    st.session_state.chat_history: Dict[str, List[Dict]] = {}
if "uploaded_courses" not in st.session_state:
    try:
        ing = load_ingested_books().get(COURSE_NAMESPACE, [])
    except Exception:
        ing = []
    st.session_state.uploaded_courses = set(ing)
if "patients" not in st.session_state:
    st.session_state.patients = set()
if "patient_meta" not in st.session_state:
    st.session_state.patient_meta: Dict[str, Dict] = {}
if "message_count" not in st.session_state:
    st.session_state.message_count = 0
if "last_uploaded_patient" not in st.session_state:
    st.session_state.last_uploaded_patient = None

# ================================
# 2) Helpers
# ================================

def highlight_medical_terms(text: str) -> str:
    terms = [
        "epinephrine", "adrenaline", "aspirin", "ibuprofen", "paracetamol", "acetaminophen",
        "CPR", "cardiopulmonary resuscitation", "Heimlich", "tourniquet",
        "shock", "anaphylaxis", "asthma", "stroke", "burn", "fracture",
        "airway", "breathing", "circulation", "defibrillator", "AED",
        "bleeding", "poisoning", "choking", "seizure"
    ]
    def repl(m): return f"<span class='pill'>{m.group(0)}</span>"
    for t in sorted(terms, key=len, reverse=True):
        text = re.sub(rf"(?i)\b{re.escape(t)}\b", repl, text)
    return text

@st.cache_data(ttl=30)
def list_patient_namespaces() -> Dict[str, int]:
    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        out = {}
        for ns, info in namespaces.items():
            if ns and ns != COURSE_NAMESPACE:
                out[ns] = int(info.get("vector_count", 0))
        return out
    except Exception:
        return {}

def ensure_chat_bucket(patient_id: str):
    if patient_id not in st.session_state.chat_history:
        st.session_state.chat_history[patient_id] = []

def patient_card(patient_id: str, chunk_count: Optional[int], mode_label: str):
    with st.container(border=True):
        st.markdown(f"### 🏥 Patient: `{patient_id}`" if patient_id != "None" else "### 🧑‍⚕️ No patient selected")
        col1, col2, col3 = st.columns(3)
        cc = "—" if chunk_count is None else f"{chunk_count}"
        last = "—"
        if patient_id in st.session_state.patient_meta:
            last = st.session_state.patient_meta[patient_id].get("last_update", "—")
            if chunk_count is None:
                chunk_count = st.session_state.patient_meta[patient_id].get("chunk_count")
                cc = "—" if chunk_count is None else f"{chunk_count}"
        col1.metric("Chunks in Pinecone", cc)
        col2.metric("Last update", last)
        col3.metric("Search mode", mode_label)

def icon_for_namespace(ns: str) -> str:
    return "📖" if ns == COURSE_NAMESPACE else "🏥"

# ================================
# 3) Sidebar Controls
# ================================

with st.sidebar:
    st.header("⚙️ Controls")

    mode = st.selectbox("🔍 Search mode", ["Both", "Patient Only", "Coursebook Only"], index=0)
    if mode == "Both":
        course_w, patient_w = 0.6, 0.4
        mode_label = "Both (weighted)"
    elif mode == "Patient Only":
        course_w, patient_w = 0.0, 1.0
        mode_label = "Patient only"
    else:
        course_w, patient_w = 1.0, 0.0
        mode_label = "Coursebook only"

    st.divider()

    st.subheader("📚 Upload Coursebook PDF")
    cb_file = st.file_uploader("Choose a coursebook PDF", type=["pdf"], key="course_pdf_uploader")
    if cb_file is not None:
        filename = cb_file.name
        save_path = os.path.join(COURSE_DIR, filename)
        if filename not in st.session_state.uploaded_courses:
            with open(save_path, "wb") as f:
                f.write(cb_file.read())
            with st.spinner("Processing coursebook…"):
                _ = process_coursebook_pdf(
                    pdf_path=save_path,
                    embedding=embedding,
                    index_name=index_name,
                    batch_size=100,
                )
            st.session_state.uploaded_courses.add(filename)
            st.success(f"Uploaded & indexed **{filename}** into `{COURSE_NAMESPACE}`.")
        else:
            st.info(f"Already processed: **{filename}**")

    st.divider()

    st.subheader("🏥 Upload Patient PDF")
    pt_file = st.file_uploader("Choose a patient PDF", type=["pdf"], key="patient_pdf_uploader")
    if pt_file is not None:
        filename = pt_file.name
        patient_id = os.path.splitext(filename)[0]
        if st.session_state.last_uploaded_patient == filename:
            st.info(f"Already processed: **{filename}**")
        else:
            save_path = os.path.join(PATIENT_DIR, filename)
            with open(save_path, "wb") as f:
                f.write(pt_file.read())
            with st.spinner(f"Processing patient PDF for `{patient_id}`…"):
                vs = process_patient_pdf(
                    pdf_path=save_path,
                    patient_id=patient_id,
                    embedding=embedding,
                    pc=pc,
                    index_name=index_name,
                    chunk_size=1000,
                    chunk_overlap=200,
                )
            st.session_state.patients.add(patient_id)
            st.session_state.current_patient = patient_id
            st.session_state.patient_meta[patient_id] = {
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.last_uploaded_patient = filename
            st.success(f"Uploaded & indexed **{filename}** into `{patient_id}` namespace.")

    st.divider()

    st.subheader("👤 Patient Selector")
    live = list_patient_namespaces()
    for ns in live.keys():
        st.session_state.patients.add(ns)
    patient_options = ["None"] + sorted(st.session_state.patients)
    selected = st.selectbox("Active patient", options=patient_options,
                            index=patient_options.index(st.session_state.current_patient)
                            if st.session_state.current_patient in patient_options else 0,
                            key="patient_selector")
    if selected != st.session_state.current_patient:
        st.session_state.current_patient = selected

# ================================
# 4) Main Panel: Patient Card + Chat
# ================================

current = st.session_state.current_patient
live_counts = list_patient_namespaces()
chunk_count = live_counts.get(current) if current != "None" else None
patient_card(current, chunk_count, mode_label)

ensure_chat_bucket(current)

st.markdown("### 💬 Chat")
for turn in st.session_state.chat_history[current]:
    with st.container():
        st.markdown('<div class="right"><div class="chat-bubble user">' + turn["question"] + '</div></div>', unsafe_allow_html=True)
    answer_html = highlight_medical_terms(turn["answer"])
    with st.container():
        st.markdown('<div class="left"><div class="chat-bubble assistant">' + answer_html + '</div></div>', unsafe_allow_html=True)

    with st.expander("🔎 Sources & Contexts"):
        if turn.get("sources"):
            st.markdown("**Sources**")
            for s in turn["sources"]:
                ns = s.get("namespace", "")
                icon = icon_for_namespace(ns)
                src = s.get("source", "—")
                page = s.get("page", "—")
                st.markdown(f"{icon} `{ns}` — {src} (page {page})")
        if turn.get("contexts"):
            st.markdown("---")
            st.markdown("**Retrieved Contexts**")
            for ctx_idx, ctx in enumerate(turn["contexts"]):
                lbl = f"{ctx.get('source','—')} (p.{ctx.get('page','—')}) • {ctx.get('namespace','—')}"
                unique_key = f"ctx_{current}_{id(turn)}_{ctx_idx}"
                st.text_area(lbl, value=ctx.get("chunk", ""), height=140, key=unique_key)

# ================================
# 5) Fixed Input Bar
# ================================
st.markdown('<div class="fixed-input"><div class="fixed-inner">', unsafe_allow_html=True)
colA, colB = st.columns([6, 1])
with colA:
    q_key = f"question_{current}_{st.session_state.message_count}"
    question = st.text_input("Ask your medical question…", label_visibility="collapsed", key=q_key, value="")
with colB:
    submitted = st.button("Send", use_container_width=True, key=f"send_{st.session_state.message_count}")
st.markdown('</div></div>', unsafe_allow_html=True)

if submitted and question.strip():
    patient_id = None if current == "None" else current

    if mode == "Both":
        retriever = build_retrievers(
            index_name=index_name, embedding=embedding, patient_id=patient_id,
            k_course=4, k_patient=4, course_namespace=COURSE_NAMESPACE,
            weights=(course_w, patient_w),
        )
    elif mode == "Patient Only":
        retriever = build_retrievers(
            index_name=index_name,
            embedding=embedding,
            patient_id=patient_id,
            k_course=4,       # must be >0
            k_patient=5,
            course_namespace=COURSE_NAMESPACE,
            weights=(0.0, 1.0),   # course ignored
        )
    else:  # Coursebook Only
        retriever = build_retrievers(
            index_name=index_name,
            embedding=embedding,
            patient_id=None,
            k_course=6,
            k_patient=4,      # must be >0
            course_namespace=COURSE_NAMESPACE,
            weights=(1.0, 0.0),   # patient ignored
        )

    chain = build_rag_chain(llm, retriever)
    with st.spinner("Thinking…"):
        result = ask(chain, question.strip())

    st.session_state.chat_history[current].append({
        "question": question.strip(),
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "contexts": result.get("contexts", []),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode_label,
    })

    st.session_state.message_count += 1

    if current != "None":
        try:
            counts = list_patient_namespaces()
            st.session_state.patient_meta.setdefault(current, {})
            st.session_state.patient_meta[current]["chunk_count"] = counts.get(current)
            st.session_state.patient_meta[current]["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    st.rerun()

# ================================
# 6) Footer Disclaimer
# ================================
st.markdown("---")
st.markdown("""
> ⚠️ **Disclaimer:** This AI assistant provides *first-aid guidance only* based on the provided medical documents.
> It is **not** a substitute for professional medical advice, diagnosis, or treatment. In emergencies, call local emergency services.
""")
