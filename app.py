# app.py

import os
import streamlit as st
from src.helper import init_pinecone, get_embedding_model, get_chat_model
from src.pdf_utils import process_patient_pdf, process_coursebook_pdf
from src.rag import build_retrievers, build_rag_chain, ask

# -------------------------------
# 1. Page Config + Styling
# -------------------------------
st.set_page_config(
    page_title="🏥 First Aid Medical Chatbot",
    page_icon="🏥",
    layout="wide"
)

st.markdown(
    """
    <style>
        .fixed-title {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            padding: 10px;
            background-color: #0e1117;
            z-index: 9999;
            border-bottom: 1px solid #333;
        }
        .block-container {
            padding-top: 80px !important;
        }
    </style>
    <div class="fixed-title">
        <h2 style="color:white;">🏥 First Aid Medical Chatbot</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# 2. Init backend (once per session)
# -------------------------------
if "pc" not in st.session_state:
    st.session_state.pc, st.session_state.index_name = init_pinecone()
    st.session_state.embedding = get_embedding_model()
    st.session_state.llm = get_chat_model(provider="openrouter")
    st.session_state.patient_id = None

# ensure dirs exist
os.makedirs("data/medical_course", exist_ok=True)
os.makedirs("data/patient_data", exist_ok=True)

# -------------------------------
# 3. Sidebar - Upload & Select
# -------------------------------
st.sidebar.header("📂 Manage Documents")

# Upload Coursebook
course_upload = st.sidebar.file_uploader("Upload Coursebook PDF", type="pdf", key="course_uploader")
if course_upload is not None and course_upload.name not in st.session_state.get("uploaded_courses", []):
    tmp_path = os.path.join("data/medical_course", course_upload.name)
    with open(tmp_path, "wb") as f:
        f.write(course_upload.getbuffer())

    st.sidebar.info(f"Processing coursebook: {course_upload.name}")
    process_coursebook_pdf(
        pdf_path=tmp_path,
        embedding=st.session_state.embedding,
        index_name=st.session_state.index_name,
    )
    st.sidebar.success("✅ Coursebook uploaded")

    # remember it
    if "uploaded_courses" not in st.session_state:
        st.session_state.uploaded_courses = []
    st.session_state.uploaded_courses.append(course_upload.name)

# Upload Patient
patient_upload = st.sidebar.file_uploader("Upload Patient PDF", type="pdf", key="patient_uploader")
if patient_upload is not None:
    tmp_path = os.path.join("data/patient_data", patient_upload.name)
    with open(tmp_path, "wb") as f:
        f.write(patient_upload.getbuffer())

    patient_id = os.path.splitext(patient_upload.name)[0]
    st.sidebar.info(f"Processing patient: {patient_id}")
    vs = process_patient_pdf(
        pdf_path=tmp_path,
        patient_id=patient_id,
        embedding=st.session_state.embedding,
        pc=st.session_state.pc,
        index_name=st.session_state.index_name,
    )

    if vs:
        st.session_state.patient_id = patient_id
        st.sidebar.success(f"✅ Patient {patient_id} uploaded & active")
    else:
        st.sidebar.error(f"❌ Failed to upload patient {patient_id}")

# Patient selection
# Fetch namespaces directly from Pinecone
index = st.session_state.pc.Index(st.session_state.index_name)

# Pinecone doesn't have a direct "list_namespaces" API,
# but we can hack it by querying metadata stats
stats = index.describe_index_stats()

# get all namespaces except coursebook
namespaces = list(stats.get("namespaces", {}).keys())
patients = [ns for ns in namespaces if ns != "Medical_Course"]

# Sidebar dropdown
selected_patient = st.sidebar.selectbox("Select Patient", patients)

st.session_state.patient_id = None if selected_patient == "None" else selected_patient

# -------------------------------
# 4. Main Chat Interface
# -------------------------------
st.subheader("💬 Chat with your documents")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_q = st.text_input("Ask a medical question:")
if user_q:
    retriever = build_retrievers(
        st.session_state.index_name,
        st.session_state.embedding,
        patient_id=st.session_state.patient_id,
    )
    chain = build_rag_chain(st.session_state.llm, retriever)

    resp = ask(chain, user_q)

    # append to history
    st.session_state.chat_history.append({
        "question": user_q,
        "answer": resp["answer"],
        "sources": resp["sources"],
        "contexts": resp["contexts"]
    })

# Display history
for i, turn in enumerate(st.session_state.chat_history):
    st.markdown(f"**👤 You:** {turn['question']}")
    st.markdown(f"**🤖 Bot:** {turn['answer']}")

    with st.expander(f"📑 Sources (Q{i+1})"):
        for s in turn["sources"]:
            st.write(f"- {s['source']} (page {s['page']}, ns={s['namespace']})")

    with st.expander(f"📝 Retrieved Context (Q{i+1})"):
        for c in turn["contexts"]:
            st.markdown(f"**{c['source']} - page {c['page']} ({c['namespace']})**")
            st.write(c["chunk"])
            st.markdown("---")
