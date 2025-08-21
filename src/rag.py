from typing import List, Dict, Optional, Tuple

from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import EnsembleRetriever

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


try:
    from src.prompt import system_prompt
except Exception:
    system_prompt = (
        "You are a First-Aid Medical assistant for question-answering tasks. "
        "Use the provided context to answer the user's question. "
        "If you don't know, say you don't know. Be concise and safe."
    )


# -------------------------------
# Build retrievers (course + patient)
# -------------------------------
def build_retrievers(
    index_name: str,
    embedding,
    patient_id: Optional[str] = None,
    k_course: int = 4,
    k_patient: int = 4,
    course_namespace: str = "Medical_Course",
    weights: Tuple[float, float] = (0.9, 0.1),
):
    """
    Creates:
      - coursebook retriever (namespace=Medical_Course)
      - optional patient retriever (namespace=patient_id)
      - if patient_id given: returns EnsembleRetriever(course, patient) with weights
      - else: returns course retriever alone
    """
    # Vector store handle (no ingestion here, just read path)
    vs = PineconeVectorStore(index_name=index_name, embedding=embedding)

    # Coursebook retriever
    course_retriever = vs.as_retriever(
        search_kwargs={"k": k_course, "namespace": course_namespace}
    )

    if not patient_id:
        # only course retriever
        return course_retriever

    # Patient retriever
    patient_retriever = vs.as_retriever(
        search_kwargs={"k": k_patient, "namespace": patient_id}
    )

    # Combine
    ens = EnsembleRetriever(
        retrievers=[course_retriever, patient_retriever],
        weights=list(weights),
    )
    return ens


# -------------------------------
# Prompt + Chains
# -------------------------------
def build_prompt() -> ChatPromptTemplate:
    """
    Stuffing-style prompt (LLM sees all retrieved docs in one go).
    Uses a strict system prompt and injects {context} + {input}.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Question:\n{input}\n\n"
                "Use the following context (quotes are chunks) to answer:\n\n{context}\n\n"
                "Answer:"
            ),
        ]
    )
    return prompt


def build_rag_chain(llm, retriever) -> any:
    """
    Creates a classic RAG chain:
      retriever -> stuff documents -> LLM
    Returns a callable chain with .invoke({'input': question})
    """
    prompt = build_prompt()
    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)
    return rag_chain


# -------------------------------
# Helper: run a query
# -------------------------------
def ask(chain, question: str) -> Dict:
    """
    Invokes the RAG chain and returns:
      {
        'answer': str,
        'sources': List[Dict],   # metadata (for citations)
        'contexts': List[Dict]   # actual retrieved text chunks
      }
    """
    result = chain.invoke({"input": question})

    docs: List = result.get("context", [])

    sources, contexts = [], []
    for d in docs:
        meta = d.metadata or {}
        text = d.page_content

        sources.append(
            {
                "source": meta.get("source"),
                "page": meta.get("page"),
                "namespace": meta.get("namespace"),
            }
        )

        contexts.append(
            {
                "chunk": text,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "namespace": meta.get("namespace"),
            }
        )

    return {
        "answer": result.get("answer", ""),
        "sources": sources,   # light info (for inline citations)
        "contexts": contexts  # full chunks (for expandable UI panels)
    }
