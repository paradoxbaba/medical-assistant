import os
from typing import Tuple
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  # OpenAI-compatible (works with OpenRouter via base_url)

# Load .env once at import time
load_dotenv()


# -------------------------------
# Pinecone: init + ensure index
# -------------------------------
def init_pinecone(index_name: str = "medical-experiment", dimension: int = 768) -> Tuple[Pinecone, str]:
    """
    Bootstraps Pinecone and ensures your index exists.
    Returns: (pc_client, index_name)
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY missing in .env")

    pc = Pinecone(api_key=api_key)

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"✅ Created Pinecone index: {index_name}")
    else:
        print(f"ℹ️ Using existing Pinecone index: {index_name}")

    return pc, index_name


# -------------------------------
# Embeddings (CPU-only)
# -------------------------------
def get_embedding_model():
    """
    HuggingFace embeddings (CPU). Works on Streamlit Cloud.
    - 'multi-qa-mpnet-base-dot-v1' -> 768-dim
    """
    return HuggingFaceEmbeddings(
        model_name="multi-qa-mpnet-base-dot-v1",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# -------------------------------
# Chat LLM (OpenRouter or OpenAI)
# -------------------------------
def get_chat_model(
    provider: str = "openrouter",
    model: str = "deepseek/deepseek-chat",  # you can change later
    temperature: float = 0.2,
    timeout: int = 60,
    max_retries: int = 2,
):
    """
    Returns a ChatOpenAI instance.
    - If provider='openrouter', expects OPENROUTER_API_KEY. Uses OpenAI-compatible endpoint via base_url.
    - If provider='openai', expects OPENAI_API_KEY, uses the native OpenAI endpoint.
    """
    if provider.lower() == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing in .env")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,   # OpenAI-compatible base
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )

    elif provider.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing in .env")

        # No base_url needed for official OpenAI
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )

    else:
        raise ValueError("provider must be either 'openrouter' or 'openai'")
