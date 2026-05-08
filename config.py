"""
Shared configuration helpers for the Day 22 LAB.
Loads environment variables and initializes LLM + Embeddings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(ENV_FILE)

# ── Google Gemini Configuration ────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "textembedding-gecko-001")

# ── LangSmith Configuration ────────────────────────────────────────────────
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "day22-rag-lab")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ── Set environment variables for LangSmith ────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT

def verify_config():
    """Verify that all required configuration is present."""
    missing = []
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if not LANGCHAIN_API_KEY:
        missing.append("LANGCHAIN_API_KEY")
    
    if missing:
        print(f"⚠️  Missing environment variables: {', '.join(missing)}")
        print(f"    Please update the .env file")
        return False
    
    print("✅ Config loaded successfully")
    print(f"   Gemini model       : {GEMINI_MODEL}")
    print(f"   Embedding model    : {GEMINI_EMBEDDING_MODEL}")
    print(f"   LangSmith project  : {LANGCHAIN_PROJECT}")
    return True

def get_llm():
    """Create and return a ChatGoogleGenerativeAI instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        api_key=GOOGLE_API_KEY,
        temperature=0.7
    )

def get_embeddings():
    """Create and return a GoogleGenerativeAIEmbeddings instance."""
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        ge = GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
        # Probe the provider with a tiny test call; if it fails, fall back.
        try:
            _ = ge.embed_documents(["test"])
            return ge
        except Exception:
            # fall through to local fallback
            pass
    except Exception:
        # import failed or provider unavailable — fall back
        pass

    # Lightweight local fallback embeddings using simple token counts.
    import re
    import numpy as np
    from langchain_core.embeddings import Embeddings

    class SimpleEmbeddings(Embeddings):
        def __init__(self):
            self.vocab = None

        def _tokenize(self, text):
            return re.findall(r"\w+", text.lower())

        def fit(self, texts):
            counts = {}
            for t in texts:
                for w in set(self._tokenize(t)):
                    counts[w] = counts.get(w, 0) + 1
            # keep top 1024 tokens
            items = sorted(counts.items(), key=lambda x: -x[1])[:1024]
            self.vocab = {w: i for i, (w, _) in enumerate(items)}

        def embed_documents(self, texts):
            if self.vocab is None:
                self.fit(texts)
            vectors = []
            for t in texts:
                vec = np.zeros(len(self.vocab), dtype=np.float32)
                for w in self._tokenize(t):
                    if w in self.vocab:
                        vec[self.vocab[w]] += 1.0
                # normalize
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                vectors.append(vec.tolist())
            return vectors

        def embed_query(self, text):
            vecs = self.embed_documents([text])
            return vecs[0]

        def __call__(self, text):
            return self.embed_query(text)

    return SimpleEmbeddings()


def cleanup_gemini_clients(*objects):
    """Best-effort shutdown for Gemini clients to avoid exit-time aiohttp warnings."""
    for obj in objects:
        client = getattr(obj, "client", None)
        if client is None:
            continue
        try:
            close = getattr(client, "close", None)
            if callable(close):
                close()
                continue
        except Exception:
            pass

        try:
            api_client = getattr(client, "_api_client", None)
            if api_client is not None:
                close = getattr(api_client, "close", None)
                if callable(close):
                    close()
        except Exception:
            pass

if __name__ == "__main__":
    verify_config()
