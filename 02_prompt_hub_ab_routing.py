"""
Step 2 — Prompt Hub & A/B Routing with Gemini
==============================================
TASK:
  1. Write two distinct system prompts (V1: concise, V2: structured)
  2. Push both to LangSmith Prompt Hub via client.push_prompt()
  3. Pull them back via client.pull_prompt()
  4. Implement deterministic A/B routing: hash(request_id) % 2 → V1 or V2
  5. Run all 50 questions through the router → ≥ 50 more LangSmith traces

DELIVERABLE: 2 named prompts visible in https://smith.langchain.com Prompt Hub
"""

import os
import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment ────────────────────────────────────────────────────────
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "day22-rag-lab")

# ── Imports ─────────────────────────────────────────────────────────────────
from config import LANGCHAIN_API_KEY, cleanup_gemini_clients, get_llm, get_embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable
from config import GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_EMBEDDING_MODEL, LANGCHAIN_API_KEY
from qa_pairs import QA_PAIRS

# ── Initialize components ───────────────────────────────────────────────────
llm = get_llm()
embeddings = get_embeddings()

client = Client(api_key=LANGCHAIN_API_KEY)

# ── Define two prompt templates ─────────────────────────────────────────────
SYSTEM_V1 = """You are a helpful AI assistant. Answer the user's question using ONLY the provided context. Keep your answer concise (2-4 sentences). If the context does not contain the answer, say: 'I don't have enough information.'

Context:
{context}"""

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "{question}"),
])

SYSTEM_V2 = """You are an expert AI tutor. Provide a structured, accurate answer based on the context.

Instructions:
1. Read the context carefully.
2. Identify key facts relevant to the question.
3. Write a clear, well-organized answer (3-5 sentences).
4. State explicitly if the context lacks sufficient information.

Context:
{context}"""

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "{question}"),
])

# Prompt Hub names
PROMPT_V1_NAME = "day22-gemini-rag-v1-concise"
PROMPT_V2_NAME = "day22-gemini-rag-v2-structured"

# ── Push prompts to Hub ─────────────────────────────────────────────────────
def push_prompts_to_hub():
    """
    Upload both prompt versions to LangSmith Prompt Hub.
    """
    print("📤 Pushing prompts to LangSmith Hub...")

    def _handle_push(label, prompt_name, prompt_obj, description):
        try:
            url = client.push_prompt(
                prompt_name,
                object=prompt_obj,
                description=description
            )
            print(f"   ✓ {label} pushed: {url}")
        except Exception as e:
            message = str(e)
            if "409" in message and "Nothing to commit" in message:
                print(f"   ✓ {label} already up to date in Hub")
            else:
                print(f"   ⚠️  {label} error: {e}")
    
    _handle_push("V1", PROMPT_V1_NAME, PROMPT_V1, "V1 – Concise 2-4 sentence answers")
    _handle_push("V2", PROMPT_V2_NAME, PROMPT_V2, "V2 – Structured 3-5 sentence expert answers")

# ── Pull prompts from Hub ───────────────────────────────────────────────────
def pull_prompts_from_hub():
    """
    Download both prompt versions from LangSmith Prompt Hub.
    Falls back to local templates if Hub is unavailable.
    """
    prompts = {}
    
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"   ✓ Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception as e:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"   ℹ️  Using local fallback for V1")
    
    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"   ✓ Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception as e:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"   ℹ️  Using local fallback for V2")
    
    return prompts

# ── A/B Routing (Deterministic Hash) ────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Route a request to prompt V1 or V2 based on MD5 hash of request_id.
    DETERMINISTIC: same request_id always maps to same version.
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME

# ── Build vectorstore ───────────────────────────────────────────────────────
def build_vectorstore():
    """Load knowledge base and build FAISS vectorstore."""
    print("📚 Building FAISS vectorstore...")
    text = Path("data/knowledge_base.txt").read_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print(f"   ✓ Indexed {len(chunks)} chunks")
    return vectorstore

# ── Traced A/B query function ───────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2", "gemini"])
def ask_with_routing(chain_v1, chain_v2, question: str, request_id: str) -> dict:
    """
    Run RAG chain with A/B routing.
    Returns both the answer and the version used.
    """
    version_name = get_prompt_version(request_id)
    chain = chain_v1 if version_name == PROMPT_V1_NAME else chain_v2
    
    answer = chain.invoke(question)

    version_short = "v1" if version_name == PROMPT_V1_NAME else "v2"
    
    return {
        "question": question,
        "answer": answer,
        "version": version_short,
        "request_id": request_id
    }

# ── Main execution ──────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("Step 2 — Prompt Hub & A/B Routing with Gemini")
    print("="*70)
    
    # Push prompts to Hub
    push_prompts_to_hub()
    
    # Pull prompts from Hub (or use local fallback)
    print("\n📥 Pulling prompts from Hub...")
    prompts = pull_prompts_from_hub()
    
    # Build vectorstore and chains
    print("\n🔗 Building RAG chains...")
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain_v1 = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompts[PROMPT_V1_NAME]
        | llm
        | StrOutputParser()
    )
    
    chain_v2 = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompts[PROMPT_V2_NAME]
        | llm
        | StrOutputParser()
    )
    
    print("   ✓ Chains ready")
    
    # Run A/B routing on all 50 questions
    print(f"\n🚀 Running {len(QA_PAIRS)} questions with A/B routing...")
    results = []
    version_counts = {"v1": 0, "v2": 0}
    
    for i, qa_pair in enumerate(QA_PAIRS, 1):
        question = qa_pair["question"]
        request_id = f"req-{i:03d}"
        
        try:
            result = ask_with_routing(chain_v1, chain_v2, question, request_id)
            version_counts[result["version"]] += 1
            results.append(result)
            print(f"   [{i:2d}/{len(QA_PAIRS)}] ✓ {result['version'].upper()} | {question[:50]}...")
        except Exception as e:
            print(f"   [{i:2d}/{len(QA_PAIRS)}] ✗ Error: {str(e)[:40]}...")
            results.append({
                "question": question,
                "error": str(e),
                "request_id": request_id
            })
    
    # Summary
    print(f"\n✅ Completed {len(results)} questions with A/B routing")
    print(f"   V1 (Concise): {version_counts['v1']} questions")
    print(f"   V2 (Structured): {version_counts['v2']} questions")
    print(f"📊 LangSmith Prompt Hub: https://smith.langchain.com/hub")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    finally:
        cleanup_gemini_clients(llm, embeddings)
