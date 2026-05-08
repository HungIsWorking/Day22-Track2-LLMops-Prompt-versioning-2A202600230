"""
Step 1 — LangSmith-instrumented RAG Pipeline with Gemini
=========================================================
TASK:
  1. Load your dataset, split into chunks, index with FAISS
  2. Build a RAG chain: retriever → prompt → LLM → output parser
  3. Decorate the query function with @traceable so every call is traced
  4. Run all 50 questions → generates ≥ 50 LangSmith traces

DELIVERABLE: Open https://smith.langchain.com and confirm traces appear.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment ────────────────────────────────────────────────────────
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "day22-rag-lab")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ── Imports ─────────────────────────────────────────────────────────────────
from config import cleanup_gemini_clients, get_llm, get_embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from config import GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_EMBEDDING_MODEL
from qa_pairs import QA_PAIRS

# ── Initialize LLM and Embeddings ───────────────────────────────────────────
llm = get_llm()
embeddings = get_embeddings()

# ── Build FAISS vector store ────────────────────────────────────────────────
def build_vectorstore():
    """
    Load the knowledge base, split into chunks, embed and index with FAISS.
    """
    print("📚 Building FAISS vectorstore...")
    
    # Read knowledge base
    text = Path("data/knowledge_base.txt").read_text()
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"   ✓ Split into {len(chunks)} chunks")
    
    # Build FAISS vectorstore
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print(f"   ✓ Indexed {len(chunks)} chunks")
    return vectorstore

# ── RAG Prompt Template ─────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant specialized in machine learning and NLP topics. 
Your task is to answer questions using ONLY the provided context. 
If the context does not contain sufficient information to answer the question, say: "I don't have enough information to answer this question."

Context:
{context}

Please provide a clear, concise answer based on the context above."""),
    ("human", "{question}"),
])

# ── Build RAG chain ─────────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Build a LangChain RAG chain using LCEL (pipe operator).
    """
    print("🔗 Building RAG chain...")
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Helper to format retrieved docs
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build LCEL chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    
    print("   ✓ RAG chain ready")
    return chain, retriever

# ── Traced query function ───────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1", "gemini"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question.
    The @traceable decorator sends input/output/latency to LangSmith.
    """
    return chain.invoke(question)

# ── Main execution ──────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("Step 1 — LangSmith RAG Pipeline with Gemini")
    print("="*70)
    
    # Build vectorstore and chain
    vectorstore = build_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)
    
    # Run all 50 questions
    print(f"\n🚀 Running {len(QA_PAIRS)} questions through RAG pipeline...")
    results = []
    
    for i, qa_pair in enumerate(QA_PAIRS, 1):
        question = qa_pair["question"]
        try:
            answer = ask(chain, question)
            results.append({
                "question": question,
                "answer": answer,
                "success": True
            })
            print(f"   [{i:2d}/{len(QA_PAIRS)}] ✓ {question[:60]}...")
        except Exception as e:
            print(f"   [{i:2d}/{len(QA_PAIRS)}] ✗ Error: {str(e)[:50]}...")
            results.append({
                "question": question,
                "error": str(e),
                "success": False
            })
    
    # Summary
    successful = sum(1 for r in results if r.get("success"))
    print(f"\n✅ Completed {successful}/{len(QA_PAIRS)} questions successfully")
    print(f"📊 LangSmith traces available at: https://smith.langchain.com")
    print(f"   Project: {os.getenv('LANGCHAIN_PROJECT', 'day22-rag-lab')}")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    finally:
        cleanup_gemini_clients(llm, embeddings)
