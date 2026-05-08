"""
Step 3 — RAGAS Evaluation with Gemini
======================================
TASK:
  1. Run all 50 QA pairs through BOTH prompt versions, capturing answers + contexts
  2. Build EvaluationDataset with SingleTurnSample objects
  3. Evaluate with 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. Print a V1 vs V2 comparison table
  5. Save results to data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 for at least one prompt version
             + data/ragas_report.json file saved
"""

import os
import json
import warnings
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── Load environment ────────────────────────────────────────────────────────
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# ── Imports ─────────────────────────────────────────────────────────────────
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import GEMINI_MODEL, cleanup_gemini_clients, get_llm, get_embeddings
from qa_pairs import QA_PAIRS

# ── Initialize LLM and Embeddings ───────────────────────────────────────────
llm = get_llm()
embeddings = get_embeddings()

# ── Define prompts ──────────────────────────────────────────────────────────
SYSTEM_V1 = """You are a helpful AI assistant. Answer using ONLY the provided context. Keep your answer concise (2-4 sentences).

Context:
{context}"""

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "{question}"),
])

SYSTEM_V2 = """You are an expert AI tutor. Provide a structured, accurate answer (3-5 sentences).

Context:
{context}"""

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "{question}"),
])

# ── Build vectorstore ───────────────────────────────────────────────────────
def build_vectorstore():
    """Load knowledge base and build FAISS vectorstore."""
    text = Path("data/knowledge_base.txt").read_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# ── Build RAG chains ────────────────────────────────────────────────────────
def build_rag_chains(vectorstore):
    """Build RAG chains for both prompt versions."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain_v1 = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT_V1
        | llm
        | StrOutputParser()
    )
    
    chain_v2 = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT_V2
        | llm
        | StrOutputParser()
    )
    
    return chain_v1, chain_v2, retriever

# ── Generate answers and contexts ───────────────────────────────────────────
def generate_samples(vectorstore, chain_v1, chain_v2):
    """Generate answers and contexts for all QA pairs using both prompt versions."""
    samples_v1 = []
    samples_v2 = []
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    print(f"🔄 Generating answers for all {len(QA_PAIRS)} questions...")
    
    for i, qa_pair in enumerate(QA_PAIRS, 1):
        question = qa_pair["question"]
        reference = qa_pair["reference"]
        
        # Retrieve context using vectorstore similarity search for compatibility
        try:
            retrieved_docs = vectorstore.similarity_search(question, k=3)
        except Exception:
            # fallback to retriever methods
            if hasattr(retriever, "get_relevant_documents"):
                retrieved_docs = retriever.get_relevant_documents(question)
            elif hasattr(retriever, "_get_relevant_documents"):
                try:
                    retrieved_docs = retriever._get_relevant_documents(question, run_manager=None)
                except Exception:
                    retrieved_docs = []
            else:
                retrieved_docs = []
        contexts = [doc.page_content for doc in retrieved_docs]
        
        # Generate answers with both chains
        try:
            answer_v1 = chain_v1.invoke(question)
            samples_v1.append(SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer_v1,
                reference=reference
            ))
        except Exception as e:
            print(f"   Error V1 for Q{i}: {str(e)[:40]}")
        
        try:
            answer_v2 = chain_v2.invoke(question)
            samples_v2.append(SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer_v2,
                reference=reference
            ))
        except Exception as e:
            print(f"   Error V2 for Q{i}: {str(e)[:40]}")
        
        if i % 10 == 0:
            print(f"   [{i:2d}/{len(QA_PAIRS)}] Generated samples")
    
    return samples_v1, samples_v2

# ── Run RAGAS evaluation ────────────────────────────────────────────────────
def run_evaluation(samples_v1, samples_v2):
    """Run RAGAS evaluation on both prompt versions."""
    print(f"\n📊 Running RAGAS evaluation (this may take 10-15 minutes)...")
    
    dataset_v1 = EvaluationDataset(samples=samples_v1)
    dataset_v2 = EvaluationDataset(samples=samples_v2)
    
    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]
    
    print("   ⏳ Evaluating V1 (concise)...")
    results_v1 = evaluate(
        dataset_v1,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )
    
    print("   ⏳ Evaluating V2 (structured)...")
    results_v2 = evaluate(
        dataset_v2,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )
    
    return results_v1, results_v2

# ── Compute averages ────────────────────────────────────────────────────────
def compute_averages(results):
    """Compute average scores for each metric."""
    averages = {}
    for metric_name in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        scores = None
        if hasattr(results, "_scores_dict") and metric_name in results._scores_dict:
            scores = results._scores_dict[metric_name]
        elif hasattr(results, "to_pandas"):
            try:
                frame = results.to_pandas()
                if metric_name in frame.columns:
                    scores = frame[metric_name].tolist()
            except Exception:
                scores = None

        if scores is not None:
            if isinstance(scores, list):
                clean_scores = [score for score in scores if score is not None]
                if clean_scores:
                    averages[metric_name] = float(np.mean(clean_scores))
            else:
                averages[metric_name] = float(scores)
    return averages

# ── Format and save results ─────────────────────────────────────────────────
def save_results(results_v1, results_v2):
    """Save evaluation results to JSON."""
    avg_v1 = compute_averages(results_v1)
    avg_v2 = compute_averages(results_v2)
    
    report = {
        "timestamp": str(Path("data/ragas_report.json").parent),
        "model": GEMINI_MODEL,
        "v1_prompt": "Concise (2-4 sentences)",
        "v2_prompt": "Structured (3-5 sentences)",
        "samples_count": 50,
        "v1_metrics": avg_v1,
        "v2_metrics": avg_v2,
        "comparison": {
            "v1_better": sum(1 for m in avg_v1 if avg_v1[m] > avg_v2.get(m, 0)),
            "v2_better": sum(1 for m in avg_v2 if avg_v2[m] > avg_v1.get(m, 0))
        }
    }
    
    # Save to file
    output_path = Path("data/ragas_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report

# ── Main execution ──────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("Step 3 — RAGAS Evaluation with Gemini")
    print("="*70)
    
    # Build vectorstore and chains
    print("\n🔧 Building RAG infrastructure...")
    vectorstore = build_vectorstore()
    chain_v1, chain_v2, retriever = build_rag_chains(vectorstore)
    print("   ✓ Ready")
    
    # Generate samples
    samples_v1, samples_v2 = generate_samples(vectorstore, chain_v1, chain_v2)
    print(f"   ✓ Generated {len(samples_v1)} samples for V1")
    print(f"   ✓ Generated {len(samples_v2)} samples for V2")
    
    # Run evaluation
    results_v1, results_v2 = run_evaluation(samples_v1, samples_v2)
    
    # Compute and display averages
    avg_v1 = compute_averages(results_v1)
    avg_v2 = compute_averages(results_v2)
    
    print("\n📈 RAGAS Evaluation Results")
    print("-" * 70)
    print(f"{'Metric':<25} {'V1 (Concise)':<20} {'V2 (Structured)':<20}")
    print("-" * 70)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        v1_score = avg_v1.get(metric, 0)
        v2_score = avg_v2.get(metric, 0)
        print(f"{metric:<25} {v1_score:<20.4f} {v2_score:<20.4f}")
    print("-" * 70)
    
    # Save results
    report = save_results(results_v1, results_v2)
    
    print(f"\n✅ Results saved to data/ragas_report.json")
    print(f"   V1 Faithfulness: {avg_v1.get('faithfulness', 0):.4f}")
    print(f"   V2 Faithfulness: {avg_v2.get('faithfulness', 0):.4f}")
    
    if avg_v1.get('faithfulness', 0) >= 0.8 or avg_v2.get('faithfulness', 0) >= 0.8:
        print("   ✓ Faithfulness requirement met!")
    else:
        print("   ⚠️  Consider refining prompts to improve faithfulness")
    
    return report

if __name__ == "__main__":
    try:
        report = main()
    finally:
        cleanup_gemini_clients(llm, embeddings)
