import os
import time
import json
import statistics
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag_pipeline import RAGConfig, load_vectorstore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langsmith_evaluation import (
    upload_dataset, correctness_evaluator, relevance_evaluator,
    hallucination_evaluator, conciseness_evaluator
)
from langsmith.evaluation import evaluate

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")

PROMPT_TEMPLATE = """You are a software-engineering teaching assistant.
Use ONLY the context below to answer concisely and accurately.
If the answer is not in the context, say "I don't know based on the provided materials."

Context:
{context}

Question: {question}

Answer:"""

def build_gemini_chain(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm       = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",   # free-tier model
        temperature=0,
        convert_system_message_to_human=True,
    )
    prompt    = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    from langchain_core.runnables import RunnablePassthrough
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def make_gemini_runnable(cfg: RAGConfig):
    vs = load_vectorstore(cfg)          # reuse embeddings from best experiment
    chain, retriever = build_gemini_chain(vs)

    def predict(inputs: dict) -> dict:
        q        = inputs["question"]
        t0       = time.time()
        answer   = chain.invoke(q)
        latency  = round(time.time() - t0, 3)
        sources  = retriever.get_relevant_documents(q)
        context  = "\n\n".join(d.page_content for d in sources)
        return {"answer": answer, "context": context,
                "question": q, "latency_s": latency}

    return predict


def benchmark_latency(predict_fn, questions: list[str], label: str) -> dict:
    latencies = []
    for q in questions:
        t0 = time.time()
        predict_fn({"question": q})
        latencies.append(time.time() - t0)
    stats = {
        "label":  label,
        "mean_s": round(statistics.mean(latencies), 3),
        "p50_s":  round(statistics.median(latencies), 3),
        "p95_s":  round(sorted(latencies)[int(0.95*len(latencies))], 3),
    }
    print(f"[{label}] mean={stats['mean_s']}s  p50={stats['p50_s']}s  p95={stats['p95_s']}s")
    return stats


INPUT_PRICE_PER_TOKEN  = 0.075 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 0.300 / 1_000_000
AVG_INPUT_TOKENS       = 800   # estimated per query (context + question)
AVG_OUTPUT_TOKENS      = 150

def estimate_cost(n_queries: int) -> float:
    cost = n_queries * (
        AVG_INPUT_TOKENS  * INPUT_PRICE_PER_TOKEN +
        AVG_OUTPUT_TOKENS * OUTPUT_PRICE_PER_TOKEN
    )
    return round(cost, 6)


def run_comparison():
    best_cfg = RAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        k=4,
        chroma_dir="./hw_1/chroma_exp_4",          # <- set to your best experiment dir
        collection_name="./hw_1/se_exp_4",
        experiment_name="gemini_comparison",
    )

    dataset_name    = upload_dataset()
    gemini_predict  = make_gemini_runnable(best_cfg)

    print("\n── Evaluating Gemini ────────────────────────────────────")
    gemini_results = evaluate(
        gemini_predict,
        data=dataset_name,
        evaluators=[
            correctness_evaluator, relevance_evaluator,
            hallucination_evaluator, conciseness_evaluator,
        ],
        experiment_prefix="gemini-flash-comparison",
        metadata={"model": "gemini-1.5-flash", **best_cfg.to_dict()},
    )

    scores = {"correctness": [], "relevance": [], "faithfulness": [], "conciseness": []}
    for r in gemini_results._results:
        for fb in r.get("evaluation_results", {}).get("results", []):
            scores[fb.key].append(fb.score)

    print("\n── COMPARISON SUMMARY ───────────────────────────────────")
    print(f"{'Metric':<22} {'Ollama (llama3.2:3b)':>18} {'Gemini 1.5 Flash':>18}")
    print("─" * 60)
    baseline = {"correctness": 0.62, "relevance": 0.71,
                "faithfulness": 0.68, "conciseness": 0.74}
    for m in ["correctness", "relevance", "faithfulness", "conciseness"]:
        gem = sum(scores[m]) / len(scores[m]) if scores[m] else 0
        oll = baseline.get(m, 0)
        print(f"  {m:<20} {oll:>18.3f} {gem:>18.3f}")

    n = 20  # dataset size

if __name__ == "__main__":
    run_comparison()
