import os
import json
from dotenv import load_dotenv
load_dotenv()
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_ollama import OllamaLLM

from rag_pipeline import RAGConfig, load_vectorstore, build_rag_chain

os.environ["LANGCHAIN_TRACING_V2"]  = os.getenv("LANGSMITH_TRACING")
os.environ["LANGCHAIN_ENDPOINT"]    = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGSMITH_PROJECT")

client = Client()

DATASET_NAME = "RAG-HW1-Eval-v1"

def upload_dataset(path: str = "./hw_1/eval_dataset.json") -> str:
    """Create (or re-create if empty) LangSmith dataset from local JSON."""
    with open(path) as f:
        qa_pairs = json.load(f)

    existing = {d.name: d for d in client.list_datasets()}
    if DATASET_NAME in existing:
        dataset = existing[DATASET_NAME]
        example_count = sum(1 for _ in client.list_examples(dataset_id=dataset.id))
        if example_count == len(qa_pairs):
            print(f"Dataset '{DATASET_NAME}' already has {example_count} examples — skipping upload.")
            return DATASET_NAME
        # Dataset exists but is empty or stale — delete and recreate
        print(f"Dataset '{DATASET_NAME}' has {example_count} examples (expected {len(qa_pairs)}) — recreating.")
        client.delete_dataset(dataset_id=dataset.id)

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="20 Q&A pairs from software-engineering course PDFs",
    )
    client.create_examples(
        inputs=[{"question": q["question"], "difficulty": q["difficulty"]}
                for q in qa_pairs],
        outputs=[{"answer": q["expected"]} for q in qa_pairs],
        dataset_id=dataset.id,
    )
    print(f"✔ Uploaded {len(qa_pairs)} examples to '{DATASET_NAME}'")
    return DATASET_NAME

def make_rag_runnable(cfg: RAGConfig):
    """Returns a callable (input_dict → output_dict) compatible with evaluate()."""
    vs = load_vectorstore(cfg)
    chain, retriever = build_rag_chain(vs, cfg)

    def predict(inputs: dict) -> dict:
        question = inputs["question"]
        answer   = chain.invoke(question)
        sources  = retriever.invoke(question)
        context  = "\n\n".join(d.page_content for d in sources)
        return {"answer": answer, "context": context, "question": question}

    return predict


judge_llm = OllamaLLM(model="llama3.2:3b", temperature=0)


def _llm_score(prompt_text: str, score_key: str):
    """Helper: run a judge prompt and parse a 0–1 float score."""
    result = judge_llm.invoke(prompt_text).strip()
    for line in result.splitlines():
        if "score" in line.lower():
            try:
                return float(line.split(":")[-1].strip().split()[0])
            except ValueError:
                pass
    try:
        return float(result.split()[0])
    except (ValueError, IndexError):
        return 0.5


# ── Evaluator 1: Correctness ──────────────────

CORRECTNESS_PROMPT = """\
You are an expert grader.
Reference Answer: {reference}
Predicted Answer: {prediction}

Score the predicted answer for CORRECTNESS on a scale 0.0–1.0:
  1.0 = fully correct and complete
  0.5 = partially correct
  0.0 = wrong or missing key information

Respond ONLY with: Score: <float>
"""

def correctness_evaluator(run, example) -> dict:
    predicted  = run.outputs.get("answer", "")
    reference  = example.outputs.get("answer", "")
    prompt     = CORRECTNESS_PROMPT.format(reference=reference, prediction=predicted)
    score      = _llm_score(prompt, "correctness")
    return {"key": "correctness", "score": score}


# ── Evaluator 2: Relevance ────────────────────

RELEVANCE_PROMPT = """\
Question: {question}
Retrieved Context: {context}

Score how RELEVANT the context is for answering the question (0.0–1.0):
  1.0 = context fully covers the answer
  0.5 = context partially relevant
  0.0 = context is unrelated

Respond ONLY with: Score: <float>
"""

def relevance_evaluator(run, example) -> dict:
    question = run.outputs.get("question", example.inputs.get("question", ""))
    context  = run.outputs.get("context", "")
    prompt   = RELEVANCE_PROMPT.format(question=question, context=context)
    score    = _llm_score(prompt, "relevance")
    return {"key": "relevance", "score": score}


# ── Evaluator 3: Hallucination Detection ──────

HALLUCINATION_PROMPT = """\
Context: {context}
Predicted Answer: {prediction}

Does the predicted answer contain ANY claims NOT supported by the context?
Score FAITHFULNESS (inverse of hallucination) 0.0–1.0:
  1.0 = every claim is grounded in the context
  0.5 = some unsupported additions
  0.0 = largely fabricated or contradicts the context

Respond ONLY with: Score: <float>
"""

def hallucination_evaluator(run, example) -> dict:
    predicted = run.outputs.get("answer", "")
    context   = run.outputs.get("context", "")
    prompt    = HALLUCINATION_PROMPT.format(context=context, prediction=predicted)
    score     = _llm_score(prompt, "faithfulness")
    return {"key": "faithfulness", "score": score}


# ── Evaluator 4: Conciseness ─────────

CONCISENESS_PROMPT = """\
Question: {question}
Answer: {prediction}

Score CONCISENESS (0.0–1.0): Is the answer appropriately brief without missing key info?
  1.0 = tight and complete
  0.5 = some padding but acceptable
  0.0 = excessively verbose or repetitive

Respond ONLY with: Score: <float>
"""

def conciseness_evaluator(run, example) -> dict:
    question  = run.outputs.get("question", example.inputs.get("question", ""))
    predicted = run.outputs.get("answer", "")
    prompt    = CONCISENESS_PROMPT.format(question=question, prediction=predicted)
    score     = _llm_score(prompt, "conciseness")
    return {"key": "conciseness", "score": score}


def run_evaluation(cfg: RAGConfig, experiment_prefix: str = "baseline"):
    dataset_name = upload_dataset()
    predict      = make_rag_runnable(cfg)

    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=[
            correctness_evaluator,
            relevance_evaluator,
            hallucination_evaluator,
            conciseness_evaluator,
        ],
        experiment_prefix=experiment_prefix,
        metadata=cfg.to_dict(),        # stored in LangSmith run metadata
        num_repetitions=1,
    )

    print(f"\n{'─'*55}")
    print(f"Experiment : {experiment_prefix}")
    scores = {"correctness": [], "relevance": [], "faithfulness": [], "conciseness": []}
    for r in results._results:
        for fb in r.get("evaluation_results", {}).get("results", []):
            scores[fb.key].append(fb.score)

    for metric, vals in scores.items():
        avg = sum(vals) / len(vals) if vals else 0
        print(f"  {metric:<20} avg = {avg:.3f}  (n={len(vals)})")
    print(f"{'─'*55}")
    return results


if __name__ == "__main__":
    cfg = RAGConfig(experiment_name="baseline")
    run_evaluation(cfg, experiment_prefix="baseline")
