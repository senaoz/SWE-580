# HW1 — RAG System for Software Engineering Papers

> LangChain · Ollama (llama3.2:3b) · ChromaDB · LangSmith

Covers Part 1 (Core RAG), Part 2 (LangSmith Evaluation), Part 3 (Experiments), and the Gemini comparison.

---

## Directory Structure

```
hw_1/
├── pdfs/                          # 5 software testing PDFs (not tracked in git)
├── 1_rag_pipeline.py              # Part 1 – ingestion, chunking, RAG chain
├── 2_langsmith_evaluation.py      # Part 2 – LangSmith tracing + 4 custom evaluators
├── 3_experiments.py               # Part 3 – 10 config experiments + analysis
├── 4_gemini_comparison.py         # Gemini vs Ollama comparison
├── eval_dataset.json              # 20 Q&A pairs (8 easy / 8 medium / 4 hard)
├── manual_test_results.json       # auto-generated after Part 1
├── experiment_results.csv         # auto-generated after Part 3
├── requirements.txt
└── README.md
```

---

## Source Papers

| File | Title |
|------|-------|
| `A_Survey_on_Unit_Testing_Practices_and_Problems.pdf` | A Survey on Unit Testing Practices and Problems |
| `A_Survey_on_What_Developers_Think_About_Testing.pdf` | A Survey on What Developers Think About Testing |
| `software-testing-introduction.pdf` | An Introduction to Software Testing |
| `test-driven-development-with-mutation-testing.pdf` | Test-Driven Development with Mutation Testing |
| `web-application-testing.pdf` | Web Application Testing – Challenges and Opportunities |

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | ≥ 3.11 | |
| Ollama | latest | `brew install ollama` |
| llama3.2:3b model | — | `ollama pull llama3.2:3b` |
| LangSmith account | — | [smith.langchain.com](https://smith.langchain.com) |

---

## Setup

```bash
git clone <repo-url>
cd SWE-580/hw_1
pip install -r requirements.txt
```

Create a `.env` file in `hw_1/`:

```
LANGSMITH_API_KEY=ls__...
LANGCHAIN_PROJECT=rag-hw1
GOOGLE_API_KEY=AIza... 
```

Place the 5 PDF files in `hw_1/pdfs/`.

---

## Running the Pipeline

Run scripts in order from the repo root (`SWE-580/`):

### Part 1 — Build & Manual Test
```bash
python hw_1/1_rag_pipeline.py
```
Builds ChromaDB from PDFs, runs 5 manual Q&A tests, saves `manual_test_results.json`.

### Part 2 — LangSmith Evaluation
```bash
python hw_1/2_langsmith_evaluation.py
```
Uploads the 20-question dataset to LangSmith, runs evaluation with 4 custom evaluators (Correctness, Relevance, Faithfulness, Conciseness), and prints per-metric averages.

### Part 3 — Experiments
```bash
python hw_1/3_experiments.py
```
Runs 10 experiments over a grid of chunk sizes, overlaps, and k values. Saves `experiment_results.csv` and prints the optimal configuration.

### Gemini Comparison
```bash
python hw_1/4_gemini_comparison.py
```
Requires `GOOGLE_API_KEY`. Compares Gemini and Ollama responses on the same questions.

---

## Evaluation Metrics

| Metric | What it measures | Method |
|--------|-----------------|--------|
| **Correctness** | Answer matches the reference answer | LLM-as-judge (0–1) |
| **Relevance** | Retrieved context covers the question | LLM-as-judge (0–1) |
| **Faithfulness** | Answer is grounded in context (anti-hallucination) | LLM-as-judge (0–1) |
| **Conciseness** | Answer is appropriately brief | LLM-as-judge (0–1) |

The judge model is `llama3.2:3b` at `temperature=0` for deterministic scoring.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| `RecursiveCharacterTextSplitter` | Respects sentence/paragraph boundaries better than fixed-size splitting |
| Separate Chroma dir per experiment | Ensures experiments don't share embeddings from different configs |
| LLM-as-judge evaluators | No ground-truth embeddings needed; interpretable and configurable |
| `temperature=0` for LLM and judge | Deterministic outputs for reproducible evaluation |
| Metadata tag on every chunk | Enables source attribution in answers |

---

## LangSmith Screenshots

Screenshots from the LangSmith dashboard

### Dataset view

### Run traces

### Experiment comparison

### Evaluator scores
