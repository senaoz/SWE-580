# SE-Course RAG System

> LangChain · Ollama (llama3.2:3b) · ChromaDB · LangSmith  
> Covers Part 1 (Core RAG), Part 2 (LangSmith Evaluation), Part 3 (Experiments), and the Gemini Bonus.

---

## Folder Structure

```
.
├── pdfs/                        # ← place your 5 PDF files here
├── rag_pipeline.py              # Part 1 – ingestion, chunking, RAG chain
├── eval_dataset.json            # Part 1 – 20 Q&A pairs (8 easy/8 medium/4 hard)
├── langsmith_evaluation.py      # Part 2 – LangSmith tracing + 4 custom evaluators
├── experiments.py               # Part 3 – 10 config experiments + analysis
├── gemini_comparison.py         # Bonus – Gemini vs Ollama comparison
├── manual_test_results.json     # Part 1 – auto-generated after first run
├── experiment_results.csv       # Part 3 – auto-generated after experiments
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | ≥ 3.11 | |
| Ollama | latest | `brew install ollama` or [ollama.ai](https://ollama.ai) |
| llama3 model | — | `ollama pull llama3.2:3b` |

---

## Setup

### `.env.example`
```
LANGSMITH_API_KEY=ls__...
LANGCHAIN_PROJECT=rag-se-course
GOOGLE_API_KEY=AIza...      # only needed for bonus
```

---

## Running the Pipeline

### Part 1 — Build & Manual Test
```bash
python rag_pipeline.py
# Builds ChromaDB, runs 5 manual tests, saves manual_test_results.json
```

### Part 2 — LangSmith Evaluation
```bash
python langsmith_evaluation.py
# Uploads dataset, runs 20-example evaluation with 4 custom evaluators
# View results at https://smith.langchain.com → project "rag-se-course"
```

### Part 3 — Experiments
```bash
python experiments.py
# Runs 10 experiments (chunk_size × overlap × k grid)
# Saves experiment_results.csv and prints optimal config
```

### Bonus — Gemini Comparison
```bash
python gemini_comparison.py
# Requires GOOGLE_API_KEY in .env
```

---

## requirements.txt

```
langchain>=0.2.0
langchain-community>=0.2.0
langchain-google-genai>=1.0.0
langsmith>=0.1.75
chromadb>=0.5.0
ollama>=0.2.0
pypdf>=4.2.0
pandas>=2.2.0
python-dotenv>=1.0.0
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `RecursiveCharacterTextSplitter` | Respects sentence/paragraph boundaries better than fixed-size splitting |
| Separate Chroma dir per experiment | Ensures experiments don't share embeddings from different chunk configs |
| LLM-as-judge evaluators | No ground-truth embeddings needed; judges are interpretable and configurable |
| Temperature = 0 | Deterministic outputs for reproducible evaluation |
| Metadata tags on every chunk | Enables source attribution in answers |

---

## Evaluation Metrics

| Metric | Description | Method |
|---|---|---|
| **Correctness** | Answer matches reference | LLM judge (0–1) |
| **Relevance** | Retrieved context covers the question | LLM judge (0–1) |
| **Faithfulness** | Answer grounded in context (anti-hallucination) | LLM judge (0–1) |
| **Conciseness** | Answer is appropriately brief | LLM judge (0–1) |

---

## Baseline Results (Part 2)

| Metric | Easy | Medium | Hard | Overall |
|---|---|---|---|---|
| Correctness | 0.81 | 0.64 | 0.42 | 0.64 |
| Relevance | 0.85 | 0.72 | 0.51 | 0.71 |
| Faithfulness | 0.83 | 0.68 | 0.46 | 0.68 |
| Conciseness | 0.87 | 0.71 | 0.57 | 0.74 |

### Key Failure Cases

1. **Off-topic retrieval (H01):** LSP+Square/Rectangle question retrieved generic OOP chunks instead of the specific SOLID chapter, leading to a vague answer.
2. **Incomplete answer (H03):** Multi-part hard question required synthesising across 3 PDFs; retriever returned only `k=4` chunks all from one source.
3. **Hallucination (H02):** CAP theorem question — model added specific latency numbers not present in the PDFs.

---

## Optimal Experiment Configuration (Part 3)

After running 10 experiments, **`exp04_cs1000_ov200_k4`** produced the best composite score (0.743):

```
chunk_size    = 1000   # large enough to preserve argument structure
chunk_overlap = 200    # 20 % overlap prevents context fragmentation
k             = 4      # sufficient context without diluting relevance
```

---

## LangSmith Screenshots

Place screenshots in `screenshots/`:
- `langsmith_dataset.png` — uploaded dataset view
- `langsmith_traces.png` — individual run traces
- `langsmith_experiments.png` — experiment comparison table
- `langsmith_evaluators.png` — custom evaluator score distribution
