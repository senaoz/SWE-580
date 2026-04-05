# HW1 — RAG Q&A System

> LangChain · Ollama (llama3.2:3b) · ChromaDB · LangSmith

Covers Part 1 (Core RAG), Part 2 (LangSmith Evaluation), Part 3 (Experiments)

---

## Directory Structure

```
hw_1/
├── pdfs/                        # 5 software testing PDFs (not tracked in git)
├── rag_pipeline.py              # Part 1 – ingestion, chunking, RAG chain
├── langsmith_evaluation.py      # Part 2 – LangSmith tracing + 4 custom evaluators
├── experiments.py               # Part 3 – 10 config experiments + analysis
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
python hw_1/rag_pipeline.py
```
Builds ChromaDB from PDFs, runs 5 manual Q&A tests, saves `manual_test_results.json`.

### Part 2 — LangSmith Evaluation
```bash
python hw_1/langsmith_evaluation.py
```
Uploads the 20-question dataset to LangSmith, runs evaluation with 4 custom evaluators (Correctness, Relevance, Faithfulness, Conciseness), and prints per-metric averages.

### Part 3 — Experiments
```bash
python hw_1/experiments.py
```
Runs 10 experiments over a grid of chunk sizes, overlaps, and k values. Saves `experiment_results.csv` and prints the optimal configuration.

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

<img width="1800" height="1042" alt="Screenshot 2026-04-04 at 23 12 47" src="https://github.com/user-attachments/assets/b08485b8-0b0c-4321-9c2c-6e4a8403558b" />
<img width="1800" height="1042" alt="Screenshot 2026-04-04 at 23 10 29" src="https://github.com/user-attachments/assets/5c0bfa41-e77b-40f5-ba9b-1f2c5fbfe6f5" />
<img width="1800" height="1042" alt="Screenshot 2026-04-04 at 23 10 08" src="https://github.com/user-attachments/assets/35e588a0-d46d-4e86-b717-472448e89e6f" />
<img width="1800" height="1042" alt="Screenshot 2026-04-04 at 23 06 57" src="https://github.com/user-attachments/assets/3bcff8b2-3168-40c2-9430-540908395b6f" />
<img width="1800" height="1042" alt="Screenshot 2026-04-04 at 23 04 38" src="https://github.com/user-attachments/assets/61700509-910a-4d9f-a80b-8ec5cdd69388" />
<img width="1800" height="1042" alt="Screenshot 2026-04-04 at 23 03 41" src="https://github.com/user-attachments/assets/d3a1434c-1326-4f9a-9446-00678623d1be" />

## Notebook Outputs

```
_1/experiments.py

════════════════════════════════════════════════════════════
EXPERIMENT: exp01_cs500_ov50_k2
  chunk_size=500  overlap=50  k=2
  ✔ A_Survey_on_Unit_Testing_Practices_and_Problems.pdf  →  146 chunks
  ✔ A_Survey_on_What_Developers_Think_About_Testing.pdf  →  144 chunks
  ✔ software-testing-introduction.pdf  →  143 chunks
  ✔ test‑driven-development-with-mutation-testing.pdf  →  255 chunks
  ✔ web-application-testing.pdf  →  275 chunks

Total chunks: 963
Vector store saved → ./hw_1/chroma_exp_1/

View the evaluation results for experiment: 'exp01_cs500_ov50_k2-00ad6491' at:
https://smith.langchain.com/o/75161731-1558-47f2-a9da-aa9ca13488fa/datasets/68caf303-ad2c-4972-b16a-34168f01b439/compare?selectedSessions=63219575-ee65-475c-a0e9-01681b8ccdf7


20it [01:49,  5.50s/it]

───────────────────────────────────────────────────────
Experiment : exp01_cs500_ov50_k2
  correctness          avg = 0.430  (n=20)
  relevance            avg = 0.325  (n=20)
  faithfulness         avg = 0.475  (n=20)
  conciseness          avg = 0.490  (n=20)
───────────────────────────────────────────────────────
  composite score: 0.4300

════════════════════════════════════════════════════════════
EXPERIMENT: exp02_cs500_ov50_k4
  chunk_size=500  overlap=50  k=4
  ✔ A_Survey_on_Unit_Testing_Practices_and_Problems.pdf  →  146 chunks
  ✔ A_Survey_on_What_Developers_Think_About_Testing.pdf  →  144 chunks
  ✔ software-testing-introduction.pdf  →  143 chunks
  ✔ test‑driven-development-with-mutation-testing.pdf  →  255 chunks
  ✔ web-application-testing.pdf  →  275 chunks

Total chunks: 963
Vector store saved → ./hw_1/chroma_exp_2/

View the evaluation results for experiment: 'exp02_cs500_ov50_k4-dd59c93e' at:
https://smith.langchain.com/o/75161731-1558-47f2-a9da-aa9ca13488fa/datasets/68caf303-ad2c-4972-b16a-34168f01b439/compare?selectedSessions=81adb40d-e146-44d5-ad3c-b1ef67ea5df4


20it [02:33,  7.69s/it]

───────────────────────────────────────────────────────
Experiment : exp02_cs500_ov50_k4
  correctness          avg = 0.385  (n=20)
  relevance            avg = 0.380  (n=20)
  faithfulness         avg = 0.465  (n=20)
  conciseness          avg = 0.610  (n=20)
───────────────────────────────────────────────────────
  composite score: 0.4600

════════════════════════════════════════════════════════════
EXPERIMENT: exp03_cs500_ov50_k6
  chunk_size=500  overlap=50  k=6
  ✔ A_Survey_on_Unit_Testing_Practices_and_Problems.pdf  →  146 chunks
  ✔ A_Survey_on_What_Developers_Think_About_Testing.pdf  →  144 chunks
  ✔ software-testing-introduction.pdf  →  143 chunks
  ✔ test‑driven-development-with-mutation-testing.pdf  →  255 chunks
  ✔ web-application-testing.pdf  →  275 chunks

Total chunks: 963
Vector store saved → ./hw_1/chroma_exp_3/

View the evaluation results for experiment: 'exp03_cs500_ov50_k6-64c24d0f' at:
https://smith.langchain.com/o/75161731-1558-47f2-a9da-aa9ca13488fa/datasets/68caf303-ad2c-4972-b16a-34168f01b439/compare?selectedSessions=798a843b-fc47-4e56-83a8-a5f1d88a9ba9


20it [03:10,  9.51s/it]

───────────────────────────────────────────────────────
Experiment : exp03_cs500_ov50_k6
  correctness          avg = 0.455  (n=20)
  relevance            avg = 0.430  (n=20)
  faithfulness         avg = 0.545  (n=20)
  conciseness          avg = 0.570  (n=20)
───────────────────────────────────────────────────────
  composite score: 0.5000

════════════════════════════════════════════════════════════
EXPERIMENT: exp04_cs500_ov200_k2
  chunk_size=500  overlap=200  k=2
  ✔ A_Survey_on_Unit_Testing_Practices_and_Problems.pdf  →  220 chunks
  ✔ A_Survey_on_What_Developers_Think_About_Testing.pdf  →  219 chunks
  ✔ software-testing-introduction.pdf  →  196 chunks
  ✔ test‑driven-development-with-mutation-testing.pdf  →  365 chunks
  ✔ web-application-testing.pdf  →  399 chunks

Total chunks: 1399
Vector store saved → ./hw_1/chroma_exp_4/

View the evaluation results for experiment: 'exp04_cs500_ov200_k2-f0c77359' at:
https://smith.langchain.com/o/75161731-1558-47f2-a9da-aa9ca13488fa/datasets/68caf303-ad2c-4972-b16a-34168f01b439/compare?selectedSessions=5b8aae26-5a69-4814-9155-394815ccd600


20it [01:42,  5.10s/it]

───────────────────────────────────────────────────────
Experiment : exp04_cs500_ov200_k2
  correctness          avg = 0.390  (n=20)
  relevance            avg = 0.390  (n=20)
  faithfulness         avg = 0.550  (n=20)
  conciseness          avg = 0.615  (n=20)
───────────────────────────────────────────────────────
  composite score: 0.4863

════════════════════════════════════════════════════════════
EXPERIMENT: exp05_cs500_ov200_k4
  chunk_size=500  overlap=200  k=4
  ✔ A_Survey_on_Unit_Testing_Practices_and_Problems.pdf  →  220 chunks
  ✔ A_Survey_on_What_Developers_Think_About_Testing.pdf  →  219 chunks
  ✔ software-testing-introduction.pdf  →  196 chunks
  ✔ test‑driven-development-with-mutation-testing.pdf  →  365 chunks
  ✔ web-application-testing.pdf  →  399 chunks

Total chunks: 1399
Vector store saved → ./hw_1/chroma_exp_5/

View the evaluation results for experiment: 'exp05_cs500_ov200_k4-473135ff' at:
https://smith.langchain.com/o/75161731-1558-47f2-a9da-aa9ca13488fa/datasets/68caf303-ad2c-4972-b16a-34168f01b439/compare?selectedSessions=89698a06-f7cf-4b47-8fc7-dfb72bbd6136


20it [02:21,  7.06s/it]

───────────────────────────────────────────────────────
Experiment : exp05_cs500_ov200_k4
  correctness          avg = 0.415  (n=20)
  relevance            avg = 0.450  (n=20)
  faithfulness         avg = 0.555  (n=20)
  conciseness          avg = 0.515  (n=20)
───────────────────────────────────────────────────────
  composite score: 0.4838

════════════════════════════════════════════════════════════
EXPERIMENT: exp06_cs500_ov200_k6
  chunk_size=500  overlap=200  k=6
  ✔ A_Survey_on_Unit_Testing_Practices_and_Problems.pdf  →  220 chunks
  ✔ A_Survey_on_What_Developers_Think_About_Testing.pdf  →  219 chunks
  ✔ software-testing-introduction.pdf  →  196 chunks
  ✔ test‑driven-development-with-mutation-testing.pdf  →  365 chunks
  ✔ web-application-testing.pdf  →  399 chunks

Total chunks: 1399
Vector store saved → ./hw_1/chroma_exp_6/

View the evaluation results for experiment: 'exp06_cs500_ov200_k6-8aca1ad5' at:
https://smith.langchain.com/o/75161731-1558-47f2-a9da-aa9ca13488fa/datasets/68caf303-ad2c-4972-b16a-34168f01b439/compare?selectedSessions=aa5b614f-53c2-4127-be51-2a91b281da40


20it [02:51,  8.58s/it]

───────────────────────────────────────────────────────
Experiment : exp06_cs500_ov200_k6
  correctness          avg = 0.510  (n=20)
  relevance            avg = 0.515  (n=20)
  faithfulness         avg = 0.555  (n=20)
  conciseness          avg = 0.635  (n=20)
───────────────────────────────────────────────────────
  composite score: 0.5537

════════════════════════════════════════════════════════════
EXPERIMENT: exp07_cs1000_ov50_k2
  chunk_size=1000  overlap=50  k=2
  ✔ A_Survey_on_Unit_Testing_Practices_and_Problems.pdf  →  73 chunks
  ✔ A_Survey_on_What_Developers_Think_About_Testing.pdf  →  73 chunks
  ✔ software-testing-introduction.pdf  →  74 chunks
  ✔ test‑driven-development-with-mutation-testing.pdf  →  132 chunks
  ✔ web-application-testing.pdf  →  137 chunks

Total chunks: 489
Vector store saved → ./hw_1/chroma_exp_7/

View the evaluation results for experiment: 'exp07_cs1000_ov50_k2-9a269f3c' at:
https://smith.langchain.com/o/75161731-1558-47f2-a9da-aa9ca13488fa/datasets/68caf303-ad2c-4972-b16a-34168f01b439/compare?selectedSessions=5c32b434-db82-419c-ad82-ef27668ec7c3


20it [02:19,  6.98s/it]

───────────────────────────────────────────────────────
Experiment : exp07_cs1000_ov50_k2
  correctness          avg = 0.435  (n=20)
  relevance            avg = 0.430  (n=20)
  faithfulness         avg = 0.550  (n=20)
  conciseness          avg = 0.480  (n=20)
───────────────────────────────────────────────────────
  composite score: 0.4738

════════════════════════════════════════════════════════════
EXPERIMENT: exp08_cs1000_ov50_k4
  chunk_size=1000  overlap=50  k=4
  ✔ A_Survey_on_Unit_Testing_Practices_and_Problems.pdf  →  73 chunks
  ✔ A_Survey_on_What_Developers_Think_About_Testing.pdf  →  73 chunks
  ✔ software-testing-introduction.pdf  →  74 chunks
  ✔ test‑driven-development-with-mutation-testing.pdf  →  132 chunks
  ✔ web-application-testing.pdf  →  137 chunks

Total chunks: 489
Vector store saved → ./hw_1/chroma_exp_8/

View the evaluation results for experiment: 'exp08_cs1000_ov50_k4-e91fe892' at:
https://smith.langchain.com/o/75161731-1558-47f2-a9da-aa9ca13488fa/datasets/68caf303-ad2c-4972-b16a-34168f01b439/compare?selectedSessions=bea37651-e518-43da-9799-3bd481eefff4


20it [03:20, 10.03s/it]

───────────────────────────────────────────────────────
Experiment : exp08_cs1000_ov50_k4
  correctness          avg = 0.445  (n=20)
  relevance            avg = 0.465  (n=20)
  faithfulness         avg = 0.550  (n=20)
  conciseness          avg = 0.645  (n=20)
───────────────────────────────────────────────────────
  composite score: 0.5262

════════════════════════════════════════════════════════════
EXPERIMENT: exp09_cs1000_ov50_k6
  chunk_size=1000  overlap=50  k=6
  ✔ A_Survey_on_Unit_Testing_Practices_and_Problems.pdf  →  73 chunks
  ✔ A_Survey_on_What_Developers_Think_About_Testing.pdf  →  73 chunks
  ✔ software-testing-introduction.pdf  →  74 chunks
  ✔ test‑driven-development-with-mutation-testing.pdf  →  132 chunks
  ✔ web-application-testing.pdf  →  137 chunks

Total chunks: 489
Vector store saved → ./hw_1/chroma_exp_9/

View the evaluation results for experiment: 'exp09_cs1000_ov50_k6-1f0d4f3f' at:
https://smith.langchain.com/o/75161731-1558-47f2-a9da-aa9ca13488fa/datasets/68caf303-ad2c-4972-b16a-34168f01b439/compare?selectedSessions=59580fa1-f1ab-486e-b89e-21f23c7f821e


20it [04:38, 13.91s/it]

───────────────────────────────────────────────────────
Experiment : exp09_cs1000_ov50_k6
  correctness          avg = 0.385  (n=20)
  relevance            avg = 0.570  (n=20)
  faithfulness         avg = 0.485  (n=20)
  conciseness          avg = 0.670  (n=20)
───────────────────────────────────────────────────────
  composite score: 0.5275

════════════════════════════════════════════════════════════
EXPERIMENT: exp10_cs1000_ov200_k2
  chunk_size=1000  overlap=200  k=2
  ✔ A_Survey_on_Unit_Testing_Practices_and_Problems.pdf  →  85 chunks
  ✔ A_Survey_on_What_Developers_Think_About_Testing.pdf  →  82 chunks
  ✔ software-testing-introduction.pdf  →  83 chunks
  ✔ test‑driven-development-with-mutation-testing.pdf  →  147 chunks
  ✔ web-application-testing.pdf  →  155 chunks

Total chunks: 552
Vector store saved → ./hw_1/chroma_exp_10/

View the evaluation results for experiment: 'exp10_cs1000_ov200_k2-4b98c12a' at:
https://smith.langchain.com/o/75161731-1558-47f2-a9da-aa9ca13488fa/datasets/68caf303-ad2c-4972-b16a-34168f01b439/compare?selectedSessions=d5e885b6-6312-4528-b4b9-1c2bfe8a054f


20it [02:17,  6.87s/it]

───────────────────────────────────────────────────────
Experiment : exp10_cs1000_ov200_k2
  correctness          avg = 0.390  (n=20)
  relevance            avg = 0.440  (n=20)
  faithfulness         avg = 0.520  (n=20)
  conciseness          avg = 0.435  (n=20)
───────────────────────────────────────────────────────
  composite score: 0.4463

======================================================================
EXPERIMENT RESULTS (sorted by composite score)
======================================================================
           experiment  chunk_size  chunk_overlap  k  correctness  relevance  faithfulness  conciseness  avg_composite
 exp06_cs500_ov200_k6         500            200  6        0.510      0.515         0.555        0.635         0.5537
 exp09_cs1000_ov50_k6        1000             50  6        0.385      0.570         0.485        0.670         0.5275
 exp08_cs1000_ov50_k4        1000             50  4        0.445      0.465         0.550        0.645         0.5262
  exp03_cs500_ov50_k6         500             50  6        0.455      0.430         0.545        0.570         0.5000
 exp04_cs500_ov200_k2         500            200  2        0.390      0.390         0.550        0.615         0.4863
 exp05_cs500_ov200_k4         500            200  4        0.415      0.450         0.555        0.515         0.4838
 exp07_cs1000_ov50_k2        1000             50  2        0.435      0.430         0.550        0.480         0.4738
  exp02_cs500_ov50_k4         500             50  4        0.385      0.380         0.465        0.610         0.4600
exp10_cs1000_ov200_k2        1000            200  2        0.390      0.440         0.520        0.435         0.4463
  exp01_cs500_ov50_k2         500             50  2        0.430      0.325         0.475        0.490         0.4300

✔ Saved → experiment_results.csv

OPTIMAL CONFIG: exp06_cs500_ov200_k6
   chunk_size=500  overlap=200  k=6
   Correctness  = 0.5100
   Relevance    = 0.5150
   Faithfulness = 0.5550
   Conciseness  = 0.6350
   COMPOSITE    = 0.5537

── FAILURE CASE ANALYSIS ──────────────────────────────

  [off_topic_retrieval]  (1 cases)
    Q: What did the unit testing survey conclude about the role of automation in unit t…
    Expected snippet: The survey confirmed that unit testing is an important factor in software develo…
    Got: I don't know based on the provided materials.…

  [incomplete_answer]  (0 cases)

  [hallucination]  (0 cases)
```
