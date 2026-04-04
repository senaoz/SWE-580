"""
rag_pipeline.py  ·  Part 1 — Core RAG System
Stack: LangChain · Ollama (llama3.2:3b) · ChromaDB
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ──────────────────────────────────────────────
# 1.  Configuration
# ──────────────────────────────────────────────

@dataclass
class RAGConfig:
    """All tuneable knobs live here — change once, propagates everywhere."""
    # chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # retrieval
    k: int = 4                          # number of docs retrieved
    # models
    llm_model: str = "llama3.2:3b"
    embed_model: str = "llama3.2:3b"
    # paths
    pdf_dir: str = "./hw_1/pdfs"
    chroma_dir: str = "./hw_1/chroma_db"
    collection_name: str = "se_docs"
    # misc
    experiment_name: str = "baseline"

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────
# 2.  Document Ingestion
# ──────────────────────────────────────────────

def load_and_split(cfg: RAGConfig) -> list:
    """Load every PDF in cfg.pdf_dir and split into chunks."""
    pdf_paths = sorted(Path(cfg.pdf_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in '{cfg.pdf_dir}/'")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    all_chunks = []
    for p in pdf_paths:
        docs = PyPDFLoader(str(p)).load()
        chunks = splitter.split_documents(docs)
        # tag every chunk with its source filename
        for c in chunks:
            c.metadata["source_file"] = p.name
        all_chunks.extend(chunks)
        print(f"  ✔ {p.name}  →  {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


# ──────────────────────────────────────────────
# 3.  Vector Store
# ──────────────────────────────────────────────

def build_vectorstore(chunks: list, cfg: RAGConfig) -> Chroma:
    """Embed chunks and persist to Chroma."""
    embeddings = OllamaEmbeddings(model=cfg.embed_model)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=cfg.collection_name,
        persist_directory=cfg.chroma_dir,
    )
    print(f"Vector store saved → {cfg.chroma_dir}/")
    return vectorstore


def load_vectorstore(cfg: RAGConfig) -> Chroma:
    """Re-open an existing Chroma collection (no re-embedding)."""
    embeddings = OllamaEmbeddings(model=cfg.embed_model)
    return Chroma(
        collection_name=cfg.collection_name,
        embedding_function=embeddings,
        persist_directory=cfg.chroma_dir,
    )


# ──────────────────────────────────────────────
# 4.  RAG Chain
# ──────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a software-engineering teaching assistant.
Use ONLY the context below to answer the question concisely and accurately.
If the answer is not in the context, say "I don't know based on the provided materials."

Context:
{context}

Question: {question}

Answer:"""


def build_rag_chain(vectorstore: Chroma, cfg: RAGConfig):
    """Return a LangChain LCEL chain: retriever → prompt → LLM → parser."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": cfg.k})
    llm       = OllamaLLM(model=cfg.llm_model, temperature=0)
    prompt    = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[{d.metadata.get('source_file','?')} p.{d.metadata.get('page','?')}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ──────────────────────────────────────────────
# 5.  Manual Testing Helper
# ──────────────────────────────────────────────

def run_manual_tests(chain, retriever, qa_pairs: list[dict], n: int = 5) -> list[dict]:
    """
    qa_pairs: list of {"question": ..., "expected": ..., "difficulty": ...}
    Returns enriched list with "actual" and "sources" filled in.
    """
    results = []
    for item in qa_pairs[:n]:
        q = item["question"]
        print(f"\n{'='*60}")
        print(f"[{item['difficulty'].upper()}] {q}")
        answer  = chain.invoke(q)
        sources = retriever.invoke(q)
        src_names = list({d.metadata.get("source_file", "?") for d in sources})
        print(f"Answer : {answer}")
        print(f"Sources: {src_names}")
        results.append({**item, "actual": answer, "sources": src_names})
    return results


# ──────────────────────────────────────────────
# 6.  Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cfg = RAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        k=4,
        experiment_name="baseline",
    )

    # ── Build ──
    chunks = load_and_split(cfg)
    vs     = build_vectorstore(chunks, cfg)

    chain, retriever = build_rag_chain(vs, cfg)

    # Load evaluation dataset
    with open("./hw_1/eval_dataset.json") as f:
        qa_pairs = json.load(f)

    # Manually test the first 5 questions
    results = run_manual_tests(chain, retriever, qa_pairs, n=5)

    # Persist manual-test results
    with open("./hw_1/manual_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✔ Results saved to manual_test_results.json")
