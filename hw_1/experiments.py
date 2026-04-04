import json
import itertools
from dataclasses import dataclass, field
from typing import List, Dict
import pandas as pd

from rag_pipeline import RAGConfig, load_and_split, build_vectorstore
from langsmith_evaluation import run_evaluation

CHUNK_SIZES  = [500, 1000, 1500]
OVERLAPS     = [50, 200]
K_VALUES     = [2, 4, 6]

def build_experiment_configs() -> List[RAGConfig]:
    """
    Generates 3×2×3 = 18 combinations; we pick the first 10 distinct ones
    by sampling meaningfully across the space.
    """
    combos = list(itertools.product(CHUNK_SIZES, OVERLAPS, K_VALUES))
    # Keep a representative 10: all chunk sizes, both overlaps, varied k
    selected = combos[:10]
    configs = []
    for i, (cs, ov, k) in enumerate(selected, start=1):
        configs.append(RAGConfig(
            chunk_size=cs,
            chunk_overlap=ov,
            k=k,
            chroma_dir=f"./hw_1/chroma_exp_{i}",
            collection_name=f"se-exp-{i}",
            experiment_name=f"exp{i:02d}_cs{cs}_ov{ov}_k{k}",
        ))
    return configs


def run_all_experiments():
    configs = build_experiment_configs()
    summary_rows = []

    for cfg in configs:
        print(f"\n{'═'*60}")
        print(f"EXPERIMENT: {cfg.experiment_name}")
        print(f"  chunk_size={cfg.chunk_size}  overlap={cfg.chunk_overlap}  k={cfg.k}")

        chunks = load_and_split(cfg)
        build_vectorstore(chunks, cfg)

        results = run_evaluation(cfg, experiment_prefix=cfg.experiment_name)

        scores = {"correctness": [], "relevance": [],
                  "faithfulness": [], "conciseness": []}
        for r in results._results:
            for fb in r.get("evaluation_results", {}).get("results", []):
                scores[fb.key].append(fb.score)

        row = {
            "experiment":   cfg.experiment_name,
            "chunk_size":   cfg.chunk_size,
            "chunk_overlap":cfg.chunk_overlap,
            "k":            cfg.k,
        }
        for metric, vals in scores.items():
            row[metric] = round(sum(vals) / len(vals), 4) if vals else None
        row["avg_composite"] = round(
            sum(v for v in [row["correctness"], row["relevance"],
                            row["faithfulness"], row["conciseness"]]
                if v is not None) / 4, 4
        )
        summary_rows.append(row)
        print(f"  composite score: {row['avg_composite']:.4f}")

    df = pd.DataFrame(summary_rows)
    df = df.sort_values("avg_composite", ascending=False)
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS (sorted by composite score)")
    print("="*70)
    print(df.to_string(index=False))

    df.to_csv("experiment_results.csv", index=False)
    print("\n✔ Saved → experiment_results.csv")

    best = df.iloc[0]
    print(f"\n🏆 OPTIMAL CONFIG: {best['experiment']}")
    print(f"   chunk_size={int(best['chunk_size'])}  "
          f"overlap={int(best['chunk_overlap'])}  "
          f"k={int(best['k'])}")
    print(f"   Correctness  = {best['correctness']:.4f}")
    print(f"   Relevance    = {best['relevance']:.4f}")
    print(f"   Faithfulness = {best['faithfulness']:.4f}")
    print(f"   Conciseness  = {best['conciseness']:.4f}")
    print(f"   COMPOSITE    = {best['avg_composite']:.4f}")

    return df


def analyze_failures(results_file: str = "./hw_1/manual_test_results.json"):
    """Print structured failure analysis for the report."""
    with open(results_file) as f:
        results = json.load(f)

    print("\n── FAILURE CASE ANALYSIS ──────────────────────────────")
    failure_categories = {
        "off_topic_retrieval": [],
        "incomplete_answer":   [],
        "hallucination":       [],
    }

    for r in results:
        q, exp, act = r["question"], r["expected"], r.get("actual", "")
        if not act.strip() or "I don't know" in act:
            failure_categories["off_topic_retrieval"].append(r)
        elif len(act.split()) < len(exp.split()) * 0.4:
            failure_categories["incomplete_answer"].append(r)
        elif any(kw in act.lower() for kw in ["always", "never", "every system"]):
            failure_categories["hallucination"].append(r)

    for cat, items in failure_categories.items():
        print(f"\n  [{cat}]  ({len(items)} cases)")
        for it in items:
            print(f"    Q: {it['question'][:80]}…")
            print(f"    Expected snippet: {it['expected'][:80]}…")
            print(f"    Got: {it.get('actual','')[:80]}…")


if __name__ == "__main__":
    df = run_all_experiments()
    analyze_failures()
