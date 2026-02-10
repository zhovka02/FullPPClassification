"""
Orchestrator script for Multi-Model Benchmarking with AI-based Evaluation and HTML Reporting.
"""
import pandas as pd
import time
import os
from dotenv import load_dotenv

# Local imports
from src.annotator import PrivacyPolicyAnnotator
from src.evaluator import Evaluator
from src.ai_evaluator import AIEvaluator
from src.llm_client import LLMClient
from src.visualizer import HTMLVisualizer
from src.utils import load_c3pa_dataset

# --- CONFIGURATION ---
DATASET_PATH = "./data"
REPORTS_DIR = "./reports"

# 1. Models to Benchmark
MODELS_TO_TEST = [
    "openrouter:x-ai/grok-4.1-fast",
    "gemini:gemini-2.5-flash",
    "openai:gpt-5-mini-2025-08-07",
    "openrouter:meta-llama/llama-4-maverick",
    "gemini:gemini-3-flash-preview"]

# 2. Judge Configuration
JUDGE_MODEL = "openai:gpt-4o-mini"

# 3. Policies to ignore (by ID)
IGNORED_POLICIES = [
  "DB_201",
  "DB_191",
  "DB_190",
  "DB_154",
  "DB_70",
  "DB_17",
  "DB_73",
  "DB_60",
  "DB_102",
  "DB_177"
]

TEST_LIMIT = 30
GENERATE_REPORTS = True

def main():
    load_dotenv()

    # Ensure reports directory exists
    if GENERATE_REPORTS:
        os.makedirs(REPORTS_DIR, exist_ok=True)

    print("--- C3PA AI-Judge Benchmark ---")
    print(f"Models: {MODELS_TO_TEST}")
    print(f"Judge: {JUDGE_MODEL}")

    # 1. Load Data
    policies = load_c3pa_dataset(DATASET_PATH)
    if not policies:
        print("ERROR: No data found.")
        return

    # 2. Initialize Evaluators
    strict_evaluator = Evaluator() # Standard F1/Exact Match
    visualizer = HTMLVisualizer()

    ai_evaluator = None
    try:
        judge_client = LLMClient(JUDGE_MODEL)
        ai_evaluator = AIEvaluator(judge_client)
        print("   > AI Judge initialized.")
    except Exception as e:
        print(f"CRITICAL: AI Judge init failed: {e}")
        return

    results = []

    # 3. Processing Loop
    for i, pol in enumerate(policies):
        if TEST_LIMIT and i >= TEST_LIMIT:
            break

        # Skip ignored policies
        if pol['id'] in IGNORED_POLICIES:
            print(f"\n[{i + 1}/{len(policies)}] Policy ID: {pol['id']} - IGNORED")
            continue

        print(f"\n[{i + 1}/{len(policies)}] Policy ID: {pol['id']}")

        ground_truth = pol.get('ground_truth', [])
        if not ground_truth:
            print("   > Skipping (No Ground Truth)")
            continue

        for model_name in MODELS_TO_TEST:
            print(f"   > Testing {model_name}...", end=" ", flush=True)

            try:
                # A. Inference
                annotator = PrivacyPolicyAnnotator(model_name=model_name)
                t0 = time.time()
                llm_preds = annotator.annotate(pol['text'])
                duration = time.time() - t0
                print(f"Done ({len(llm_preds)} preds in {duration:.1f}s)")

                # B. Standard Metrics (Reference)
                strict_metrics = strict_evaluator.compare_annotations(ground_truth, llm_preds)

                # C. AI Judging (Returns Metrics AND Decision Map)
                # This uses the logic: Filter by Label -> Filter by Overlap -> Ask LLM
                ai_metrics, ai_decisions, missed_gts = ai_evaluator.evaluate_batch(ground_truth, llm_preds)

                # D. Combine & Save
                row_data = strict_metrics.copy()
                row_data.update({
                    "policy_id": pol['id'],
                    "model": model_name,
                    "duration_sec": round(duration, 2),
                    "ai_precision": ai_metrics["precision"],
                    "ai_recall": ai_metrics["recall"],
                    "ai_f1": ai_metrics["f1"]
                })
                results.append(row_data)

                print(f"     > Strict F1: {strict_metrics['f1']:.2f}")
                print(f"     > AI Stats : P={ai_metrics['precision']} | R={ai_metrics['recall']} | F1={ai_metrics['f1']}")

                # E. Visualization
                if GENERATE_REPORTS:
                    # Sanitize filename
                    safe_name = model_name.replace(":", "_").replace("/", "_")
                    fname = os.path.join(REPORTS_DIR, f"{pol['id']}_{safe_name}.html")

                    visualizer.generate_report(
                        policy_id=pol['id'],
                        full_text=pol['text'],
                        human_anns=ground_truth,
                        llm_anns=llm_preds,
                        filename=fname,
                        ai_decisions=ai_decisions, # Pass the detailed judge results for coloring
                        missed_gts=missed_gts
                    )

            except Exception as e:
                print(f"\n     > FAILED: {e}")
                results.append({"model": model_name, "error": str(e)})

    # 4. Final Leaderboard
    if results:
        df = pd.DataFrame(results)

        # Save Raw Data
        df.to_csv("benchmark_full_results.csv", index=False)

        if "ai_f1" in df.columns:
            print("\n" + "="*60)
            print("FINAL LEADERBOARD (Sorted by AI F1)")
            print("="*60)

            # Filter out rows with errors (where ai_f1 might be NaN)
            valid_df = df[df["ai_f1"].notna()]

            if not valid_df.empty:
                leaderboard = valid_df.groupby("model")[["f1", "ai_precision", "ai_recall", "ai_f1", "duration_sec"]].mean()
                leaderboard = leaderboard.sort_values("ai_f1", ascending=False)
                print(leaderboard)
            else:
                print("No valid results to calculate leaderboard.")

        print("\nResults saved to 'benchmark_full_results.csv'")

if __name__ == "__main__":
    main()
