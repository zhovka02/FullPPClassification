"""
Orchestrator script for Multi-Model Benchmarking with HTML Visualization.
"""
import pandas as pd
import time
import os
from dotenv import load_dotenv

from src.annotator import PrivacyPolicyAnnotator
from src.evaluator import Evaluator
from src.visualizer import HTMLVisualizer
from src.utils import load_c3pa_dataset

# --- CONFIGURATION ---
DATASET_PATH = "./data"
REPORTS_DIR = "./reports"  # Folder to save HTML visualisations

# LIST of models to benchmark
MODELS_TO_TEST = [
    "openrouter:x-ai/grok-4.1-fast",
    "gemini:gemini-2.5-flash",
    "openai:gpt-5-mini-2025-08-07",
    "openrouter:meta-llama/llama-4-maverick",
    "openrouter:qwen/qwen-turbo"
]

TEST_LIMIT = 10
GENERATE_REPORTS = True


def main():
    load_dotenv()

    # Create reports directory if it doesn't exist
    if GENERATE_REPORTS:
        os.makedirs(REPORTS_DIR, exist_ok=True)

    print("--- C3PA Multi-Model Benchmark ---")
    print(f"Models selected: {MODELS_TO_TEST}")

    # 1. Load Data
    policies = load_c3pa_dataset(DATASET_PATH)
    if not policies:
        print("ERROR: No data found.")
        return

    # 2. Initialize Evaluator & Visualizer
    evaluator = Evaluator()
    visualizer = HTMLVisualizer()

    results = []

    # 3. Processing Loop
    for i, pol in enumerate(policies):
        if TEST_LIMIT and i >= TEST_LIMIT:
            break

        print(f"\n[{i + 1}/{len(policies)}] Policy ID: {pol['id']}")
        print("-" * 40)

        # --- INNER LOOP: Iterate through all defined models ---
        for model_name in MODELS_TO_TEST:
            print(f"   > Running Model: {model_name}...", end=" ", flush=True)

            try:
                # Instantiate annotator specifically for this model
                annotator = PrivacyPolicyAnnotator(model_name=model_name)

                start_time = time.time()

                # A. Inference
                llm_predictions = annotator.annotate(pol['text'])
                duration = time.time() - start_time

                # B. Evaluation
                metrics = evaluator.compare_annotations(pol['ground_truth'], llm_predictions)

                # C. Tagging results
                metrics["policy_id"] = pol['id']
                metrics["model"] = model_name
                metrics["duration_sec"] = round(duration, 2)
                metrics["num_extracted"] = len(llm_predictions)

                results.append(metrics)

                print(f"Done ({duration:.1f}s)")
                print(f"     F1: {metrics['f1']:.2f} | Precision: {metrics['precision']:.2f} | Recall: {metrics['recall']:.2f}")

                # D. Visualization (Save HTML)
                if GENERATE_REPORTS:
                    # Sanitize model name for filename (remove colons)
                    safe_model_name = model_name.replace(":", "_").replace("/", "_")
                    filename = os.path.join(REPORTS_DIR, f"{pol['id']}_{safe_model_name}.html")

                    visualizer.generate_report(
                        policy_id=pol['id'],
                        full_text=pol['text'],
                        human_anns=pol['ground_truth'],
                        llm_anns=llm_predictions,
                        filename=filename
                    )
                    print(f"     [Visual] Saved to {filename}")

            except Exception as e:
                print(f"FAILED: {e}")
                results.append({
                    "policy_id": pol['id'],
                    "model": model_name,
                    "f1": 0.0,
                    "error": str(e)
                })

    # 4. Final Comparison Report
    if results:
        df = pd.DataFrame(results)

        print("\n" + "=" * 50)
        print("BENCHMARK LEADERBOARD (Average Scores)")
        print("=" * 50)

        # Group by Model to see who wins
        leaderboard = df.groupby("model")[["precision", "recall", "f1", "duration_sec"]].mean()
        print(leaderboard)

        # Save raw data
        df.to_csv("benchmark_results.csv", index=False)
        print("\nDetailed results saved to 'benchmark_results.csv'")


if __name__ == "__main__":
    main()