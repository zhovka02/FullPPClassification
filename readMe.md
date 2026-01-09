# 📜 Privacy Policy Classification using Long-Context LLMs
## 📖 Overview

This project implements an automated pipeline for analyzing and classifying Privacy Policies based on the **C3PA Taxonomy**. By leveraging Large Language Models (LLMs) with **long-context windows**, the system extracts data practices, categories, and attributes from complex legal texts.

The core innovation of this project lies in its **robust evaluation strategy**, designed to handle the discrepancy between verbose human annotations (C3PA dataset) and concise LLM extractions.

## 🚀 Key Features

* **⚡ Multi-Model Support:** Integrated with OpenAI (GPT-4o, GPT-5-mini), Google Gemini (1.5 Pro/Flash), and OpenRouter (Grok, Llama, Qwen).
* **🧠 Dynamic Prompting:** Injects taxonomy definitions from `config.py` directly into the system prompt, allowing the LLM to act as a "trained expert" rather than guessing labels.
* **⚖️ Fuzzy Evaluation Logic:** Implements a custom evaluator using **textual similarity thresholds (0.3)** instead of strict string matching. This accounts for LLMs correcting punctuation or extracting minimal spans compared to messy dataset annotations.
* **📊 HTML Visualization:** Automatically generates side-by-side HTML reports comparing Ground Truth vs. Prediction for qualitative analysis.

## 📂 Project Structure

```text
fullppclassification/
├── data/                   # Dataset (Texts, HTMLs, Contexts, Annotations)
├── reports/                # Generated HTML visualization reports
├── src/
│   ├── annotator.py        # Logic for LLM interaction and classification
│   ├── evaluator.py        # Fuzzy matching and metric calculation (F1, Precision, Recall)
│   ├── visualizer.py       # Generates HTML side-by-side comparisons
│   ├── config.py           # C3PA Taxonomy definitions and configuration
│   └── utils.py            # Helper functions
├── main.py                 # Entry point for the pipeline
├── benchmark_results.csv   # Raw performance data
├── pyproject.toml          # Dependency configuration (uv)
└── requirements.txt        # Standard requirements

```

## 🛠️ Installation & Setup

This project uses `uv` for fast package management, but supports standard `pip` as well.

### Prerequisites

* Python 3.10+
* API Keys for OpenAI, Google Gemini, or OpenRouter.

### 1. Clone the repository

```bash
git clone https://github.com/zhovka02/fullppclassification.git
cd fullppclassification

```

### 2. Install dependencies

**Using uv (Recommended):**

```bash
uv sync

```

**Using pip:**

```bash
pip install -r requirements.txt

```

### 3. Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=...

```

## 💻 Usage

To run the full pipeline (Annotation -> Evaluation -> Visualization):

```bash
python main.py

```

*You can modify `main.py` or `src/config.py` to switch between models (e.g., `gpt-4o` vs `gemini-1.5-pro`).*
