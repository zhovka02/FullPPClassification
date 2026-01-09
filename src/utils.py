import os
import json
import re
import pandas as pd
from typing import List, Dict, Any


def parse_llm_json(response_text: str) -> List[Dict[str, Any]]:
    try:
        # Improved regex to catch JSON between markdown blocks or raw
        match = re.search(r"```json\s*(.*?)```", response_text, re.DOTALL)
        if match:
            cleaned_text = match.group(1).strip()
        else:
            cleaned_text = response_text.strip()

        data = json.loads(cleaned_text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            return []
    except json.JSONDecodeError:
        print(f"DEBUG: JSON Parse Error. Response excerpt: {response_text[:100]}...")
        return []


def load_c3pa_dataset(root_path: str) -> List[Dict[str, Any]]:
    dataset = []
    subsets = ['DB', 'WS']

    print(f"Loading C3PA data from {root_path}...")

    for subset in subsets:
        anno_dir = os.path.join(root_path, "Annotations", subset)
        text_dir = os.path.join(root_path, "Texts", subset)

        if not os.path.exists(anno_dir): continue

        for filename in os.listdir(anno_dir):
            if not filename.endswith(".csv"): continue

            file_id = filename.replace(".csv", "")
            csv_path = os.path.join(anno_dir, filename)
            txt_path = os.path.join(text_dir, f"{file_id}.txt")

            if not os.path.exists(txt_path): continue

            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()

                df = pd.read_csv(csv_path)
                df.columns = [c.lower() for c in df.columns]

                annotator_col = next((c for c in df.columns if 'ranumb' in c), None)
                text_col = next((c for c in df.columns if 'text' in c or 'segment' in c), None)
                label_col = next((c for c in df.columns if 'category' in c or 'label' in c), None)

                if annotator_col and text_col:

                    df['text_len'] = df[text_col].astype(str).str.len()
                    completeness = df.groupby(annotator_col)['text_len'].sum()

                    best_annotator = completeness.idxmax()
                    df = df[df[annotator_col] == best_annotator]

                if not label_col or not text_col: continue

                ground_truth = []
                for _, row in df.iterrows():
                    ground_truth.append({
                        "label": str(row[label_col]).strip(),
                        "text": str(row[text_col]).strip()
                    })

                dataset.append({
                    "id": f"{subset}_{file_id}",
                    "text": full_text,
                    "ground_truth": ground_truth
                })

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Successfully loaded {len(dataset)} policy documents.")
    return dataset