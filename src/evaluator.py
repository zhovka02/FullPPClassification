import string
from typing import List, Dict


class Evaluator:
    def __init__(self, match_threshold: float = 0.3):
        # Lowered threshold to 0.3 because F1 is much stricter than Containment.
        # An F1 of 0.3 usually implies a significant, valid overlap in long texts.
        self.match_threshold = match_threshold
        self.stopwords = {
            "the", "and", "or", "of", "to", "a", "in", "is", "that", "for",
            "on", "with", "as", "by", "at", "it", "be", "this", "from", "an",
            "which", "we", "our", "us", "you", "your", "are", "not", "have",
            "may", "can", "will", "data", "information", "services", "privacy"
        }

    def _clean_tokens(self, text: str) -> List[str]:
        """
        Splits text into tokens, removes punctuation and stop words.
        Returns a LIST (not set) to preserve frequency for F1 counting.
        """
        if not text: return []
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        return [t for t in tokens if t not in self.stopwords]

    def _compute_token_f1(self, text_pred: str, text_ref: str) -> float:
        """
        Calculates SQuAD-style Token F1 Score.
        """
        pred_toks = self._clean_tokens(text_pred)
        ref_toks = self._clean_tokens(text_ref)

        if len(pred_toks) == 0 or len(ref_toks) == 0:
            return 0.0

        common = collections.Counter(pred_toks) & collections.Counter(ref_toks)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(ref_toks)

        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def compare_annotations(self, human_anns: List[Dict], llm_anns: List[Dict]) -> Dict[str, float]:
        tp = 0
        fp = 0

        # Track which human annotations were matched
        matched_human_indices = set()

        # Import collections inside the method if not at top level (or use standard dicts)
        global collections
        import collections

        for pred in llm_anns:
            label = pred.get("label")
            text_pred = pred.get("text", "")

            candidates = [
                (i, h) for i, h in enumerate(human_anns)
                if h['label'].lower() == label.lower()
            ]

            best_score = 0.0
            best_human_text = ""
            best_idx = -1

            for idx, hum in candidates:
                # Use Token F1 instead of Containment
                score = self._compute_token_f1(text_pred, hum['text'])
                if score > best_score:
                    best_score = score
                    best_human_text = hum['text']
                    best_idx = idx

            # Store metadata
            pred['_match_score'] = best_score
            pred['_matched_human_text'] = best_human_text

            if best_score >= self.match_threshold:
                tp += 1
                matched_human_indices.add(best_idx)
                pred['_match_status'] = "Hit"
            else:
                fp += 1
                pred['_match_status'] = "Miss"

        fn = len(human_anns) - len(matched_human_indices)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }