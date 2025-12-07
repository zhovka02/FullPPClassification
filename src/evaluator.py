from typing import List, Dict


class Evaluator:
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def _compute_iou(self, text_a: str, text_b: str) -> float:
        """
        Calculates Jaccard Similarity (Intersection over Union) of tokens.
        """
        set_a = set(str(text_a).lower().split())
        set_b = set(str(text_b).lower().split())

        if not set_a and not set_b: return 1.0
        if not set_a or not set_b: return 0.0

        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union

    def compare_annotations(self, human_anns: List[Dict], llm_anns: List[Dict]) -> Dict[str, float]:
        """
        Compares LLM extracted segments against Human segments.
        """
        tp = 0
        fp = 0

        # Create a working copy of human annotations to tick off matches
        unmatched_humans = human_anns.copy()

        for pred in llm_anns:
            label = pred.get("label")
            text_pred = pred.get("text", "")

            # Filter human annotations to only those with matching labels (fuzzy match safe)
            candidates = [h for h in unmatched_humans if h['label'] in label or label in h['label']]

            best_iou = 0.0
            best_match = None

            # Find the best text overlap for this label
            for hum in candidates:
                iou = self._compute_iou(text_pred, hum['text'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = hum

            # Threshold Check
            if best_iou >= self.iou_threshold:
                tp += 1
                if best_match in unmatched_humans:
                    unmatched_humans.remove(best_match)
            else:
                fp += 1

        fn = len(unmatched_humans)

        # Metrics Calculation
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