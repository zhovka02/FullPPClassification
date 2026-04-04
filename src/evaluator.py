import collections
import string
from typing import List, Dict


def clean_tokens(text: str) -> List[str]:
    """
    Splits text into tokens, removes punctuation and stop words.
    Returns a LIST (not set) to preserve frequency for F1 counting.

    Args:
        text (str): The input text to clean and tokenize.

    Returns:
        List[str]: A list of cleaned tokens.
    """
    stopwords = {
        "the", "and", "or", "of", "to", "a", "in", "is", "that", "for",
        "on", "with", "as", "by", "at", "it", "be", "this", "from", "an",
        "which", "we", "our", "us", "you", "your", "are", "not", "have",
        "may", "can", "will", "data", "information", "services", "privacy"
    }
    if not text: return []
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in stopwords]


def compute_token_f1(text_pred: str, text_ref: str) -> float:
    """
    Calculates SQuAD-style Token F1 Score.

    Args:
        text_pred (str): The predicted text.
        text_ref (str): The reference (ground truth) text.

    Returns:
        float: The calculated F1 score.
    """
    pred_toks = clean_tokens(text_pred)
    ref_toks = clean_tokens(text_ref)

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


class Evaluator:
    """
    Class responsible for evaluating predicted annotations against human annotations.
    """
    def __init__(self, match_threshold: float = 0.3):
        """
        Initializes the Evaluator.

        Args:
            match_threshold (float): The minimum F1 score required to consider a match a true positive.
        """
        self.match_threshold = match_threshold

    def compare_annotations(self, human_anns: List[Dict], llm_anns: List[Dict]) -> Dict[str, float]:
        """
        Compares LLM annotations against human annotations and calculates precision, recall, and F1 score.

        Args:
            human_anns (List[Dict]): The list of ground truth human annotations.
            llm_anns (List[Dict]): The list of predicted LLM annotations.

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics including precision, recall, and f1.
        """
        tp = 0
        fp = 0

        matched_human_indices = set()

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
                score = compute_token_f1(text_pred, hum['text'])
                if score > best_score:
                    best_score = score
                    best_human_text = hum['text']
                    best_idx = idx

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