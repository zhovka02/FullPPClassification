import string

from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from tqdm import tqdm

from src.llm_client import LLMClient


class CustomDeepEvalLLM(DeepEvalBaseLLM):
    """
    Wrapper to make our existing LLMClient compatible with DeepEval.
    """

    def __init__(self, client: LLMClient):
        """
        Initializes the custom DeepEval LLM wrapper with a given LLMClient.

        Args:
            client (LLMClient): The LLM client to use for generating completions.
        """
        self.client = client

    def load_model(self):
        """
        Returns the underlying LLMClient instance.
        """
        return self.client

    def generate(self, prompt: str) -> str:
        """
        Generates a completion from the underlying model using the given prompt.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The generated response.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.client.get_completion(messages)

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generates a completion from the underlying model.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The generated response.
        """
        return self.generate(prompt)

    def get_model_name(self):
        """
        Retrieves the name of the underlying model.

        Returns:
            str: The model name.
        """
        return self.client.model


def normalize_text(text: str) -> str:
    """
    Normalizes text for comparison: lowercases, removes punctuation, reduces whitespace.
    Returns a clean string.

    Args:
        text (str): The input text to normalize.

    Returns:
        str: The normalized text.
    """
    if not text: return ""
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator).lower()
    return " ".join(text.split())


def check_containment(pred_text: str, gt_text: str) -> float:
    """
    Returns the percentage of GT tokens found in Pred text (Recall).

    Args:
        pred_text (str): The predicted text.
        gt_text (str): The ground truth text.

    Returns:
        float: The containment score between 0.0 and 1.0.
    """
    p_norm = normalize_text(pred_text)
    g_norm = normalize_text(gt_text)

    if not g_norm: return 0.0
    if g_norm in p_norm: return 1.0

    p_tokens = set(p_norm.split())
    g_tokens = g_norm.split()

    if not g_tokens: return 0.0

    found = sum(1 for t in g_tokens if t in p_tokens)
    return found / len(g_tokens)


class AIEvaluator:
    """
    Evaluator class for assessing AI model predictions against ground truth labels.
    """
    def __init__(self, client: LLMClient):
        """
        Initializes the AIEvaluator.

        Args:
            client (LLMClient): The client used for AI-based evaluation.
        """
        self.client = client
        self._cache = {}
        self.deepeval_model = CustomDeepEvalLLM(client)

    def _get_val(self, item, keys):
        """
        Safely retrieves a value from a dictionary using a list of possible keys.

        Args:
            item (dict or any): The item from which to retrieve the value.
            keys (list): The list of keys to try.

        Returns:
            str: The found value, or a string representation of the item if not a dict.
        """
        if not isinstance(item, dict): return str(item)
        for k in keys:
            if k in item: return item[k]
        return ""

    def _are_labels_compatible(self, label1: str, label2: str) -> bool:
        """
        Checks if two labels are compatible based on string matching.

        Args:
            label1 (str): The first label.
            label2 (str): The second label.

        Returns:
            bool: True if the labels are compatible, False otherwise.
        """
        l1 = label1.lower().strip()
        l2 = label2.lower().strip()
        if l1 == l2: return True
        if len(l1) > 5 and len(l2) > 5:
            if l1 in l2 or l2 in l1: return True
        return False

    def evaluate_batch(self, true_labels: list, pred_labels: list) -> tuple:
        """
        Evaluates a batch of predictions against ground truth labels.

        Args:
            true_labels (list): The ground truth labels.
            pred_labels (list): The predicted labels.

        Returns:
            tuple: A tuple containing:
                - metrics (dict): Precision, recall, and f1 scores.
                - decision_map (list): Detailed status for every prediction.
                - missed_gts (list): Ground truth items that were not matched.
        """
        if not pred_labels and not true_labels:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}, [], []
        if not pred_labels:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}, [], true_labels

        decision_map = []
        found_gt_indices = set()
        tp_preds = 0

        for pred in tqdm(pred_labels, desc="Evaluating", unit="pred", leave=False):
            p_text = self._get_val(pred, ['text', 'span', 'segment'])
            p_label = self._get_val(pred, ['category', 'label', 'type'])

            potential_gts = []
            for i, gt in enumerate(true_labels):
                gt_label = self._get_val(gt, ['category', 'label', 'type'])
                if self._are_labels_compatible(gt_label, p_label):
                    potential_gts.append((i, gt))

            matched_gts_for_this_pred = []
            best_match_score = 0.0
            closest_gt_text = None

            ai_reasoning_map = {}
            ai_rejection_reasons = []

            for i, gt in potential_gts:
                gt_text = self._get_val(gt, ['text', 'span', 'segment'])

                recall_score = check_containment(p_text, gt_text)
                precision_score = check_containment(gt_text, p_text)

                avg_score = (recall_score + precision_score) / 2
                if avg_score > best_match_score:
                    best_match_score = avg_score
                    closest_gt_text = gt_text

                match_type = None

                if recall_score >= 0.9:
                    match_type = "CORRECT_CONTAINMENT"
                elif precision_score >= 0.9:
                    match_type = "CORRECT_SUBSTRING"
                elif recall_score > 0.4 or precision_score > 0.4:
                    is_ai_match, _, reasoning = self._geval_judge(p_text, gt_text, p_label)
                    if is_ai_match:
                        match_type = "CORRECT_AI"
                        ai_reasoning_map[i] = reasoning
                    else:
                        ai_rejection_reasons.append({
                            "gt_text": gt_text,
                            "reasoning": reasoning
                        })

                if match_type:
                    matched_gts_for_this_pred.append((i, gt, match_type))

            if matched_gts_for_this_pred:
                tp_preds += 1

                primary_match_text = ""
                primary_status = "CORRECT_AI"
                is_deterministic = False

                for idx, gt, m_type in matched_gts_for_this_pred:
                    found_gt_indices.add(idx)

                    if "CONTAINMENT" in m_type:
                        primary_status = m_type
                        is_deterministic = True
                    elif "SUBSTRING" in m_type and "CONTAINMENT" not in primary_status:
                        primary_status = m_type
                        is_deterministic = True
                    elif "STRICT" in m_type:
                        primary_status = m_type
                        is_deterministic = True

                    if not primary_match_text:
                        primary_match_text = self._get_val(gt, ['text', 'span', 'segment'])

                final_reasoning = None
                if not is_deterministic:
                    first_idx = matched_gts_for_this_pred[0][0]
                    final_reasoning = ai_reasoning_map.get(first_idx, "")

                decision_map.append({
                    "text": p_text,
                    "label": p_label,
                    "status": primary_status,
                    "match_with": primary_match_text,
                    "reasoning": final_reasoning,
                    "matched_count": len(matched_gts_for_this_pred)
                })
            else:
                rejection_reasoning = None
                if ai_rejection_reasons:
                    rejection_reasoning = ai_rejection_reasons[0]["reasoning"]

                decision_map.append({
                    "text": p_text,
                    "label": p_label,
                    "status": "WRONG",
                    "closest_match": closest_gt_text if best_match_score > 0.1 else None,
                    "closest_score": round(best_match_score, 2),
                    "reasoning": rejection_reasoning
                })

        precision = tp_preds / len(pred_labels) if pred_labels else 0.0
        recall = len(found_gt_indices) / len(true_labels) if true_labels else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3)
        }

        missed_gts = [gt for i, gt in enumerate(true_labels) if i not in found_gt_indices]

        return metrics, decision_map, missed_gts

    def _geval_judge(self, pred_text: str, gt_text: str, label: str) -> tuple:
        """
        Uses DeepEval's GEval to determine if pred_text is equivalent to gt_text.

        Args:
            pred_text (str): The predicted text.
            gt_text (str): The ground truth text.
            label (str): The label associated with the text.

        Returns:
            tuple: (is_match: bool, score: float, reasoning: str)
        """
        key = (pred_text, gt_text, label)
        if key in self._cache: return self._cache[key]
        test_case = LLMTestCase(
            input=f"Extract text for label: {label}",
            actual_output=pred_text,
            expected_output=gt_text
        )

        try:
            metric = GEval(
                name="Legal Extraction Equivalence",
                criteria=(
                    "Compare the Actual Output (AI Prediction) with the Expected Output (Ground Truth). "
                    "1. If the Actual Output contains the Expected Output (even if it has extra text), it is CORRECT. "
                    "2. If the Actual Output is a list/table and the Expected Output is one item from that list, it is CORRECT. "
                    "3. If the Actual Output is a substring of the Expected Output that preserves the main meaning, it is CORRECT."
                ),
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                model=self.deepeval_model,
                threshold=0.6
            )

            metric.measure(test_case)

            score = metric.score
            reasoning = metric.reason
            is_match = metric.is_successful()

            result = (is_match, score, reasoning)
            self._cache[key] = result
            return result
        except Exception as e:
            print(f"AI Judge Error: {e}")
            return False, 0.0, f"Error: {str(e)}"