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
        self.client = client

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        # DeepEval passes a raw string prompt
        messages = [{"role": "user", "content": prompt}]
        return self.client.get_completion(messages)

    async def a_generate(self, prompt: str) -> str:
        # For async calls, we just use the sync version for now
        return self.generate(prompt)

    def get_model_name(self):
        return self.client.model


def normalize_text(text: str) -> str:
    """
    Normalizes text for comparison: lowercases, removes punctuation, reduces whitespace.
    Returns a clean string.
    """
    if not text: return ""
    # Remove punctuation
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator).lower()
    # Remove extra whitespace
    return " ".join(text.split())


def check_containment(pred_text: str, gt_text: str) -> float:
    """
    Returns the percentage of GT tokens found in Pred text (Recall).
    """
    p_norm = normalize_text(pred_text)
    g_norm = normalize_text(gt_text)

    if not g_norm: return 0.0
    if g_norm in p_norm: return 1.0  # Exact substring match

    # Token-based check for fuzzy containment
    p_tokens = set(p_norm.split())
    g_tokens = g_norm.split()

    if not g_tokens: return 0.0

    found = sum(1 for t in g_tokens if t in p_tokens)
    return found / len(g_tokens)


class AIEvaluator:
    def __init__(self, client: LLMClient):
        self.client = client
        self._cache = {}
        self.deepeval_model = CustomDeepEvalLLM(client)

    def _get_val(self, item, keys):
        if not isinstance(item, dict): return str(item)
        for k in keys:
            if k in item: return item[k]
        return ""

    def _are_labels_compatible(self, label1: str, label2: str) -> bool:
        l1 = label1.lower().strip()
        l2 = label2.lower().strip()
        if l1 == l2: return True
        # Loose matching for labels like "Data Collected" vs "Personal Information Collected"
        if len(l1) > 5 and len(l2) > 5:
            if l1 in l2 or l2 in l1: return True
        return False

    def evaluate_batch(self, true_labels: list, pred_labels: list) -> tuple:
        """
        Returns:
            metrics (dict): {'precision': 0.8, ...}
            decision_map (list): List of dicts with detailed status for every prediction.
            missed_gts (list): List of GT items that were not matched.
        """
        if not pred_labels and not true_labels:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}, [], []
        if not pred_labels:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}, [], true_labels

        decision_map = []
        found_gt_indices = set()
        tp_preds = 0

        # --- EVALUATION LOOP ---
        for pred in tqdm(pred_labels, desc="Evaluating", unit="pred", leave=False):
            p_text = self._get_val(pred, ['text', 'span', 'segment'])
            p_label = self._get_val(pred, ['category', 'label', 'type'])

            # Find all compatible GTs (Label Match)
            potential_gts = []
            for i, gt in enumerate(true_labels):
                gt_label = self._get_val(gt, ['category', 'label', 'type'])
                if self._are_labels_compatible(gt_label, p_label):
                    potential_gts.append((i, gt))

            matched_gts_for_this_pred = []
            best_match_score = 0.0
            closest_gt_text = None

            # Temp storage to capture reasoning if needed
            ai_reasoning_map = {}
            ai_rejection_reasons = []  # Store reasons why AI rejected matches

            # CHECK AGAINST ALL CANDIDATES
            for i, gt in potential_gts:
                gt_text = self._get_val(gt, ['text', 'span', 'segment'])

                # Check 1: Strict Containment (Recall focus)
                # Does the Prediction contain the GT? (Fixes "Big Block" issue)
                recall_score = check_containment(p_text, gt_text)

                # Check 2: Reverse Containment (Precision focus)
                # Is the Prediction a substring of the GT?
                precision_score = check_containment(gt_text, p_text)

                # Track closest match for reporting (debugging)
                avg_score = (recall_score + precision_score) / 2
                if avg_score > best_match_score:
                    best_match_score = avg_score
                    closest_gt_text = gt_text

                match_type = None

                # A. Direct Matches (Deterministic)
                if recall_score >= 0.9:  # GT is fully inside Prediction
                    match_type = "CORRECT_CONTAINMENT"
                elif precision_score >= 0.9:  # Prediction is fully inside GT
                    match_type = "CORRECT_SUBSTRING"

                # B. AI Judge (Only if not a direct match, but close)
                elif recall_score > 0.4 or precision_score > 0.4:
                    is_ai_match, _, reasoning = self._geval_judge(p_text, gt_text, p_label)
                    if is_ai_match:
                        match_type = "CORRECT_AI"
                        ai_reasoning_map[i] = reasoning
                    else:
                        # Store rejection reasoning for wrong predictions
                        ai_rejection_reasons.append({
                            "gt_text": gt_text,
                            "reasoning": reasoning
                        })

                if match_type:
                    matched_gts_for_this_pred.append((i, gt, match_type))

            # --- DECISION LOGIC ---
            if matched_gts_for_this_pred:
                tp_preds += 1

                primary_match_text = ""
                primary_status = "CORRECT_AI"  # Default weak status
                is_deterministic = False

                # Register ALL matched GTs as found (One-to-Many support)
                for idx, gt, m_type in matched_gts_for_this_pred:
                    found_gt_indices.add(idx)

                    # Upgrade status if we find a better one
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

                # FIX: Only include reasoning if the match relies SOLELY on AI
                # This ensures the badge is suppressed for deterministic matches
                final_reasoning = None
                if not is_deterministic:
                    # Just grab the reasoning from the first matched GT for display
                    first_idx = matched_gts_for_this_pred[0][0]
                    final_reasoning = ai_reasoning_map.get(first_idx, "")

                decision_map.append({
                    "text": p_text,
                    "label": p_label,
                    "status": primary_status,
                    "match_with": primary_match_text,
                    "reasoning": final_reasoning,  # Will be None if deterministic match found
                    "matched_count": len(matched_gts_for_this_pred)
                })
            else:
                # Build reasoning from AI rejections if any
                rejection_reasoning = None
                if ai_rejection_reasons:
                    # Use the first rejection reason (or combine if multiple)
                    rejection_reasoning = ai_rejection_reasons[0]["reasoning"]

                decision_map.append({
                    "text": p_text,
                    "label": p_label,
                    "status": "WRONG",
                    "closest_match": closest_gt_text if best_match_score > 0.1 else None,
                    "closest_score": round(best_match_score, 2),
                    "reasoning": rejection_reasoning  # AI reasoning for why it was rejected
                })

        # --- METRICS ---
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
        Returns: (is_match: bool, score: float, reasoning: str)
        """
        key = (pred_text, gt_text, label)
        if key in self._cache: return self._cache[key]
        test_case = LLMTestCase(
            input=f"Extract text for label: {label}",
            actual_output=pred_text,
            expected_output=gt_text
        )

        try:
            # Revised criteria to support one-to-many and containment
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