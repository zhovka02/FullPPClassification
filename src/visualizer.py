import os
import html


class HTMLVisualizer:
    def __init__(self):
        self.css = """
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background-color: #f9f9f9; }
            h1 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
            h2 { color: #555; margin-top: 30px; }
            .badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; margin-right: 8px; font-size: 0.85em; display: inline-block; min-width: 80px; text-align: center; }

            /* Status Colors */
            .strict { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .substring { background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }
            .ai-match { background-color: #cce5ff; color: #004085; border: 1px solid #b8daff; }
            .wrong { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .missed { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
            .judge-badge { background-color: #6f42c1; color: white; border: 1px solid #59359a; }

            .container { display: flex; gap: 20px; }
            .column { flex: 1; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }

            .card { margin-bottom: 15px; padding: 12px; border: 1px solid #eee; border-radius: 6px; background-color: #fff; }
            .card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.05); }

            .label-tag { font-weight: bold; color: #444; display: block; margin-bottom: 6px; }
            .text-content { font-family: 'Consolas', 'Monaco', monospace; font-size: 0.9em; color: #333; white-space: pre-wrap; }

            .match-info { margin-top: 8px; padding-top: 8px; border-top: 1px dashed #eee; font-size: 0.85em; color: #666; }
            .reasoning { margin-top: 5px; font-style: italic; color: #555; background: #f8f9fa; padding: 5px; border-radius: 4px; }
            .closest-match { margin-top: 5px; font-size: 0.85em; color: #999; border-top: 1px dotted #eee; padding-top: 5px; }

            .stats-box { background: #fff; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; gap: 20px; }
            .stat-item { text-align: center; }
            .stat-value { font-size: 1.5em; font-weight: bold; color: #007bff; }
            .stat-label { font-size: 0.9em; color: #666; }
        </style>
        """

    def generate_report(self, policy_id, full_text, human_anns, llm_anns, filename, ai_decisions=None, missed_gts=None,
                        metrics=None):
        """
        metrics: Optional dict containing pre-calculated {'precision', 'recall', 'f1'} from AIEvaluator.
        """

        # 1. Calculate visual counts (Cards displayed)
        total_preds = len(ai_decisions) if ai_decisions else len(llm_anns)

        # Count Correct Predictions (for visual validation)
        tp_visual_count = sum(1 for d in (ai_decisions or []) if "CORRECT" in d['status'])
        fp_visual_count = total_preds - tp_visual_count
        fn_visual_count = len(missed_gts) if missed_gts is not None else 0

        # 2. Determine Stats to Display
        if metrics:
            # Use the correct, de-duplicated stats from AIEvaluator
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            f1 = metrics.get('f1', 0.0)
        else:
            # Fallback (Legacy logic - prone to Recall > 1 error)
            total_gt = len(human_anns)
            precision = tp_visual_count / total_preds if total_preds > 0 else 0
            recall = tp_visual_count / total_gt if total_gt > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        content = [f"<!DOCTYPE html><html><head>{self.css}</head><body>"]
        content.append(f"<h1>Policy Report: {policy_id}</h1>")

        # Stats Header
        content.append(f"""
        <div class="stats-box">
            <div class="stat-item"><div class="stat-value">{precision:.2f}</div><div class="stat-label">Precision</div></div>
            <div class="stat-item"><div class="stat-value">{recall:.2f}</div><div class="stat-label">Recall</div></div>
            <div class="stat-item"><div class="stat-value">{f1:.2f}</div><div class="stat-label">F1 Score</div></div>
            <div class="stat-item" style="border-left: 1px solid #eee; padding-left: 20px;">
                <div class="stat-value">{tp_visual_count}</div><div class="stat-label">Correct Preds</div>
            </div>
            <div class="stat-item"><div class="stat-value">{fp_visual_count}</div><div class="stat-label">Wrong</div></div>
            <div class="stat-item"><div class="stat-value">{fn_visual_count}</div><div class="stat-label">Missed GT</div></div>
        </div>
        """)

        content.append('<div class="container">')

        # --- COLUMN 1: LLM PREDICTIONS ---
        content.append('<div class="column"><h2>LLM Predictions (What AI Found)</h2>')

        if ai_decisions:
            for item in ai_decisions:
                status = item.get('status', 'WRONG')

                # Logic to determine badge color/text
                if "CORRECT_STRICT" in status:
                    badge_class = "strict"
                    badge_text = "EXACT"
                elif "CORRECT_CONTAINMENT" in status:
                    badge_class = "strict"
                    badge_text = "INCLUDES"
                elif "CORRECT_SUBSTRING" in status:
                    badge_class = "substring"
                    badge_text = "SUBSTR"
                elif "CORRECT_AI" in status:
                    badge_class = "ai-match"
                    badge_text = "AI JUDGE"
                else:
                    badge_class = "wrong"
                    badge_text = "WRONG"

                # Check specifically if reasoning exists to show the badge
                judge_badge = ""
                if item.get('reasoning'):
                    judge_badge = '<span class="badge judge-badge">LLM-Judge</span>'

                match_html = ""
                if item.get('match_with'):
                    match_html = f"<div class='match-info'><b>Matched GT:</b> {html.escape(item['match_with'])}</div>"

                geval_html = ""
                if item.get("reasoning"):
                    # Only show reasoning if it's an AI match
                    geval_html = f"<div class='reasoning'><b>DeepEval:</b> {html.escape(item['reasoning'])}</div>"

                closest_html = ""
                if status == "WRONG" and item.get('closest_match'):
                    closest_html = f"<div class='closest-match'><b>Closest GT (F1={item.get('closest_score', 0):.2f}):</b> {html.escape(item['closest_match'])}</div>"

                content.append(f"""
                <div class="card" style="border-left: 5px solid {self._get_color(badge_class)}">
                    <div>
                        <span class="badge {badge_class}">{badge_text}</span>
                        {judge_badge}
                        <span class="label-tag">{html.escape(item['label'])}</span>
                    </div>
                    <div class="text-content">{html.escape(item['text'])}</div>
                    {match_html}
                    {geval_html}
                    {closest_html}
                </div>
                """)
        else:
            content.append("<p>No prediction data available.</p>")

        content.append('</div>')  # End Column 1

        # --- COLUMN 2: MISSED GROUND TRUTH ---
        content.append('<div class="column"><h2>Missed Ground Truth (What AI Missed)</h2>')

        if missed_gts:
            for gt in missed_gts:
                label = gt.get('label') or gt.get('category') or 'Unknown'
                text = gt.get('text') or gt.get('span') or ''

                content.append(f"""
                <div class="card" style="border-left: 5px solid #ffeeba">
                    <div>
                        <span class="badge missed">MISSED</span>
                        <span class="label-tag">{html.escape(label)}</span>
                    </div>
                    <div class="text-content">{html.escape(text)}</div>
                </div>
                """)
        elif not missed_gts:
            content.append("<p>All ground truth items were successfully found!</p>")

        content.append('</div>')  # End Column 2
        content.append('</div>')  # End Container
        content.append("</body></html>")

        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

    def _get_color(self, css_class):
        colors = {
            "strict": "#28a745",
            "substring": "#6c757d",
            "ai-match": "#007bff",
            "wrong": "#dc3545",
            "missed": "#ffc107"
        }
        return colors.get(css_class, "#ccc")