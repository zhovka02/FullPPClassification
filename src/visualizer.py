import os
import html


class HTMLVisualizer:
    def __init__(self):
        self.header = """<!DOCTYPE html>
<html>
<head>
    <title>C3PA Annotation Audit</title>
    <style>
        body { font-family: "Segoe UI", sans-serif; padding: 20px; max-width: 1400px; margin: auto; background: #f5f5f5; }
        h2 { background: #e9ecef; padding: 10px; border-left: 5px solid #007bff; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; background: #fff; table-layout: fixed; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th, td { padding: 12px; border: 1px solid #dee2e6; vertical-align: top; overflow-wrap: break-word; }
        th.human-col { background: #d4edda; width: 45%; }
        th.llm-col { background: #f8d7da; width: 45%; }
        th.score-col { background: #e2e3e5; width: 10%; text-align: center; }
        .human-text { font-family: monospace; font-size: 13px; color: #155724; white-space: pre-wrap; }
        .llm-text { font-family: monospace; font-size: 13px; color: #721c24; white-space: pre-wrap; }
        tr.match { background-color: #fff3cd; }
        tr.miss { background-color: #fff; }
        tr.fn { background-color: #f8f9fa; border-left: 4px solid #dc3545; }
        .score-badge { font-weight: bold; padding: 2px 6px; border-radius: 4px; color: #fff; }
        .score-high { background-color: #28a745; }
        .score-med { background-color: #ffc107; color: #000; }
        .score-low { background-color: #dc3545; }
        .empty-cell { color: #aaa; font-style: italic; text-align: center; }
    </style>
</head>
<body>
    <h1>C3PA vs LLM Audit Report</h1>
"""
        self.footer = "</body></html>"

    def generate_report(self, policy_id, full_text, human_anns, llm_anns, filename="report.html"):
        content_html = f"<h3>Policy ID: {policy_id}</h3>"

        all_labels = sorted(list(set(h['label'] for h in human_anns).union(set(l['label'] for l in llm_anns))))

        for label in all_labels:
            # 1. Get items for this label
            lbl_human = [h for h in human_anns if h['label'] == label]
            lbl_llm = [l for l in llm_anns if l['label'] == label]

            if not lbl_human and not lbl_llm: continue

            content_html += f"<h2>{html.escape(label)}</h2>"
            content_html += "<table><thead><tr><th class='human-col'>Human Ground Truth</th><th class='llm-col'>LLM Prediction</th><th class='score-col'>Score</th></tr></thead><tbody>"

            # Track which human annotations have been 'displayed' via a match
            displayed_human_texts = set()

            # 2. Render LLM Predictions (and their best matches)
            for l_item in lbl_llm:
                matched_text = l_item.get('_matched_human_text', "")
                score = l_item.get('_match_score', 0.0)
                status = l_item.get('_match_status', 'Miss')

                if matched_text:
                    displayed_human_texts.add(matched_text)

                row_class = "match" if status == "Hit" else "miss"
                score_class = "score-high" if score > 0.8 else ("score-med" if score > 0.4 else "score-low")

                content_html += f"<tr class='{row_class}'>"
                # Show the Human text that caused the match score
                if matched_text:
                    content_html += f"<td><div class='human-text'>{html.escape(matched_text)}</div></td>"
                else:
                    content_html += "<td class='empty-cell'>No close match found</td>"

                content_html += f"<td><div class='llm-text'>{html.escape(l_item['text'])}</div></td>"
                content_html += f"<td style='text-align:center;'><span class='score-badge {score_class}'>{score:.2f}</span></td>"
                content_html += "</tr>"

            # 3. Render False Negatives (Human annotations that were NOT matched)
            for h_item in lbl_human:
                if h_item['text'] not in displayed_human_texts:
                    content_html += "<tr class='fn'>"
                    content_html += f"<td><div class='human-text'>{html.escape(h_item['text'])}</div></td>"
                    content_html += "<td class='empty-cell'>Not Extracted by LLM</td>"
                    content_html += "<td style='text-align:center;'>FN</td>"
                    content_html += "</tr>"

            content_html += "</tbody></table>"

        final_html = self.header + content_html + self.footer
        with open(filename, "w", encoding="utf-8") as f:
            f.write(final_html)