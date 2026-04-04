import pandas as pd
import os
import sys

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Error: Plotly is not installed. Please run 'pip install plotly' to generate the reports.")
    sys.exit(1)


def generate_average_report(input_csv="benchmark_full_results.csv", output_html="benchmark_average_report.html",
                            output_img_dir="../plots"):
    """
    Reads benchmark results, filters out invalid rows, averages metrics per model,
    generates an HTML report, and outputs PNGs for LaTeX.

    Args:
        input_csv (str): Path to the input CSV file containing benchmark results.
        output_html (str): Path to save the generated HTML report.
        output_img_dir (str): Directory to save the generated PNG plots.
    """
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)

    df['is_failed'] = (
            (df['f1'] == 0.0) &
            (df['precision'] == 0.0) &
            (df['recall'] == 0.0) &
            (df['ai_f1'] == 0.0) &
            (df['ai_precision'] == 0.0) &
            (df['ai_recall'] == 0.0)
    )

    failed_policies = df[df['is_failed']]['policy_id'].unique()

    print(f"Total policies before filtering: {df['policy_id'].nunique()}")
    if len(failed_policies) > 0:
        print(f"Policies with failures (to be excluded): {len(failed_policies)}")
        print(f"Failed Policy IDs: {list(failed_policies)}")
    else:
        print("No failed policies found.")

    clean_df = df[~df['policy_id'].isin(failed_policies)].copy()

    print(f"Total policies after filtering: {clean_df['policy_id'].nunique()}")

    if clean_df.empty:
        print("Error: No data left after filtering.")
        return

    metrics = ['precision', 'recall', 'f1', 'ai_precision', 'ai_recall', 'ai_f1', 'duration_sec']
    leaderboard = clean_df.groupby('model')[metrics].mean().reset_index()

    leaderboard = leaderboard.sort_values('ai_f1', ascending=False)

    print("\n--- Leaderboard (Averaged - Full Metrics) ---")
    print(leaderboard.to_string(index=False))
    print("-" * 50)

    main_color = '#70ad47'
    main_color_light = '#a9d18e'
    secondary_color = '#70ad47'
    tertiary_color = '#ed7d31'
    quaternary_color = '#ed7d31'

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("AI F1 vs Strict F1", "AI Precision vs Recall", "Duration (sec)", "Detailed Metrics Table"),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "table"}]]
    )

    fig.add_trace(go.Bar(x=leaderboard['model'], y=leaderboard['ai_f1'], name='AI F1', marker_color=main_color,
                         text=leaderboard['ai_f1'].round(3), textposition='auto'), row=1, col=1)
    fig.add_trace(go.Bar(x=leaderboard['model'], y=leaderboard['f1'], name='Strict F1', marker_color=main_color_light,
                         text=leaderboard['f1'].round(3), textposition='auto'), row=1, col=1)

    fig.add_trace(
        go.Bar(x=leaderboard['model'], y=leaderboard['ai_precision'], name='AI Precision', marker_color=secondary_color),
        row=1, col=2)
    fig.add_trace(go.Bar(x=leaderboard['model'], y=leaderboard['ai_recall'], name='AI Recall', marker_color=tertiary_color),
                  row=1, col=2)

    fig.add_trace(
        go.Bar(x=leaderboard['model'], y=leaderboard['duration_sec'], name='Avg Duration (s)', marker_color=quaternary_color,
               text=leaderboard['duration_sec'].round(1), textposition='auto'), row=2, col=1)

    display_df = leaderboard.round(3)
    fig.add_trace(go.Table(
        header=dict(values=list(display_df.columns), fill_color=main_color, font=dict(color='white'), align='left'),
        cells=dict(values=[display_df[k].tolist() for k in display_df.columns], fill_color=main_color_light, font=dict(color='black'), align='left')
    ), row=2, col=2)

    fig.update_layout(title_text=f"Benchmark Results Summary (N={clean_df['policy_id'].nunique()} policies)",
                      height=1000, showlegend=True, barmode='group')
    fig.write_html(output_html)
    print(f"\nHTML Report generated: {os.path.abspath(output_html)}")

    os.makedirs(output_img_dir, exist_ok=True)

    try:
        fig_f1 = go.Figure(data=[
            go.Bar(name='AI F1', x=leaderboard['model'], y=leaderboard['ai_f1'], marker_color=main_color),
            go.Bar(name='Strict F1', x=leaderboard['model'], y=leaderboard['f1'], marker_color=main_color_light)
        ])
        fig_f1.update_layout(title='AI F1 vs Strict F1', barmode='group', template='plotly_white')
        f1_path = os.path.join(output_img_dir, 'f1_comparison.png')
        fig_f1.write_image(f1_path, width=800, height=500, scale=2)
        print(f"Exported PNG: {f1_path}")

        fig_pr = go.Figure(data=[
            go.Bar(name='AI Precision', x=leaderboard['model'], y=leaderboard['ai_precision'],
                   marker_color=secondary_color),
            go.Bar(name='AI Recall', x=leaderboard['model'], y=leaderboard['ai_recall'], marker_color=tertiary_color)
        ])
        fig_pr.update_layout(title='AI Precision vs AI Recall', barmode='group', template='plotly_white')
        pr_path = os.path.join(output_img_dir, 'precision_recall.png')
        fig_pr.write_image(pr_path, width=800, height=500, scale=2)
        print(f"Exported PNG: {pr_path}")

        fig_dur = go.Figure(data=[
            go.Bar(name='Avg Duration (s)', x=leaderboard['model'], y=leaderboard['duration_sec'],
                   marker_color=quaternary_color)
        ])
        fig_dur.update_layout(title='Average Duration per Policy (Seconds)', template='plotly_white')
        dur_path = os.path.join(output_img_dir, 'duration.png')
        fig_dur.write_image(dur_path, width=800, height=500, scale=2)
        print(f"Exported PNG: {dur_path}")

    except ValueError as e:
        print("\nWarning: Could not export PNGs. Make sure 'kaleido' is installed (pip install -U kaleido).")
        print(f"Error details: {e}")


if __name__ == "__main__":
    generate_average_report("../benchmark_full_results.csv", "../benchmark_average_report.html", "../plots")