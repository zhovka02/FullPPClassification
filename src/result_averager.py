import pandas as pd
import os
import sys

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Error: Plotly is not installed. Please run 'pip install plotly' to generate the HTML report.")
    sys.exit(1)

def generate_average_report(input_csv="benchmark_full_results.csv", output_html="benchmark_average_report.html"):
    """
    Reads benchmark results, filters out invalid rows (all stats 0.0),
    averages metrics per model, and generates an HTML report with graphs.
    """
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)

    # --- Data Cleaning ---
    # Identify "failed" rows. A row is failed if metrics are all 0.
    # The user specified that rows with data errors have all stats as 0.0.
    # We must exclude the policies containing these errors from ALL models to ensure fair comparison.
    
    # Check for 0.0 in key metrics
    df['is_failed'] = (
        (df['f1'] == 0.0) & 
        (df['precision'] == 0.0) & 
        (df['recall'] == 0.0) & 
        (df['ai_f1'] == 0.0) &
        (df['ai_precision'] == 0.0) &
        (df['ai_recall'] == 0.0)
    )

    # Identify policies that have ANY failed row
    failed_policies = df[df['is_failed']]['policy_id'].unique()
    
    print(f"Total policies before filtering: {df['policy_id'].nunique()}")
    if len(failed_policies) > 0:
        print(f"Policies with failures (to be excluded): {len(failed_policies)}")
        print(f"Failed Policy IDs: {list(failed_policies)}")
    else:
        print("No failed policies found.")

    # Filter the dataframe to exclude these policies entirely
    clean_df = df[~df['policy_id'].isin(failed_policies)].copy()
    
    print(f"Total policies after filtering: {clean_df['policy_id'].nunique()}")

    if clean_df.empty:
        print("Error: No data left after filtering.")
        return

    # --- Aggregation ---
    # Group by model and calculate mean of metrics
    metrics = ['precision', 'recall', 'f1', 'ai_precision', 'ai_recall', 'ai_f1', 'duration_sec']
    leaderboard = clean_df.groupby('model')[metrics].mean().reset_index()
    
    # Sort by AI F1
    leaderboard = leaderboard.sort_values('ai_f1', ascending=False)

    print("\n--- Leaderboard (Averaged) ---")
    print(leaderboard)

    # --- Visualization ---
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("AI F1 vs Strict F1", "AI Precision vs Recall", "Duration (sec)", "Detailed Metrics Table"),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "table"}]]
    )

    # 1. AI F1 vs Strict F1 (Bar Chart)
    fig.add_trace(go.Bar(
        x=leaderboard['model'], 
        y=leaderboard['ai_f1'], 
        name='AI F1',
        marker_color='royalblue',
        text=leaderboard['ai_f1'].round(3),
        textposition='auto'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=leaderboard['model'], 
        y=leaderboard['f1'], 
        name='Strict F1',
        marker_color='lightgray',
        text=leaderboard['f1'].round(3),
        textposition='auto'
    ), row=1, col=1)

    # 2. AI Precision vs Recall (Grouped Bar)
    fig.add_trace(go.Bar(
        x=leaderboard['model'], 
        y=leaderboard['ai_precision'], 
        name='AI Precision',
        marker_color='forestgreen'
    ), row=1, col=2)
    
    fig.add_trace(go.Bar(
        x=leaderboard['model'], 
        y=leaderboard['ai_recall'], 
        name='AI Recall',
        marker_color='orange'
    ), row=1, col=2)

    # 3. Duration
    fig.add_trace(go.Bar(
        x=leaderboard['model'], 
        y=leaderboard['duration_sec'], 
        name='Avg Duration (s)',
        marker_color='firebrick',
        text=leaderboard['duration_sec'].round(1),
        textposition='auto'
    ), row=2, col=1)

    # 4. Table
    display_df = leaderboard.round(3)
    
    fig.add_trace(go.Table(
        header=dict(values=list(display_df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[display_df[k].tolist() for k in display_df.columns],
                   fill_color='lavender',
                   align='left')
    ), row=2, col=2)

    # Layout updates
    fig.update_layout(
        title_text=f"Benchmark Results Summary (N={clean_df['policy_id'].nunique()} policies)",
        height=1000,
        showlegend=True,
        barmode='group'
    )

    # Save to HTML
    fig.write_html(output_html)
    print(f"\nReport generated: {os.path.abspath(output_html)}")

if __name__ == "__main__":
    generate_average_report("../benchmark_full_results.csv", "../benchmark_average_report.html")
