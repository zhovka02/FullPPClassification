import pandas as pd
import os
import sys

# --- Configuration ---
# The path to your benchmark results CSV file.
CSV_FILE_PATH = '/Users/aleksey/PycharmProjects/FullPPClassification/benchmark_full_results.csv'

# The path to the directory containing your report files.
# PLEASE VERIFY THIS PATH IS CORRECT.
REPORTS_DIR = '/Users/aleksey/PycharmProjects/FullPPClassification/reports/'

# Set to False to actually delete files.
# When True, the script will only print which files would be deleted.
DRY_RUN = False

def sanitize_model_name(model_name: str) -> str:
    """Sanitizes the model name to be filesystem-friendly."""
    return model_name.replace(':', '_').replace('/', '_')

def get_valid_report_basenames(csv_path: str) -> set:
    """Reads the benchmark CSV and returns a set of valid report basenames."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
        return set()

    valid_basenames = set()
    for _, row in df.iterrows():
        policy_id = row['policy_id']
        model = row['model']
        sanitized_model = sanitize_model_name(model)
        basename = f"{policy_id}_{sanitized_model}"
        valid_basenames.add(basename)
    return valid_basenames

def clean_reports_directory(reports_dir: str, valid_basenames: set, dry_run: bool = True):
    """
    Deletes files from the reports directory that are not in the
    set of valid report basenames.
    """
    if not os.path.isdir(reports_dir):
        print(f"Error: Reports directory not found at {reports_dir}", file=sys.stderr)
        return

    print(f"Scanning directory: {reports_dir}")
    
    for filename in os.listdir(reports_dir):
        basename, _ = os.path.splitext(filename)
        
        if basename not in valid_basenames:
            file_path = os.path.join(reports_dir, filename)
            if dry_run:
                print(f"[DRY RUN] Would delete: {file_path}")
            else:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except OSError as e:
                    print(f"Error deleting file {file_path}: {e}", file=sys.stderr)

if __name__ == '__main__':
    valid_names = get_valid_report_basenames(CSV_FILE_PATH)
    if valid_names:
        clean_reports_directory(REPORTS_DIR, valid_names, dry_run=DRY_RUN)
    print("\nScript finished.")
    if DRY_RUN:
        print("This was a DRY RUN. No files were actually deleted.")
        print("To delete files, set DRY_RUN = False in the script and run again.")