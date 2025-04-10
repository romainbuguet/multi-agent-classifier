from argparse import ArgumentParser
import pandas as pd

def evaluate_accuracy(input_path: str) -> float:
    """
    Evaluates accuracy of the classification system.

    Args:
        path_to_csv (str): Path to the CSV file with 'category' and 'constructiveness_label' columns.

    Returns:
        float: Accuracy as a percentage.
    """
    df = pd.read_csv(input_path)

    if 'category' not in df.columns or 'constructiveness_label' not in df.columns:
        raise ValueError("CSV must contain 'category' and 'constructiveness_label' columns.")

    total = len(df)
    correct = (df['category'].str.strip().str.lower() == df['constructiveness_label'].str.strip().str.lower()).sum()

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct predictions)")
    return accuracy

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the CSV file (must contain 'category' and 'constructiveness_label' columns).")
    args = parser.parse_args()

    evaluate_accuracy(args.input_path)
