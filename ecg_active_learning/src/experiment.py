import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import RESULTS_DIR, INITIAL_LABELED_RATIO, BATCH_SIZE, MAX_LABEL_RATIO, RANDOM_SEED, CLASSIFIER, MAX_TRAIN_POOL_SIZE
from src.dataset_split import get_data_splits
from src.feature_extraction import get_combined_features
from src.models.baseline import get_baseline_model
from src.active_learning.active_learning_loop import active_learning_loop
from src.evaluation import evaluate_model
from src.visualization import plot_learning_curves, plot_confusion_matrix, plot_ecg_examples
from src.preprocessing import preprocess_dataset
from src.download_data import download_mitbih

def calculate_label_efficiency(df, threshold=0.80, metric='macro_f1'):
    reached = df[df[metric] >= threshold]
    return int(reached.iloc[0]['labeled_samples']) if not reached.empty else "Not reached"

def generate_report(results_dict, cm_dict, split_info, data_stats):
    report_path = RESULTS_DIR / "final_report.md"
    with open(report_path, "w") as f:
        f.write("# Active Learning to Reduce Annotation Cost in ECG Classification\n\n")

        f.write("## 1. Dataset Statistics\n")
        f.write(f"- Total Samples: {data_stats['total_samples']}\n")
        f.write("- Class Distribution:\n")
        for cls, count in data_stats['class_counts'].items():
            f.write(f"  - Class {cls}: {count} samples\n")
        f.write("\n")

        f.write("## 2. Train/Validation/Test Splits\n")
        f.write(f"- Train Records ({len(split_info['train_recs'])}): {', '.join(split_info['train_recs'])}\n")
        f.write(f"- Validation Records ({len(split_info['val_recs'])}): {', '.join(split_info['val_recs'])}\n")
        f.write(f"- Test Records ({len(split_info['test_recs'])}): {', '.join(split_info['test_recs'])}\n\n")

        f.write("## 3. Experiment Setup\n")
        f.write(f"- **Classifier**: {CLASSIFIER}\n- **Max Training Pool Size**: {MAX_TRAIN_POOL_SIZE}\n")
        f.write(f"- **Initial Labeled Ratio**: {INITIAL_LABELED_RATIO}\n- **Batch Size**: {BATCH_SIZE}\n")
        f.write(f"- **Max Label Ratio**: {MAX_LABEL_RATIO}\n- **Random Seed**: {RANDOM_SEED}\n\n")

        f.write("## 4. Label Efficiency Analysis\nTarget Macro F1 = 0.80\n\n")
        f.write("| Strategy | Samples to reach 0.80 Macro F1 | Macro F1 @ 10% | Macro F1 @ 30% | Macro F1 @ 50% |\n")
        f.write("|----------|--------------------------------|----------------|----------------|----------------|\n")

        for strategy, df in results_dict.items():
            eff = calculate_label_efficiency(df, threshold=0.80)
            def get_metric_at_pct(pct):
                idx = (df['annotation_percentage'] - pct).abs().idxmin()
                return df.loc[idx, 'macro_f1']
            f1_10, f1_30, f1_50 = get_metric_at_pct(10), get_metric_at_pct(30), get_metric_at_pct(50)
            f.write(f"| {strategy} | {eff} | {f1_10:.4f} | {f1_30:.4f} | {f1_50:.4f} |\n")

        f.write("\n## 5. Conclusion\n")
        f.write("The learning curves and label efficiency metrics show the performance of active learning vs random sampling. Uncertainty and Margin sampling strategies select the most informative segments, aiming to reach higher performance faster and save on annotation cost.\n")

def run_experiment(strategies):
    data_dir = Path(__file__).resolve().parent.parent / "data"
    mitbih_dir = data_dir / "mit-bih"
    processed_dir = data_dir / "processed"

    if not mitbih_dir.exists():
        print("MIT-BIH dataset not found. Downloading...")
        download_mitbih()

    if not processed_dir.exists() or not (processed_dir / "X.npy").exists():
        print("Processed data not found. Running preprocessing...")
        preprocess_dataset()

    # Get raw data and classes to extract stats
    y_full = np.load(processed_dir / "y.npy")
    unique_classes, class_counts = np.unique(y_full, return_counts=True)
    data_stats = {
        'total_samples': len(y_full),
        'class_counts': dict(zip(unique_classes, class_counts))
    }

    print("\nDataset Class Distribution:")
    for cls, count in data_stats['class_counts'].items():
        print(f"Class {cls}: {count} samples")

    (X_train_raw, y_train, train_recs), (X_val_raw, y_val, val_recs), (X_test_raw, y_test, test_recs) = get_data_splits()

    print(f"\nRecord Splits:")
    print(f"Train Records: {list(train_recs)}")
    print(f"Validation Records: {list(val_recs)}")
    print(f"Test Records: {list(test_recs)}\n")

    split_info = {
        'train_recs': list(train_recs),
        'val_recs': list(val_recs),
        'test_recs': list(test_recs)
    }

    X_train = get_combined_features(X_train_raw)
    X_test = get_combined_features(X_test_raw)

    classes = np.unique(y_test)
    for cls in classes:
        idx = np.where(y_test == cls)[0]
        if len(idx) > 0:
            plot_ecg_examples(X_test_raw[idx[:3]], y_test[idx[:3]], f"Example of {cls}", f"example_ecg_{cls}.png")

    results_dict, cm_dict = {}, {}
    for strategy in strategies:
        print(f"\n=========================================")
        print(f"Running Active Learning Strategy: {strategy}")
        print(f"=========================================\n")

        np.random.seed(RANDOM_SEED)
        model = get_baseline_model()
        results, final_model = active_learning_loop(
            model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            eval_fn=evaluate_model, strategy_name=strategy, initial_ratio=INITIAL_LABELED_RATIO,
            batch_size=BATCH_SIZE, max_ratio=MAX_LABEL_RATIO
        )
        df = pd.DataFrame(results)
        df.drop(columns=['confusion_matrix']).to_csv(RESULTS_DIR / f"{strategy}_sampling.csv", index=False)
        results_dict[strategy] = df
        final_cm = np.array(results[-1]['confusion_matrix'])
        cm_dict[strategy] = final_cm
        plot_confusion_matrix(final_cm, classes, f"Confusion Matrix: {strategy}", f"cm_{strategy}.png")

    plot_learning_curves(results_dict)
    generate_report(results_dict, cm_dict, split_info, data_stats)
    print("Experiment Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='all', choices=['random', 'uncertainty', 'margin', 'all'])
    args = parser.parse_args()
    strats_to_run = ['random', 'uncertainty', 'margin'] if args.strategy == 'all' else [args.strategy]
    run_experiment(strats_to_run)
