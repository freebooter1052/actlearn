import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR

def plot_learning_curves(results_dict):
    metrics = ['macro_f1', 'accuracy', 'weighted_f1', 'recall']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for strategy, df in results_dict.items():
            plt.plot(df['labeled_samples'], df[metric], marker='o', label=strategy)
        plt.title(f'Active Learning Performance: {metric.replace("_", " ").title()}')
        plt.xlabel('Number of Labeled ECG Samples')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'learning_curve_{metric}.png', dpi=300)
        plt.close()

def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

def plot_ecg_examples(X, y, title, filename, probas=None):
    fig, axes = plt.subplots(len(X), 1, figsize=(10, 2*len(X)))
    if len(X) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(X[i])
        lbl = f"True: {y[i]}" + (f" | Probas: {probas[i]}" if probas is not None else "")
        ax.set_title(lbl)
        ax.set_ylabel("Amplitude")
        ax.grid(True)
    plt.xlabel("Samples")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()
