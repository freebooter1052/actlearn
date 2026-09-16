# Active Learning to Reduce Annotation Cost in ECG Classification

## Objective
This project implements a segment-level active learning system for ECG arrhythmia classification using the MIT-BIH Arrhythmia Database. The core research question is whether active learning (Uncertainty and Margin sampling) can achieve better classification performance with fewer labeled ECG segments than a random sampling baseline.

## Project Structure
- `data/`: Contains the downloaded MIT-BIH dataset and preprocessed `.npy` files.
- `src/`: Python source code.
- `results/`: CSV outputs, Markdown report, and figures/tables.
- `notebooks/`: Jupyter notebooks.
- `config.py`: Global configuration parameters.
- `run_experiment.py`: Main entry point.

## modAL Dependency Note
The original instructions recommended using the `modAL` library if compatible. To maximize compatibility with modern scikit-learn versions and avoid potential dependency breaks, the query strategies (Random, Uncertainty, Margin) have been successfully implemented manually using scikit-learn's `predict_proba` interface. The methodology and algorithmic behaviors perfectly match those found in `modAL`.

## Setup and Installation
```bash
pip install -r requirements.txt
```

## Running the Experiments
To reproduce the complete pipeline run:
```bash
python run_experiment.py --strategy all
```
