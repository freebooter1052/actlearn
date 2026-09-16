import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, TEST_SIZE, VALIDATION_SIZE, RANDOM_SEED, MAX_TRAIN_POOL_SIZE

def load_data():
    processed_dir = DATA_DIR / "processed"
    return (np.load(processed_dir / "X.npy"), np.load(processed_dir / "y.npy"),
            np.load(processed_dir / "record_ids.npy"), np.load(processed_dir / "beat_positions.npy"))

def split_records(records, test_size=TEST_SIZE, val_size=VALIDATION_SIZE, seed=RANDOM_SEED):
    np.random.seed(seed)
    unique_records = np.unique(records)
    np.random.shuffle(unique_records)
    n_total = len(unique_records)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    return unique_records[n_test+n_val:], unique_records[n_test:n_test+n_val], unique_records[:n_test]

def get_data_splits():
    X, y, rec_ids, positions = load_data()
    train_recs, val_recs, test_recs = split_records(rec_ids)

    train_mask, val_mask, test_mask = np.isin(rec_ids, train_recs), np.isin(rec_ids, val_recs), np.isin(rec_ids, test_recs)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_train) > MAX_TRAIN_POOL_SIZE:
        print(f"Subsampling training pool from {len(X_train)} to {MAX_TRAIN_POOL_SIZE} samples for simulation speed.")
        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(len(X_train), MAX_TRAIN_POOL_SIZE, replace=False)
        X_train, y_train = X_train[indices], y_train[indices]

    return (X_train, y_train, train_recs), (X_val, y_val, val_recs), (X_test, y_test, test_recs)
