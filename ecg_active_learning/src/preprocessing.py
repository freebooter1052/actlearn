import wfdb
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, WINDOW_SIZE
from src.label_mapping import map_symbol_to_aami

def get_record_ids():
    mitbih_dir = DATA_DIR / "mit-bih"
    records = set()
    for file in os.listdir(mitbih_dir):
        if file.endswith('.dat'):
            records.add(file.split('.')[0])
    return sorted(list(records))

def preprocess_dataset():
    mitbih_dir = DATA_DIR / "mit-bih"
    record_ids = get_record_ids()
    half_window = WINDOW_SIZE // 2
    X_list, y_list, rec_ids_list, positions_list = [], [], [], []

    for rec in record_ids:
        rec_path = str(mitbih_dir / rec)
        try:
            record = wfdb.rdrecord(rec_path)
            annotation = wfdb.rdann(rec_path, 'atr')
        except Exception as e:
            continue

        channel_names = record.sig_name
        ch_idx = channel_names.index('MLII') if 'MLII' in channel_names else 0
        signal = record.p_signal[:, ch_idx]
        sig_len = len(signal)

        for pos, sym in zip(annotation.sample, annotation.symbol):
            aami_class = map_symbol_to_aami(sym)
            if aami_class is None:
                continue
            if pos - half_window < 0 or pos + half_window >= sig_len:
                continue

            segment = signal[pos - half_window : pos + half_window]
            seg_mean = np.mean(segment)
            seg_std = np.std(segment)
            segment = (segment - seg_mean) / seg_std if seg_std > 0 else segment - seg_mean

            X_list.append(segment)
            y_list.append(aami_class)
            rec_ids_list.append(rec)
            positions_list.append(pos)

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(exist_ok=True)
    np.save(processed_dir / "X.npy", np.array(X_list))
    np.save(processed_dir / "y.npy", np.array(y_list))
    np.save(processed_dir / "record_ids.npy", np.array(rec_ids_list))
    np.save(processed_dir / "beat_positions.npy", np.array(positions_list))
    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess_dataset()
