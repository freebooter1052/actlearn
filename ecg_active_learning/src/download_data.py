import wfdb
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

def download_mitbih():
    mitbih_dir = DATA_DIR / "mit-bih"
    if not mitbih_dir.exists():
        mitbih_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MIT-BIH database to {mitbih_dir} ...")
    wfdb.dl_database('mitdb', dl_dir=str(mitbih_dir))
    print("Download complete.")

if __name__ == "__main__":
    download_mitbih()
