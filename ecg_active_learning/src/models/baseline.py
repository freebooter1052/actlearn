from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import CLASSIFIER, N_ESTIMATORS, RANDOM_SEED

def get_baseline_model():
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])
