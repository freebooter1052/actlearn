import numpy as np
import scipy.stats as stats

def extract_features(X):
    features = []
    for segment in X:
        mean, std = np.mean(segment), np.std(segment)
        min_val, max_val, median = np.min(segment), np.max(segment), np.median(segment)
        energy = np.sum(segment ** 2)
        p2p = max_val - min_val
        skew, kurt = stats.skew(segment), stats.kurtosis(segment)
        zcr = ((segment[:-1] * segment[1:]) < 0).sum()
        diff = np.diff(segment)
        diff_mean, diff_std = np.mean(diff), np.std(diff)
        features.append([mean, std, min_val, max_val, median, energy, p2p, skew, kurt, zcr, diff_mean, diff_std])
    return np.array(features)

def get_combined_features(X):
    stats_features = extract_features(X)
    return np.hstack([X, stats_features])
