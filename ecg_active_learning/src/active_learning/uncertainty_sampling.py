import numpy as np

def uncertainty_sampling_strategy(model, X_pool, n_instances):
    probas = model.predict_proba(X_pool)
    uncertainty = 1.0 - np.max(probas, axis=1)
    return np.argsort(uncertainty)[-n_instances:]
