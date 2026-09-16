import numpy as np

def margin_sampling_strategy(model, X_pool, n_instances):
    probas = model.predict_proba(X_pool)
    if probas.shape[1] < 2:
        return np.arange(n_instances)
    sorted_probas = np.sort(probas, axis=1)[:, ::-1]
    margins = sorted_probas[:, 0] - sorted_probas[:, 1]
    return np.argsort(margins)[:n_instances]
