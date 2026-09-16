import numpy as np

def random_sampling_strategy(model, X_pool, n_instances):
    return np.random.choice(len(X_pool), n_instances, replace=False)
