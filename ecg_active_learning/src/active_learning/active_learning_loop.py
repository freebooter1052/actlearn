import numpy as np
import copy
from src.active_learning.random_sampling import random_sampling_strategy
from src.active_learning.uncertainty_sampling import uncertainty_sampling_strategy
from src.active_learning.margin_sampling import margin_sampling_strategy

STRATEGIES = {'random': random_sampling_strategy, 'uncertainty': uncertainty_sampling_strategy, 'margin': margin_sampling_strategy}

def active_learning_loop(model, X_train, y_train, X_test, y_test, eval_fn, strategy_name, initial_ratio, batch_size, max_ratio):
    strategy_fn = STRATEGIES[strategy_name]
    n_total = len(X_train)
    n_initial = int(n_total * initial_ratio)
    labeled_idx = np.random.choice(n_total, n_initial, replace=False)
    pool_idx = np.setdiff1d(np.arange(n_total), labeled_idx)
    results = []
    iteration = 0
    while True:
        current_model = copy.deepcopy(model)
        current_model.fit(X_train[labeled_idx], y_train[labeled_idx])
        y_pred = current_model.predict(X_test)
        metrics = eval_fn(y_test, y_pred)
        metrics['iteration'], metrics['labeled_samples'], metrics['annotation_percentage'] = iteration, len(labeled_idx), len(labeled_idx) / n_total * 100
        results.append(metrics)
        if len(labeled_idx) >= int(n_total * max_ratio) or len(pool_idx) == 0:
            break
        current_batch_size = min(batch_size, len(pool_idx))
        query_idx_rel = strategy_fn(current_model, X_train[pool_idx], current_batch_size)
        query_idx_abs = pool_idx[query_idx_rel]
        labeled_idx = np.concatenate([labeled_idx, query_idx_abs])
        pool_idx = np.delete(pool_idx, query_idx_rel)
        iteration += 1
        print(f"[{strategy_name}] Iteration {iteration}: {len(labeled_idx)} labeled samples.")
    return results, current_model
