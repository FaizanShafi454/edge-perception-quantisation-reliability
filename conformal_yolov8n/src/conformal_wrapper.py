import numpy as np

def prediction_set_from_scores(class_probs, alpha=0.1):
    class_probs = np.asarray(class_probs)
    order = np.argsort(class_probs)[::-1]
    sorted_probs = class_probs[order]
    cumulative = np.cumsum(sorted_probs)
    k = np.searchsorted(cumulative, 1 - alpha, side="left") + 1
    return order[:k].tolist()

def nonconformity_score(true_prob):
    return 1.0 - float(true_prob)
