import numpy as np
from sklearn.metrics import precision_score, recall_score


def f1_score(actuals, predictions):
    precision = precision_score(actuals, predictions, zero_division=0)
    recall = recall_score(actuals, predictions)
    if precision * recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def f05_score(actuals, predictions):
    precision = precision_score(actuals, predictions, zero_division=0)
    recall = recall_score(actuals, predictions)
    if precision * recall == 0:
        return 0
    return (1.25 * precision * recall) / (0.25 * precision * recall)


# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def f_scores(probabilities, actuals):
    thresholds = np.arange(0, 1, 0.001)
    scores = [f1_score(actuals, to_labels(probabilities, i)) for i in thresholds]
    ix = np.argmax(scores)
    f1_thresholds = thresholds[ix]
    f1_f_score = scores[ix]
    scores = [f05_score(actuals, to_labels(probabilities, i)) for i in thresholds]
    ix = np.argmax(scores)
    f05_threshold = thresholds[ix]
    f05_f_score = scores[ix]
    return f1_f_score, f1_thresholds, f05_f_score, f05_threshold


def pr_metrics(probabilities, threshold, actuals):
    predictions = to_labels(probabilities, threshold)
    precision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    return precision, recall
