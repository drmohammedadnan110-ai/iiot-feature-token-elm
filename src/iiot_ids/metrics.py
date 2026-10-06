"""Fixed-class metrics, binary collapse, and validation-fitted calibration."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import average_precision_score, confusion_matrix, matthews_corrcoef, roc_auc_score


def fit_temperature(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    if scores.ndim != 2 or len(scores) != len(labels) or not len(labels) or not np.isfinite(scores).all():
        raise ValueError("Require finite, aligned validation scores")

    def objective(log_t):
        scaled = scores / np.exp(log_t)
        return np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels])

    fit = minimize_scalar(objective, bounds=(-6, 6), method="bounded")
    candidates = [0.0, -6.0, 6.0, float(fit.x)]
    return float(np.exp(min(candidates, key=objective)))


def evaluate(labels, probabilities, classes, normal="Normal", bins=15):
    y, p = np.asarray(labels, int), np.asarray(probabilities, float)
    if p.shape != (len(y), len(classes)) or not len(y) or not np.isfinite(p).all():
        raise ValueError("Invalid probability shape or values")
    if (p < 0).any() or not np.allclose(p.sum(axis=1), 1) or (y < 0).any() or (y >= len(classes)).any():
        raise ValueError("Invalid probabilities or labels")
    pred = p.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(len(classes)))
    tp, n, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    recall = np.divide(tp, n, out=np.zeros(len(classes), float), where=n > 0)
    precision = np.divide(tp, predicted, out=np.zeros(len(classes), float), where=predicted > 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros(len(classes)), where=precision + recall > 0
    )
    confidence, correct = p.max(1), pred == y
    bin_ids = np.minimum((confidence * bins).astype(int), bins - 1)
    reliability, ece = [], 0.0
    for b in range(bins):
        mask = bin_ids == b
        accuracy, conf = (
            (float(correct[mask].mean()), float(confidence[mask].mean())) if mask.any() else (0.0, 0.0)
        )
        ece += mask.mean() * abs(accuracy - conf)
        reliability.append(
            {
                "lower": b / bins,
                "upper": (b + 1) / bins,
                "n": int(mask.sum()),
                "accuracy": accuracy,
                "confidence": conf,
            }
        )
    roc, pr = [], []
    for k in range(len(classes)):
        if 0 < (y == k).sum() < len(y):
            roc.append(float(roc_auc_score(y == k, p[:, k])))
            pr.append(float(average_precision_score(y == k, p[:, k])))
        else:
            roc.append(None)
            pr.append(None)
    result = {
        "n": len(y),
        "classes": classes,
        "accuracy": correct.mean(),
        "macro_f1": f1.mean(),
        "balanced_accuracy": recall.mean(),
        "weighted_f1": float(f1 @ n / n.sum()),
        "confusion_matrix": cm,
        "per_class": [
            {
                "class": c,
                "support": int(n[i]),
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "roc_auc": roc[i],
                "pr_auc": pr[i],
            }
            for i, c in enumerate(classes)
        ],
        "macro_roc_auc": float(np.mean(roc)) if all(v is not None for v in roc) else None,
        "macro_pr_auc": float(np.mean(pr)) if all(v is not None for v in pr) else None,
        "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-300, 1)).mean()),
        "brier": float(np.square(p - np.eye(len(classes))[y]).sum(1).mean()),
        "ece": float(ece),
        "bins": reliability,
    }
    if normal in classes:
        i = classes.index(normal)
        actual, guessed = y != i, pred != i
        tn, fp, fn, tp_b = confusion_matrix(actual, guessed, labels=[False, True]).ravel()
        divide = lambda a, b: float(a / b) if b else None
        result["binary_argmax_collapse"] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp_b),
            "accuracy": float((actual == guessed).mean()),
            "precision": divide(tp_b, tp_b + fp),
            "recall": divide(tp_b, tp_b + fn),
            "specificity": divide(tn, tn + fp),
            "fpr": divide(fp, fp + tn),
            "fnr": divide(fn, fn + tp_b),
            "f1": divide(2 * tp_b, 2 * tp_b + fp + fn),
            "mcc": float(matthews_corrcoef(actual, guessed)),
            "attack_score": "1 - P(Normal); hard labels collapse multiclass argmax",
        }
    return result


def macro_f1(labels, predictions, classes):
    cm = confusion_matrix(labels, predictions, labels=np.arange(classes))
    denominator = cm.sum(0) + cm.sum(1)
    return float(
        np.divide(2 * np.diag(cm), denominator, out=np.zeros(classes, float), where=denominator > 0).mean()
    )


def probabilities(scores, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    return softmax(np.asarray(scores, float) / temperature, axis=1)
