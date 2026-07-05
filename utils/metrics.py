import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, accuracy_score


def compute_metrics(y_true, y_prob, average="macro"):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=average),
        "ck": cohen_kappa_score(y_true, y_pred),
    }
    try:
        out["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
    except Exception:
        out["auc"] = float("nan")
    return out
