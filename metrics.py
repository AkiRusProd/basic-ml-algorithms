import numpy as np


def accuracy(targets, predictions):
    return np.equal(targets, predictions).mean()

def roc_сurve(y_true: np.ndarray[int], y_score: np.ndarray[float]):
    # https://www.youtube.com/watch?v=4jRBRDbJemM
    sorted_indices = np.argsort(y_score)[::-1]

    y_true = y_true[sorted_indices]
    y_score = y_score[sorted_indices]

    tp = 0
    fp = 0
    tn = len(y_true) - np.sum(y_true)
    fn = np.sum(y_true)

    fpr = []
    tpr = []
    thresholds = []

    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1
        fpr.append(fp / (fp + tn) if fp + tn > 0 else 0) #exception if all y_true are 1
        tpr.append(tp / (tp + fn) if tp + fn > 0 else 0) #exception if all y_true are 0
        thresholds.append(y_score[i])
       
    return fpr, tpr, thresholds

def auc(fpr: np.ndarray[float], tpr: np.ndarray[float]):
    return np.trapz(tpr, fpr)

