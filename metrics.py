import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression
from utils import generate_clusterization_data, split_data


def accuracy(targets, predictions):
    return np.equal(targets, predictions).mean()

def roc_сurve(y_true: np.ndarray[int], y_score: np.ndarray[float]):
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

if __name__ == "__main__":
    """ROC-AUC with LogReg example"""
    X_train, y_train = generate_clusterization_data(n_clusters = 2, n_samples = 100)
    X_train, X_test, y_train, y_test =  split_data(X_train, y_train, ratio = 0.25)

    model = LogisticRegression(n_iterations=1000)
    losses = model.fit(X_train, y_train[:, None])
    y_score = model.predict(X_test).squeeze(1)
    
    y_pred = np.where(y_score >= 0.5, 1, 0)
    print(f"accuracy: {accuracy(y_test, y_pred) * 100}%")

    fpr, tpr, thresholds = roc_сurve(y_test, y_score)
    print(f"AUC: {auc(fpr, tpr)}")
 
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.show()

