"""
INFO9023 - Machine Learning Systems Design - Spotify Hit Predictor

Evaluation of classifier

Team AI-DJ :
    - Michon Charlotte
    - Mohamed-Khalil Ankri
    - Paulis Antoine
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def metricsEval(yValFold, yPred):
    """
    This function computes some metrics of model evaluation given a set of
    true output and estimated output.

    Args:
        yValFold: set of true output.
        yPred: set of estimated output.

    Returns:
        accuracy: accuracy.
        recall: recall.
        f1: f1 score.
        precision: precision.
    """
    accuracy = accuracy_score(yValFold, yPred)
    recall = recall_score(yValFold, yPred)
    f1 = f1_score(yValFold, yPred)
    precision = precision_score(yValFold, yPred, zero_division=0)
    return accuracy, recall, f1, precision

def evaluation(model, XTrain, yTrain, scale, n_folds=5):
    """
    This function evalutes the accuracy, recall, f1 and precision of a model by the use
    a cross validation.

    Args:
        model: the sklearn classifier.
        XTrain: panda data frame containing the inputs.
        yTrain: panda data frame containing the output.
        scale: if the inputs need to be scale. 
        n_folds: the number of folds of the cross validation.

    Returns:
        meanAccuracy: the mean accuracy of the cross validation.
        meanRecall: the mean recall of the cross validation.
        meanF1: the mean f1 of the cross validation.
        meanPrecision: the mean precision of the cross validation.
    """
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accuracies = []
    recalls = []
    f1s = []
    precisions = []
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(XTrain, yTrain), 1):
        print("fold "+str(fold_num))
        XTrainFold = XTrain.iloc[train_idx]
        yTrainFold = yTrain.iloc[train_idx]
        XValFold = XTrain.iloc[val_idx]
        yValFold = yTrain.iloc[val_idx]

        if scale:
            scaler = StandardScaler()
            XTrainFold = scaler.fit_transform(
                XTrainFold)  # learn the scale and apply
            # just apply the scale based on the previous one
            XValFold = scaler.transform(XValFold)

        model.fit(XTrainFold, yTrainFold)
        yPred = model.predict(XValFold)

        accuracy, recall, f1, precision = metricsEval(yValFold, yPred)

        accuracies.append(accuracy)
        recalls.append(recall)
        f1s.append(f1)
        precisions.append(precision)

    meanAccuracy = np.mean(accuracies)
    meanRecall = np.mean(recalls)
    meanF1 = np.mean(f1s)
    meanPrecision = np.mean(precisions)

    return meanAccuracy, meanRecall, meanF1, meanPrecision
