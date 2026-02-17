import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import StratifiedKFold

def evaluation(model, XTrain, yTrain, n_folds=5):
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

        model.fit(XTrainFold, yTrainFold)
        yPred = model.predict(XValFold)
        
        accuracy = accuracy_score(yValFold, yPred)
        recall = recall_score(yValFold, yPred)
        f1 = f1_score(yValFold, yPred)
        precision = precision_score(yValFold, yPred, zero_division=0)
        
        accuracies.append(accuracy)
        recalls.append(recall)
        f1s.append(f1)
        precisions.append(precision)
    
    return np.mean(accuracies), np.mean(recalls), np.mean(f1s), np.mean(precisions)

df = pd.read_csv('../data/raw/dataset.csv')

df["hit"] = (df["popularity"] >= 65).astype(int) # Or true/False
df = df.drop(columns=["popularity"])
df = df.drop(columns=["Unnamed: 0"])

X = df.drop("hit", axis=1)
X = X.select_dtypes(include=["number"])
y = df["hit"]

# Stratify=y so that the proportion of 92% flops and 8% hits are kept in the train and test set.
XTrain, XTest, yTrain, yTest = train_test_split(X, y,test_size=0.3,random_state=42,stratify=y)

bestN_estimators = 100
bestMax_depth = 3
bestLearning_rate = 0.05
bestF1 = -1

for learning_rate in [0.05, 0.1, 0.2]:
    for n_estimators in [100, 200, 500]:
        for max_depth in [3, 5]:
            print("n_estimators = "+str(n_estimators)+", max_depths = "+str(max_depth)+
                  " and learning_rate = "+str(learning_rate))
            
            model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                               learning_rate=learning_rate, random_state=42)

            accuracy, recall, f1, precision = evaluation(model, XTrain, yTrain)
            
            print("mean accuracy = "+str(accuracy))
            print("mean recall = "+str(recall))
            print("mean f1 = "+str(f1))
            print("mean precision = "+str(precision))
            print("---------------------------------------------------------------------")
            
            if f1 > bestF1:
                bestF1 = f1
                bestN_estimators = n_estimators
                bestMax_depth = max_depth
                bestLearning_rate = learning_rate

print("bestN_estimators = "+str(bestN_estimators)+", bestMax_depth = "+str(bestMax_depth)+
      " and bestLearning_rate = "+str(bestLearning_rate))

model = GradientBoostingClassifier(n_estimators=bestN_estimators, max_depth=bestMax_depth, 
                                               learning_rate=bestLearning_rate, random_state=42)

model.fit(XTrain, yTrain)
yPred = model.predict(XTest)

print("test accuracy = "+str(accuracy_score(yTest, yPred)))
print("test recall = "+str(recall_score(yTest, yPred)))
print("test f1 = "+str(f1_score(yTest, yPred)))
print("test precision = "+str(precision_score(yTest, yPred, zero_division=0)))

"""
bestN_estimators = 500, bestMax_depth = 5 and bestLearning_rate = 0.2
test accuracy = 0.9330116959064327
test recall = 0.25923320436158986
test f1 = 0.39150066401062417
test precision = 0.7993492407809111
"""