import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import KFold, train_test_split

def evaluation(model, XTrain, yTrain, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    accuracies = []
    recalls = []
    f1s = []
    precisions = []
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(XTrain), 1):
        print("fold "+str(fold_num))
        XTrainFold = XTrain.iloc[train_idx] 
        yTrainFold = yTrain.iloc[train_idx]
        XValFold = XTrain.iloc[val_idx]
        yValFold = yTrain.iloc[val_idx]
        
        scaler = StandardScaler()
        XTrainFold = scaler.fit_transform(XTrainFold) # learn the scale and apply
        XValFold = scaler.transform(XValFold) # just apply the scale based on the previous one

        model.fit(XTrainFold, yTrainFold)
        yPred = model.predict(XValFold)
        
        
        accuracy = accuracy_score(yValFold, yPred)
        recall = recall_score(yValFold, yPred)
        f1 = f1_score(yValFold, yPred)
        precision = precision_score(yValFold, yPred)
        
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

XTrain, XTest, yTrain, yTest = train_test_split(X, y,test_size=0.3,random_state=42,stratify=y)

bestN_neighbors = 5
bestF1 = -1

for n_neighbors in [5, 10, 100, 200, 500, 750, 1000, 1500, 2000]:
    print("n_neighbors = "+str(n_neighbors))
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")

    accuracy, recall, f1, precision = evaluation(model, XTrain, yTrain)
    print("mean accuracy = "+str(accuracy))
    print("mean recall = "+str(recall))
    print("mean f1 = "+str(f1))
    print("mean precision = "+str(precision))
    print("---------------------------------------------------------------------")
    if f1 > bestF1:
        bestF1 = f1
        bestN_neighbors = n_neighbors

print("bestN_neighbors = "+str(bestN_neighbors))

model = KNeighborsClassifier(n_neighbors=bestN_neighbors, weights="distance")

scaler = StandardScaler()
XTrainScaled = scaler.fit_transform(XTrain)
XTestScaled = scaler.transform(XTest)

model.fit(XTrainScaled, yTrain)
yPred = model.predict(XTestScaled)


print("test accuracy = "+str(accuracy_score(yTest, yPred)))
print("test recall = "+str(recall_score(yTest, yPred)))
print("test f1 = "+str(f1_score(yTest, yPred)))
print("test precision = "+str(precision_score(yTest, yPred)))

"""
bestN_neighbors = 1500
test accuracy = 0.9438304093567251
test recall = 0.9438304093567251
test f1 = 0.5574752361207095
test precision = 0.976594027441485
"""