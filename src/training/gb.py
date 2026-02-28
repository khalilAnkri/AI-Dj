"""
INFO9023 - Machine Learning Systems Design - Spotify Hit Predictor

Training and testing of gb model.

Team AI-DJ : 
    - Michon Charlotte
    - Mohamed-Khalil Ankri
    - Paulis Antoine
"""

import pandas as pd
from evaluate import evaluation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('../../data/raw/dataset.csv')

df["hit"] = (df["popularity"] >= 65).astype(int)
df = df.drop(columns=["popularity"])
df = df.drop(columns=["Unnamed: 0"])

X = df.drop("hit", axis=1)
X = X.select_dtypes(include=["number"])
y = df["hit"]

# Stratify=y so that the proportion of 92% flops and 8% hits are kept in the train and test set.
XTrain, XTest, yTrain, yTest = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

bestN_estimators = 100
bestMax_depth = 3
bestLearning_rate = 0.05
bestF1 = -1

for learning_rate in [0.05, 0.1, 0.2]:
    for n_estimators in [100, 200, 500]:
        for max_depth in [3, 5]:
            print("n_estimators = "+str(n_estimators)+", max_depths = "+str(max_depth) +
                  " and learning_rate = "+str(learning_rate))

            model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                               learning_rate=learning_rate, random_state=42)

            accuracy, recall, f1, precision = evaluation(
                model, XTrain, yTrain, scale=False)

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

print("bestN_estimators = "+str(bestN_estimators)+", bestMax_depth = "+str(bestMax_depth) +
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
