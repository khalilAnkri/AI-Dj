"""
INFO9023 - Machine Learning Systems Design - Spotify Hit Predictor

Training and testing of rf model.

Team AI-DJ : 
    - Michon Charlotte
    - Mohamed-Khalil Ankri
    - Paulis Antoine
"""

import pandas as pd
from evaluate import evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

gcs_path = "gs://ai-dj-487610-bucket/data/uploaded_file.csv"
df = pd.read_csv(gcs_path)

df["hit"] = (df["popularity"] >= 65).astype(int)
df = df.drop(columns=["popularity"])
df = df.drop(columns=["Unnamed: 0"])

X = df.drop("hit", axis=1)
X = X.select_dtypes(include=["number"])
y = df["hit"]

# Stratify=y so that the proportion of 92% flops and 8% hits are kept in the train and test set.
XTrain, XTest, yTrain, yTest = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

bestN_estimators = 50
bestMax_depth = None
bestF1 = -1

for n_estimators in [50, 100, 200, 500]:
    for max_depth in [None, 5, 10, 20, 30]:
        print("n_estimators = "+str(n_estimators) +
              " and max_depth = "+str(max_depth))

        # class_weight="balanced" so that the model give more attention to the hit class
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=42, class_weight="balanced")

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

print("bestN_estimators = "+str(bestN_estimators) +
      " and bestMax_depth = "+str(bestMax_depth))

model = RandomForestClassifier(n_estimators=bestN_estimators, max_depth=bestMax_depth,
                               random_state=42, class_weight="balanced")

model.fit(XTrain, yTrain)
yPred = model.predict(XTest)

print("test accuracy = "+str(accuracy_score(yTest, yPred)))
print("test recall = "+str(recall_score(yTest, yPred)))
print("test f1 = "+str(f1_score(yTest, yPred)))
print("test precision = "+str(precision_score(yTest, yPred)))

"""
bestN_estimators = 100 and bestMax_depth = None
test accuracy = 0.9519298245614035
test recall = 0.5501231093914879
test f1 = 0.6554903604358759
test precision = 0.8107827890098497
"""
