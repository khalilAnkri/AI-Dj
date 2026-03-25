import pickle

import pandas as pd
from importDataset import importBigQuerry
from sklearn.ensemble import RandomForestClassifier
from trainingPreprocess import trainingPreprocess

df = importBigQuerry()

X, y = trainingPreprocess(df)

model = RandomForestClassifier(n_estimators=100, max_depth=None,
                               random_state=42, class_weight="balanced")

model.fit(X,y) # Evaluation separated in another file

importance = model.feature_importances_

featureImportance = pd.DataFrame({ "feature": X.columns,
                                   "importance": importance})

featureImportance = featureImportance.sort_values(by="importance", ascending=False)

listTop5 = featureImportance["feature"].head(5).tolist()

# Save and export model for deployment
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

with open("top_5.pkl", "wb") as f:
    pickle.dump(listTop5, f)
