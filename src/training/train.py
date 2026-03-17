import pandas as pd
from importDataset import importBigQuerry
from trainingPreprocess import trainingPreprocess
from sklearn.ensemble import RandomForestClassifier

df = importBigQuerry()

X, y = trainingPreprocess(df)

model = RandomForestClassifier(n_estimators=100, max_depth=None,
                               random_state=42, class_weight="balanced")

model.fit(X,y)

importance = model.feature_importances_

featureImportance = pd.DataFrame({ "feature": X.columns,
                                   "importance": importance})

featureImportance = featureImportance.sort_values(by="importance", ascending=False)

listTop5 = featureImportance["feature"].head(5).tolist()