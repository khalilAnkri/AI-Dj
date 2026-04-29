# Modelling methodology

## Train/test split

The dataset is split 70% train / 30% test. The split is stratified on the
binary `hit` target so that the 8.31% hit ratio is preserved in both subsets.

## Cross-validation

Hyperparameter selection on the training set uses `StratifiedKFold` with
`shuffle=True`, again to keep the class proportions inside each fold. The
selection criterion is the F1-score because the dataset is imbalanced:
accuracy would be misleading and F1 penalises models that trade precision for
recall (or vice versa) too aggressively.

## Algorithms compared

Three classifiers are trained and compared:

### K-Nearest Neighbors

`KNeighborsClassifier` with `weights="distance"` so that closer neighbours
have more influence on the vote. Inputs are scaled because audio features
live on different scales (`duration_ms` in milliseconds, `loudness` in dB,
others in [0, 1]). The best `n_neighbors` found by cross-validation is 2000,
yielding a test F1 of 0.67.

### Random Forest

`RandomForestClassifier` with `class_weight="balanced"` to compensate for the
8.31% hit ratio by up-weighting the minority class during training. The best
hyperparameters are `n_estimators=100` and `max_depth=None`, with a test F1
of 0.65.

### Gradient Boosting

The best gradient boosting configuration is `learning_rate=500`,
`n_estimators=5`, `max_depth=0.2` with a test F1 of 0.39. The drop in
performance is consistent with the very small `n_estimators` selected by
cross-validation.

## Final choice

The selected production model is **KNN with `n_neighbors=2000` and
`weights="distance"`**, on scaled inputs, with a test F1 of 0.67.
