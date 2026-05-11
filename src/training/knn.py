"""
INFO9023 - Machine Learning Systems Design - Spotify Hit Predictor

Training and testing of knn model.

Team AI-DJ :
    - Michon Charlotte
    - Mohamed-Khalil Ankri
    - Paulis Antoine
"""

from evaluate import evaluation
from importDataset import importBigQuerry
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from trainingPreprocess import trainingPreprocess

df = importBigQuerry()

X, y = trainingPreprocess(df)

# Stratify=y so that the proportion of 92% flops and 8% hits are kept in the train and test set.
XTrain, XTest, yTrain, yTest = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

bestN_neighbors = 5
bestF1 = -1

for n_neighbors in [5, 10, 100, 200, 500, 750, 1000, 1500, 2000]:
    print("n_neighbors = " + str(n_neighbors))
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")

    accuracy, recall, f1, precision = evaluation(model, XTrain, yTrain, scale=True)
    print("mean accuracy = " + str(accuracy))
    print("mean recall = " + str(recall))
    print("mean f1 = " + str(f1))
    print("mean precision = " + str(precision))
    print("---------------------------------------------------------------------")
    if f1 > bestF1:
        bestF1 = f1
        bestN_neighbors = n_neighbors

print("bestN_neighbors = " + str(bestN_neighbors))

model = KNeighborsClassifier(n_neighbors=bestN_neighbors, weights="distance")

scaler = StandardScaler()
XTrainScaled = scaler.fit_transform(XTrain)
XTestScaled = scaler.transform(XTest)

model.fit(XTrainScaled, yTrain)
yPred = model.predict(XTestScaled)


print("test accuracy = " + str(accuracy_score(yTest, yPred)))
print("test recall = " + str(recall_score(yTest, yPred)))
print("test f1 = " + str(f1_score(yTest, yPred)))
print("test precision = " + str(precision_score(yTest, yPred)))

"""
bestN_neighbors = 2000
test accuracy = 0.9575730994152046
test recall = 0.5230390432641576
test f1 = 0.672090395480226
test precision = 0.9399494310998736
"""
