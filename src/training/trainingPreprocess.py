"""
INFO9023 - Machine Learning Systems Design - Spotify Hit Predictor

Preprocess of the dataset for the training

Team AI-DJ : 
    - Michon Charlotte
    - Mohamed-Khalil Ankri
    - Paulis Antoine
"""

def trainingPreprocess(df):
    df["hit"] = (df["popularity"] >= 65).astype(int)
    df = df.drop(columns=["popularity"])
    df = df.drop(columns=["Unnamed: 0"])

    X = df.drop("hit", axis=1)
    X = X.select_dtypes(include=["number"])
    y = df["hit"]
    
    return X, y