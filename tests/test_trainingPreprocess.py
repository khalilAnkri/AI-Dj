"""
INFO9023 - Machine Learning Systems Design - Spotify Hit Predictor

PyTest related to trainingPreprocess.py

Team AI-DJ :
    - Michon Charlotte
    - Mohamed-Khalil Ankri
    - Paulis Antoine
"""

from importDataset import importBigQuerry
from trainingPreprocess import trainingPreprocess


def test_trainingPreprocess():
    """
    This function tests the trainingPreprocess function by checking if the X and y returned
    by trainingPreprocess contains the correct column.
    """

    df = importBigQuerry()
    X,y = trainingPreprocess(df)
    XColumns = ["duration_ms","danceability","energy","key","loudness","mode",
               "speechiness","acousticness","instrumentalness","liveness","valence","tempo",
               "time_signature"]

    for column in XColumns:
        assert column in X.columns

    assert "hit" in y.name
