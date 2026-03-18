"""
INFO9023 - Machine Learning Systems Design - Spotify Hit Predictor

PyTest related to dataset_import.py

Team AI-DJ : 
    - Michon Charlotte
    - Mohamed-Khalil Ankri
    - Paulis Antoine
"""
from importDataset import importBigQuerry

def test_dataset_import():
    """
    This function tests the importBigQuerry function by checking if the df returned
    by importBigQuerry contains the correct column.
    """
    
    df = importBigQuerry()
    
    columns = ["track_id","artists","album_name","track_name","popularity","duration_ms","explicit","danceability",
        "energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence",
        "tempo","time_signature","track_genre"]
    
    for column in columns:
        assert column in df.columns