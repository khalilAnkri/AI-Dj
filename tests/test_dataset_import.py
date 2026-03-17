from importDataset import importBigQuerry

def test_dataset_import():
    
    df = importBigQuerry()
    
    columns = ["track_id","artists","album_name","track_name","popularity","duration_ms","explicit","danceability",
        "energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence",
        "tempo","time_signature","track_genre"]
    
    for column in columns:
        assert column in df.columns