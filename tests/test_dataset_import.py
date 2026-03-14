from google.cloud import bigquery

def test_dataset_import():
    client = bigquery.Client()
    query = """
    SELECT *
    FROM `ai-dj-487610.Spotify_Tracks.tracks`
    """
    df = client.query(query).to_dataframe()
    
    columns = ["track_id","artists","album_name","track_name","popularity","duration_ms","explicit","danceability",
        "energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence",
        "tempo","time_signature","track_genre"]
    
    for column in columns:
        assert column in df.columns