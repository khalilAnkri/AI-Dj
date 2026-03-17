"""
INFO9023 - Machine Learning Systems Design - Spotify Hit Predictor

Import of the dataset from BigQuerry

Team AI-DJ : 
    - Michon Charlotte
    - Mohamed-Khalil Ankri
    - Paulis Antoine
"""
from google.cloud import bigquery

def importBigQuerry():
    client = bigquery.Client()

    query = """
    SELECT *
    FROM `ai-dj-487610.Spotify_Tracks.tracks`
    """

    df = client.query(query).to_dataframe()
    return df