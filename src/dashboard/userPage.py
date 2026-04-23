import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from importDataset import importBigQuerry
from trainingPreprocess import trainingPreprocess
from sklearn.ensemble import RandomForestClassifier
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="16bf378ec05a4f00aab59c1d882319f1",
    client_secret="97f0c3c9017e45f8831e87ff4b7fae73"
))

# Not needed part just to make the interactive page
# ---------------------------------------------------------------------------
df = importBigQuerry()

X, y = trainingPreprocess(df)

model = RandomForestClassifier(n_estimators=10, max_depth=5,
                               random_state=42, class_weight="balanced")
# ----------------------------------------------------------------------------


st.set_page_config(page_title="AI-DJ Hit Predictor")

st.title("AI-DJ Hit Predictor")
st.image("spotify.jpg")

st.subheader("Do you want to know if our favorite track is a hit ?")

track_url = st.text_input("Enter our Spotify track URL", placeholder="https://open.spotify.com/track/...")

if track_url:
    track_id = track_url.split("/track/")[1]
    
    track = sp.track(track_id)
    
    name = track["name"]
    artist = track["artists"][0]["name"]
    album = track["album"]["name"]

    cover_url = track["album"]["images"][0]["url"]
    
    st.markdown("### Track information")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(cover_url, width=180)

    with col2:
        st.write(f"**Artist:** {artist}")
        st.write(f"**Album name:** {album}")
        st.write(f"**Track name:** {name}")

    features = sp.audio_features(track_id)[0]
    
    input = [[
        features["duration_ms"],
        features["danceability"],
        features["energy"],
        features["key"],
        features["loudness"],
        features["mode"],
        features["speechiness"],
        features["acousticness"],
        features["instrumentalness"],
        features["liveness"],
        features["valence"],
        features["tempo"],
        features["time_signature"]
    ]]
    
    prediction = model.predict(input)
    
    st.markdown("### Prediction")

    if prediction == 1:
        st.success("HIT")
    else:
        st.error("FLOP")

    categories = ['acousticness', 'duration_ms', 'loudness', 'energy', 'valence']
    keyFeatures = [features["acousticness"], features["duration_ms"], features["loudness"], features["energy"], features["valence"]]

    st.markdown("### Key features")

    # Use of claude to make the plot

    scaler = MinMaxScaler()
    normalized = [
        keyFeatures[0],
        keyFeatures[1] / 300000,
        (keyFeatures[2] + 60) / 60,
        keyFeatures[3],
        keyFeatures[4],
    ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    normalized += normalized[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    ax.fill(angles, normalized, color='green', alpha=0.4)
    ax.plot(angles, normalized, color='green', linewidth=2)
    ax.set_xticks(angles[:N])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    st.pyplot(fig)