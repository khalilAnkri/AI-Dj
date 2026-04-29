import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from importDataset import importBigQuerry
from trainingPreprocess import trainingPreprocess
from sklearn.ensemble import RandomForestClassifier

def radar_creation(keyFeatures):
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

df = importBigQuerry()

# Not needed part just to make the interactive page
# ---------------------------------------------------------------------------
X, y = trainingPreprocess(df)

model = RandomForestClassifier(n_estimators=10, max_depth=5,
                               random_state=42, class_weight="balanced")
model.fit(X,y)
# ----------------------------------------------------------------------------

st.set_page_config(page_title="AI-DJ Hit Predictor")

st.title("AI-DJ Hit Predictor")
st.image("spotify.jpg")

page = st.sidebar.selectbox("Select prediction", [
    "Existing tracks",
    "Customized tracks",
])

st.subheader("Do you want to know if our favorite track is a hit ?")

if page == "Existing tracks":
    st.write("Select among those a track_id among this set of tracks:")

    track_cols = ["track_id", "artists", "album_name", "track_name"]
    track_table = df[track_cols].drop_duplicates().reset_index(drop=True)

    st.dataframe(track_table, height=200)

    # --- Input ---
    track_id = st.text_input("Enter a track_id :")
        
    button = st.button("Predict")

    if track_id and button:
        track_row = df[df["track_id"] == track_id]
        
        if track_row.empty:
            st.warning("Unknown track_id")
        else:
            track = track_row.iloc[0]
            name = track["track_name"]
            artist = track["artists"]
            album = track["album_name"]
            
            st.subheader("Track information")

            st.write(f"**Artist:** {artist}")
            st.write(f"**Album name:** {album}")
            st.write(f"**Track name:** {name}")
            
            X,_ = trainingPreprocess(track_row)
            
            prediction = model.predict(X)[0]
            
            st.subheader("Prediction")

            if prediction == 1:
                st.success("HIT")
            else:
                st.error("FLOP")

            categories = ['acousticness', 'duration_ms', 'loudness', 'energy', 'valence']
            keyFeatures = [track["acousticness"], track["duration_ms"], track["loudness"], track["energy"], track["valence"]]

            st.subheader("Key features")

            radar_creation(keyFeatures)
            
elif page == "Customized tracks":
    st.write("Customize your own track")
    
    duration_ms = st.slider("Duration (ms)", 30000, 600000, 315000)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5, step=0.01)
    energy = st.slider("Energy", 0.0, 1.0, 0.5, step=0.01)
    key = st.slider("Key", -1, 11, 5)
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -30.0, step=0.1)
    time_signature = st.slider("Time signature", 3, 7, 5)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5, step=0.01)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, step=0.01)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5, step=0.01)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.5, step=0.01)
    valence = st.slider("Valence", 0.0, 1.0, 0.5, step=0.01)
    tempo = st.slider("Tempo (BPM)", 50.0, 250.0, 150.0, step=0.1)
    mode = 1 if st.checkbox("Major mode (uncheck for Minor)", value=True) else 0

    button = st.button("Predict")
    
    if button:
        X = [[
            duration_ms,
            danceability,
            energy,
            key,
            loudness,
            mode,
            speechiness,
            acousticness,
            instrumentalness,
            liveness,
            valence,
            tempo,
            time_signature
        ]]
        
        prediction = model.predict(X)[0]

        st.subheader("Prediction")
        
        if prediction == 1:
            st.success("HIT")
        else:
            st.error("FLOP")