import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from importDataset import importBigQuerry
from trainingPreprocess import trainingPreprocess
from sklearn.ensemble import RandomForestClassifier

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

st.subheader("Do you want to know if our favorite track is a hit ?")

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