import streamlit as st
import requests
import os

st.set_page_config(page_title="AI-DJ Predictor", page_icon="🎵")


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/predict")

st.title("🎵 AI-DJ: Spotify Hit Predictor")
st.markdown("Enter a song link to see if it has viral potential.")

song_url = st.text_input("Paste Spotify Track URL", placeholder="https://open.spotify.com/track/...")

if st.button("Predict Hit Potential"):
    if not song_url:
        st.warning("Please provide a Spotify URL first!")
    else:
        payload = {"url": song_url}
        
        try:
            with st.spinner("Analyzing song..."):
                response = requests.post(BACKEND_URL, json=payload, timeout=10)
                
            if response.status_code == 200:
                result = response.json()
                st.success(f"### Prediction: {result['prediction']}")
                st.write(f"**Confidence Score:** {result.get('confidence', 0)*100:.1f}%")
                
                with st.expander("Show Technical Details"):
                    st.json(result)
            else:
                st.error(f"Backend Error: Status code {response.status_code}")
        except Exception as e:
            st.error(f"Could not connect to Backend: {e}")
            st.info(f"Targeting Backend at: {BACKEND_URL}")