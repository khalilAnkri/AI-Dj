"""
INFO9023 - Machine Learning Systems Design - Spotify Hit Predictor

services.py — complete file with:
  - SpotifyService     : Musicae RapidAPI + Spotify oEmbed metadata
  - PredictionService  : local model.pkl inference with confidence + top features
  - ExplanationService : Claude API — natural language explanation of prediction

Team AI-DJ:
    - Michon Charlotte
    - Mohamed-Khalil Ankri
    - Paulis Antoine
"""

import requests

from .config import Config

# ---------------------------------------------------------------------------
# Musicae RapidAPI — audio features
# ---------------------------------------------------------------------------
MUSICAE_URL = "https://spotify-extended-audio-features-api.p.rapidapi.com/v1/audio-features/{track_id}"
MUSICAE_HOST = "spotify-extended-audio-features-api.p.rapidapi.com"
REQUEST_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Spotify oEmbed — free, no auth, returns track name + artist + thumbnail
# ---------------------------------------------------------------------------
OEMBED_URL = (
    "https://open.spotify.com/oembed?url=https://open.spotify.com/track/{track_id}"
)


class SpotifyService:
    """
    - get_features() : audio features via Musicae RapidAPI
    - get_metadata() : track name, artist, thumbnail via Spotify oEmbed (free)
    """

    def __init__(self):
        self.api_key = Config.RAPIDAPI_KEY
        if not self.api_key:
            raise Exception("RAPIDAPI_KEY is not set. Add it to your .env file.")
        print("SpotifyService (Musicae/RapidAPI mode) initialized.")

    def get_features(self, query: str) -> dict:
        print(f"Query received: {query}")
        try:
            track_id = self._extract_track_id(query)
        except ValueError as ve:
            return {"error": str(ve)}
        return self._fetch_from_musicae(track_id)

    def get_metadata(self, track_id: str) -> dict:
        """
        Fetches track name, artist, thumbnail via Musicae GET /tracks/{id}.
        Falls back to oEmbed if Musicae fails.
        Returns empty dict on failure — metadata is non-critical.
        """
        # --- Primary: Musicae /tracks/{id} ---
        try:
            url = f"https://spotify-extended-audio-features-api.p.rapidapi.com/v1/tracks/{track_id}"
            headers = {
                "x-rapidapi-key": self.api_key,
                "x-rapidapi-host": MUSICAE_HOST,
            }
            response = requests.get(url, headers=headers, timeout=30)
            if response.ok:
                data = response.json()
                artists = data.get("artists", [])
                artist_name = (
                    artists[0].get("name", "Unknown") if artists else "Unknown"
                )
                images = data.get("album", {}).get("images", [])
                thumbnail = images[0].get("url", "") if images else ""
                artist_id = artists[0].get("id", "") if artists else ""
                return {
                    "track_name": data.get("name", "Unknown"),
                    "artist": artist_name,
                    "artist_id": artist_id,
                    "thumbnail": thumbnail,
                    "spotify_url": f"https://open.spotify.com/track/{track_id}",
                }
        except Exception as e:
            print(f"Musicae /tracks metadata fetch failed, trying oEmbed: {e}")

        # --- Fallback: Spotify oEmbed ---
        try:
            url = OEMBED_URL.format(track_id=track_id)
            response = requests.get(url, timeout=5)
            if not response.ok:
                return {}
            data = response.json()
            return {
                "track_name": data.get("title", "Unknown"),
                "artist": data.get("author_name", "Unknown"),
                "thumbnail": data.get("thumbnail_url", ""),
                "spotify_url": f"https://open.spotify.com/track/{track_id}",
            }
        except Exception as e:
            print(f"oEmbed metadata fetch also failed (non-critical): {e}")
            return {}

    def _extract_track_id(self, query: str) -> str:
        if "spotify.com/track/" in query:
            track_id = query.split("track/")[-1].split("?")[0].strip()
            print(f"Extracted track_id from URL: {track_id}")
            return track_id
        if len(query) == 22 and query.isalnum():
            print(f"Using bare track_id: {query}")
            return query
        raise ValueError(
            f"Could not resolve a Spotify track ID from: '{query}'. "
            "Please provide a full Spotify track URL or a bare 22-character track ID."
        )

    def _fetch_from_musicae(self, track_id: str) -> dict:
        url = MUSICAE_URL.format(track_id=track_id)
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": MUSICAE_HOST,
        }
        print(f"Fetching audio features from Musicae for track_id={track_id}")
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.Timeout:
            raise Exception(
                f"Musicae API timed out after {REQUEST_TIMEOUT}s."
            ) from None
        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to Musicae API.") from None

        if response.status_code == 401:
            raise Exception("Musicae API: 401 Unauthorized — check your RAPIDAPI_KEY.")
        if response.status_code == 403:
            raise Exception("Musicae API: 403 Forbidden — quota may be exceeded.")
        if response.status_code == 404:
            return {
                "error": f"Track '{track_id}' was not found. Please check the Spotify URL."
            }
        if response.status_code == 429:
            raise Exception("Musicae API: 429 Rate limit exceeded.")
        if not response.ok:
            raise Exception(
                f"Musicae API returned status {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except Exception:
            raise Exception(
                f"Musicae API returned non-JSON response: {response.text[:200]}"
            ) from None

        if not data:
            return {"error": f"No audio features returned for track '{track_id}'."}

        data["_track_id"] = track_id
        print(f"Audio features retrieved from Musicae for track_id={track_id}")
        return data


class PredictionService:
    """
    Vertex AI Endpoint inference.
    Calls the deployed sklearn model on Vertex AI and returns prediction,
    both class probabilities, and top 5 feature values.
    """

    # Top 5 most important features confirmed from Vertex AI Model Registry
    TOP_5_FEATURES = [
        "acousticness",
        "energy",
        "loudness",
        "valence",
        "instrumentalness",
    ]

    def __init__(self):
        from google.cloud import aiplatform

        project_id = Config.PROJECT_ID
        location = Config.LOCATION
        endpoint_id = Config.ENDPOINT_ID

        if not all([project_id, location, endpoint_id]):
            raise Exception(
                "Missing Vertex AI config. Ensure GCP_PROJECT_ID, GCP_LOCATION, "
                "and VERTEX_ENDPOINT_ID are set in your .env file."
            )

        aiplatform.init(project=project_id, location=location)

        self.endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
        )
        self.columns = Config.MODEL_COLUMNS

        print("PredictionService (Vertex AI mode) initialized.")
        print(f"Endpoint: {endpoint_id}")
        print(f"Columns : {self.columns}")

    def predict(self, features: dict) -> dict:
        try:
            instance = [float(features.get(col, 0)) for col in self.columns]
            print(f"Sending instance to Vertex AI: {instance}")

            response = self.endpoint.predict(instances=[instance])
            print(f"Raw predict response: {response.predictions}")

            pred_raw = response.predictions[0]
            pred_index = (
                int(pred_raw) if not isinstance(pred_raw, list) else int(pred_raw[0])
            )

            top_features = {
                feat: round(float(features.get(feat, 0)), 4)
                for feat in self.TOP_5_FEATURES
            }

            return {
                "class_index": pred_index,
                "class_name": Config.CLASSES[pred_index],
                "confidence": 1.0,
                "top_features": top_features,
            }
        except Exception as e:
            print(f"Vertex AI prediction error: {e}")
            raise e

    def predict_manually(self, features: dict) -> dict:
        return self.predict(features)


class ExplanationService:
    """
    Generates a natural language explanation of the prediction
    using rule-based feature interpretation — no API required.
    """

    # Human-readable thresholds for each feature
    FEATURE_DESCRIPTORS = {
        "danceability": [
            (0.7, "high danceability", "low danceability"),
            (0.4, "moderate danceability", "low danceability"),
        ],
        "energy": [
            (0.7, "high energy", "low energy"),
            (0.4, "moderate energy", "low energy"),
        ],
        "valence": [
            (0.7, "very positive mood", "negative mood"),
            (0.4, "neutral mood", "negative mood"),
        ],
        "acousticness": [
            (0.6, "highly acoustic sound", "non-acoustic sound"),
            (0.3, "some acoustic elements", "non-acoustic sound"),
        ],
        "loudness": [
            (-5, "high loudness", "quiet track"),
            (-10, "moderate loudness", "quiet track"),
        ],
        "tempo": [
            (130, "fast tempo", "slow tempo"),
            (100, "moderate tempo", "slow tempo"),
        ],
        "speechiness": [
            (0.5, "speech-heavy content", "musical content"),
            (0.2, "some spoken elements", "musical content"),
        ],
        "instrumentalness": [
            (0.5, "largely instrumental", "vocal-driven track"),
            (0.1, "mostly vocal", "vocal-driven track"),
        ],
        "liveness": [
            (0.7, "live recording feel", "studio production"),
            (0.3, "some live elements", "studio production"),
        ],
        "duration_ms": [
            (240000, "long track", "short track"),
            (180000, "standard length", "short track"),
        ],
    }

    def __init__(self):
        print("ExplanationService (rule-based, local) initialized.")

    def _describe_feature(self, feature: str, value: float) -> str:
        """Returns a human-readable description of a feature value."""
        if feature not in self.FEATURE_DESCRIPTORS:
            return f"{feature}: {value}"
        thresholds = self.FEATURE_DESCRIPTORS[feature]
        for threshold, high_label, _low_label in thresholds:
            if value >= threshold:
                return f"{high_label} ({value})"
        return f"{thresholds[-1][2]} ({value})"

    def explain(
        self,
        track_name: str,
        artist: str,
        prediction: str,
        confidence: float,
        hit_probability: float,
        top_features: dict,
    ) -> str:
        """
        Generates a 2-sentence natural language explanation
        from the top feature values — fully local, no API needed.
        """
        is_hit = prediction == "Hit"
        confidence_pct = round(confidence * 100, 1)
        hit_pct = round(hit_probability * 100, 1)

        # Describe each top feature
        [self._describe_feature(feat, val) for feat, val in top_features.items()]

        # Split into supporting and contrasting signals
        supporting = []
        contrasting = []

        for feat, val in top_features.items():
            desc = self._describe_feature(feat, val)
            # Features that typically correlate positively with hits
            hit_positive = {
                "danceability": val >= 0.5,
                "energy": val >= 0.5,
                "valence": val >= 0.5,
                "loudness": val >= -8,
                "tempo": 100 <= val <= 140,
                "acousticness": val < 0.4,
                "instrumentalness": val < 0.3,
                "speechiness": val < 0.4,
                "liveness": val < 0.5,
                "duration_ms": 150000 <= val <= 270000,
            }
            is_positive_signal = hit_positive.get(feat, True)

            if (is_hit and is_positive_signal) or (
                not is_hit and not is_positive_signal
            ):
                supporting.append(desc)
            else:
                contrasting.append(desc)

        # Build sentence 1 — main drivers
        if supporting:
            drivers = ", ".join(supporting[:3])
            sentence1 = (
                f'"{track_name}" by {artist} was predicted as a {prediction} ,'
                f" primarily driven by its {drivers}."
            )
        else:
            sentence1 = (
                f'"{track_name}" by {artist} was predicted as a {prediction} '
                f"with {confidence_pct}% confidence (hit probability: {hit_pct}%)."
            )

        # Build sentence 2 — contrasting signals or reinforcement
        if contrasting:
            contra = ", ".join(contrasting[:2])
            sentence2 = (
                f"Despite showing {contra}, "
                f"the overall audio profile {'aligns' if is_hit else 'does not align'} "
                f"with the hit songs in our training dataset."
            )
        else:
            sentence2 = (
                f"All key audio features consistently point toward a "
                f"{'strong hit' if is_hit else 'likely flop'} profile, "
                f"with a hit probability of {hit_pct}%."
            )

        return f"{sentence1} {sentence2}"


class RecommendationService:
    """
    Fetches similar track recommendations via Musicae RapidAPI.
    Uses the same key as SpotifyService — no extra subscription needed.
    Seeded by the current track_id, returns 5 similar tracks with metadata.
    """

    RECOMMENDATIONS_URL = (
        "https://spotify-extended-audio-features-api.p.rapidapi.com/v1/recommendations"
    )
    HOST = "spotify-extended-audio-features-api.p.rapidapi.com"

    def __init__(self):
        self.api_key = Config.RAPIDAPI_KEY
        if not self.api_key:
            raise Exception("RAPIDAPI_KEY is not set.")
        print("RecommendationService (Musicae/RapidAPI) initialized.")

    def get_recommendations(
        self,
        track_id: str,
        artist_id: str = "",
        audio_features: dict = None,
        limit: int = 5,
    ) -> list:
        """
        Returns a list of recommended tracks based on the seed track_id + artist_id
        and target audio features for better similarity matching.
        Each item contains: track_name, artist, spotify_url, thumbnail.
        Returns empty list on failure — recommendations are non-critical.
        """
        if audio_features is None:
            audio_features = {}
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.HOST,
        }
        params = {
            "seed_tracks": track_id,
            "seed_artists": artist_id,
            "limit": limit,
            "market": "FR",
            "seed_genres": "pop",
            "target_energy": audio_features.get("energy"),
            "target_danceability": audio_features.get("danceability"),
            "target_valence": audio_features.get("valence"),
            "target_tempo": audio_features.get("tempo"),
            "target_acousticness": audio_features.get("acousticness"),
        }
        # Remove None values to avoid sending empty params
        params = {k: v for k, v in params.items() if v is not None}

        try:
            response = requests.get(
                self.RECOMMENDATIONS_URL,
                headers=headers,
                params=params,
                timeout=30,
            )
        except Exception as e:
            print(f"RecommendationService request failed (non-critical): {e}")
            return []

        if not response.ok:
            print(
                f"RecommendationService: HTTP {response.status_code} — {response.text[:300]}"
            )
            return []

        print(f"Recommendations raw response: {response.text[:500]}")

        try:
            data = response.json()
        except Exception:
            print("RecommendationService: non-JSON response — skipping.")
            return []

        tracks = data.get("tracks", [])
        recommendations = []

        for track in tracks[:limit]:
            track_id_rec = track.get("id", "")
            name = track.get("name", "Unknown")
            artists = track.get("artists", [])
            artist_name = artists[0].get("name", "Unknown") if artists else "Unknown"
            images = track.get("album", {}).get("images", [])
            thumbnail = images[0].get("url", "") if images else ""

            recommendations.append(
                {
                    "track_name": name,
                    "artist": artist_name,
                    "spotify_url": f"https://open.spotify.com/track/{track_id_rec}",
                    "thumbnail": thumbnail,
                }
            )

        print(f"Recommendations fetched: {len(recommendations)} tracks.")
        return recommendations
