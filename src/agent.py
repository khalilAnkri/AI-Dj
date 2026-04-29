import os
from pathlib import Path

import chromadb
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

PROJECT_ID = "ai-dj-487610"
LOCATION = "europe-west1"
MODEL_ID = "gemini-2.5-flash"

# knowledge base at the project root
KB_DIR = Path(__file__).resolve().parent.parent / "kb"

# --- RAG setup ---
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="ai_dj_knowledge")

# Load documents into ChromaDB
def load_documents(directory):

    documents = []
    metadatas = []
    ids = []

    for filename in os.listdir(directory):
        if not filename.endswith(".md"):
            continue
        with open(os.path.join(directory, filename)) as f:
            content = f.read()

        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk": i})
            ids.append(f"{filename}_{i}")

    if not documents:
        raise RuntimeError(f"No markdown files found in {directory}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Loaded {len(documents)} chunks from {directory}")


load_documents(KB_DIR)


def search_knowledge_base(query: str) -> str:
    """Search the AI-DJ project knowledge base.

    The kb contains documents about the project overview, EDA findings,
    modelling methodology (3 comparison) and the deployment architecture.

    Args:
        query: A natural language search query.

    Returns:
        The most relevant passages, each prefixed with its source file.
    """
    results = collection.query(query_texts=[query], n_results=3)
    formatted = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        formatted.append(f"[Source: {metadata['source']}]\n{doc}")
    return "\n\n---\n\n".join(formatted)


def analyze_track_features(
    danceability: float,
    energy: float,
    valence: float,
    tempo: float,
    acousticness: float,
) -> str:
    """Analyze Spotify audio features and estimate hit potential.

    Combines five Spotify audio features into a heuristic hit-likelihood score
    aligned with the AI-DJ project's findings (popularity >= 65 == hit).
    Use this tool whenever the user gives you Spotify audio features and asks
    whether the track sounds like a hit, or wants a musical interpretation
    of the numbers.

    Args:
        danceability: 0.0 to 1.0. How suitable for dancing the track is.
        energy: 0.0 to 1.0. Perceptual measure of intensity and activity.
        valence: 0.0 to 1.0. Musical positiveness (1.0 = happy, 0.0 = sad).
        tempo: Tempo in beats per minute (BPM). Typical pop range 90-130.
        acousticness: 0.0 to 1.0. 1.0 = strong confidence the track is acoustic.

    Returns:
        A short paragraph with a hit-probability score in [0, 1] and a
        natural-language interpretation of the track's profile.
    """
    # Each component is in [0, 1] and contributes to the heuristic.
    danceability_score = max(0.0, min(1.0, danceability))
    energy_score = max(0.0, min(1.0, energy))
    valence_score = max(0.0, min(1.0, valence))

    # Sweet spot for mainstream pop tempo is roughly 100-130 BPM.
    if 100.0 <= tempo <= 130.0:
        tempo_score = 1.0
    elif 80.0 <= tempo < 100.0 or 130.0 < tempo <= 150.0:
        tempo_score = 0.6
    else:
        tempo_score = 0.3

    # High acousticness usually correlates with lower mainstream popularity.
    acoustic_penalty = max(0.0, min(1.0, 1.0 - acousticness))

    hit_probability = (
        0.30 * danceability_score
        + 0.25 * energy_score
        + 0.20 * valence_score
        + 0.15 * tempo_score
        + 0.10 * acoustic_penalty
    )
    hit_probability = round(hit_probability, 3)

    if hit_probability >= 0.70:
        verdict = "strong hit potential"
    elif hit_probability >= 0.50:
        verdict = "moderate hit potential"
    else:
        verdict = "low hit potential"

    mood = "upbeat" if valence_score >= 0.6 else "moody"
    drive = "high-energy" if energy_score >= 0.6 else "laid-back"

    return (
        f"Hit probability: {hit_probability} ({verdict}). "
        f"The track is {mood} and {drive}, with danceability={danceability_score:.2f}, "
        f"tempo={tempo:.0f} BPM (tempo_score={tempo_score:.2f}) and "
        f"acousticness={acousticness:.2f}. Note: this is a heuristic, the "
        f"AI-DJ KNN model (n_neighbors=2000) reaches a test F1 of 0.67 and "
        f"should be the source of truth for production scoring."
    )


agent = Agent(
    model=Gemini(
        id=MODEL_ID, vertexai=True, project_id=PROJECT_ID, location=LOCATION
    ),
    tools=[
        DuckDuckGoTools(),
        YFinanceTools(
            enable_stock_price=True,
            enable_company_info=True,
            enable_company_news=True,
        ),
        analyze_track_features,
        search_knowledge_base,
    ],
    instructions=(
        "You are an AI-DJ assistant. You help producers, curators and music "
        "researchers reason about Spotify tracks and the AI-DJ Hit Predictor "
        "project. Use search_knowledge_base for any question about the AI-DJ "
        "project itself (dataset, EDA, model, deployment). Use "
        "analyze_track_features when the user provides Spotify audio features. "
        "Use DuckDuckGo for current music news and YFinance for music-industry "
        "stocks (e.g. SPOT, WMG, UMG). Always cite your sources."
    ),
    markdown=True,
)


if __name__ == "__main__":
    # Quick smoke test that exercises every tool.
    agent.print_response(
        "I'm evaluating a track with danceability=0.82, energy=0.74, "
        "valence=0.65, tempo=118 BPM and acousticness=0.08. "
        "Does it sound like a hit according to your custom analyzer, "
        "and what does the AI-DJ project documentation say about how "
        "the production model was selected?",
        stream=True,
    )
