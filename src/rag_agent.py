import os
from pathlib import Path

import chromadb
from agno.agent import Agent
from agno.models.google import Gemini

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
        The most relevant passages from the knowledge base.
    """
    results = collection.query(query_texts=[query], n_results=3)
    formatted = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        formatted.append(f"[Source: {metadata['source']}]\n{doc}")
    return "\n\n---\n\n".join(formatted)


rag_agent = Agent(
    name="AI-DJ Expert",
    model=Gemini(
        id="gemini-2.5-flash", vertexai=True,
        project_id=PROJECT_ID, location=LOCATION
    ),
    tools=[search_knowledge_base],
    instructions=(
        "You are an AI-DJ project expert. "
        "Use the knowledge base to answer questions. "
        "Always cite the source document. "
        "Use search_knowledge_base for any question about the AI-DJ  "
        "project itself (dataset, EDA, model, deployment). "
    ),
    markdown=True,
)


if __name__ == "__main__":
    rag_agent.print_response(
        "Which model was selected as the production model for the AI-DJ "
        "Hit Predictor, what is its test F1 score, and how does it handle "
        "the class imbalance in the training data?",
        stream=True,
    )
