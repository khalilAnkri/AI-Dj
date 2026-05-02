from agno.models.google import Gemini
from agno.team import Team

# --- Specialist agents ---
from src.first_agent import analyze_agent, finance_agent, web_agent
from src.rag_agent import rag_agent

PROJECT_ID = "ai-dj-487610"
LOCATION = "europe-west1"
MODEL_ID = "gemini-2.5-flash"

# --- Team ---
team = Team(
    name="Expert Team",
    mode="coordinate",
    model=Gemini(
        id="gemini-2.5-flash", vertexai=True,
        project_id=PROJECT_ID, location=LOCATION
    ),
    members=[
        web_agent,
        finance_agent,
        analyze_agent,
        rag_agent
        ],
    instructions=(
        "You lead a team of experts on AI-DJ Project. "
        "Route questions to the right specialist; "
        "read the user's question, split it into parts and delegate: \n"
        "- For AI-DJ project topics, use the AI-DJ Expert. \n"
        "- For finance use the Financial Analyst. \n"
        "- For general research use the Web Researcher. \n"
        "- For mood classification of a track (from valence and energy), "
        "use the Mood Analyst. "
        "Synthesize their findings into a clear final answer. "
        "Ensure to return factual, sourced answers and real numbers. "
        "Always cite the sources used, "
        "and double-check it before answering."
    ),
    markdown=True,
)

# --- Test ---
if __name__ == "__main__":
    team.print_response(
        "I'm working on the AI-DJ project. What model was selected "
        "and why?",
    stream=True,
    )
    team.print_response(
        "Which song is currently the biggest global hit on Spotify, "
        "and how has Spotify's (SPOT) stock performed this week? "
        "Briefly comment on whether the platform's momentum aligns "
        "with the strength of its top hit.",
    stream=True,
    )
    team.print_response(
        "I'm evaluating a candidate track for the AI-DJ Hit Predictor "
        "with valence=0.78 and energy=0.82. (1) What mood quadrant does "
        "it fall into? (2) Which song is currently the biggest hit on "
        "Spotify globally, and does it share a similar mood profile? ",
    stream=True,
    )
