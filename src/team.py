from agno.models.google import Gemini
from agno.team import Team

from src.first_agent import finance_agent, web_agent

PROJECT_ID = "ai-dj-487610"
LOCATION = "europe-west1"
MODEL_ID = "gemini-2.5-flash"

team = Team(
    name="Research Team",
    mode="coordinate",
    model=Gemini(
        id="gemini-2.5-flash", vertexai=True,
        project_id=PROJECT_ID, location=LOCATION
    ),
    members=[
        web_agent,
        finance_agent
        ],
    instructions=(
        "You are the team leader of research team on AI-DJ Project. "
        "Delegate research (e.g. music news) to the Web Researcher "
        "and financial questions (e.g. music industry) to the Financial Analyst. "
        "Synthesize their findings into a final answer. "
        "Ensure to return factual, sourced answers and real numbers. "
        "Always cite the sources used, "
        "and double-check it before answering."
    ),
    markdown=True,
)

if __name__ == "__main__":
    team.print_response(
        "What is the current state of Spotify performance on the stock market "
        "right now? What track is the biggest hit?"
        "And what are the most recent news about the music industry?",
        stream=True,
    )
