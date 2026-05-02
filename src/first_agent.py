from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

PROJECT_ID = "ai-dj-487610"
LOCATION = "europe-west1"
MODEL_ID = "gemini-2.5-flash"


def mood_quadrant(valence: float, energy: float) -> str:
    """Return the Russell mood quadrant of a Spotify track.
    Source:
    https://psu.pb.unizin.org/psych425/chapter/circumplex-models/

    Args:
        valence: 0.0 (unpleasant) to 1.0 (pleasant).
        energy: 0.0 (deactivated) to 1.0 (activated).

    Returns:
        One emotion "happy", "angry", "sad", "calm".
    """
    if valence >= 0.5 and energy >= 0.5:
        return "happy"
    elif valence < 0.5 and energy >= 0.5:
        return "angry"
    elif valence < 0.5 and energy < 0.5:
        return "sad"
    else:
        return "calm"

web_agent = Agent(
    name="Web Researcher",
    model=Gemini(
        id="gemini-2.5-flash", vertexai=True,
        project_id=PROJECT_ID, location=LOCATION
    ),
    tools=[DuckDuckGoTools()],
    instructions=(
    "You are a research assistant. "
    "Search the web when you need current information. "
    "Return factual, sourced answers. "
    "Use DuckDuckGo for current music news. "
    ),
    markdown=True,
)

finance_agent = Agent(
    name="Financial Analyst",
    model=Gemini(
        id="gemini-2.5-flash", vertexai=True,
        project_id=PROJECT_ID, location=LOCATION
    ),
    tools=[YFinanceTools(
        enable_stock_price=True,
        enable_analyst_recommendations=True,
        enable_company_info=True,
        enable_company_news=True,
    )],
    instructions=(
        "You are a financial analyst. "
        "Always look up real data before answering. "
        "Use YFinance for music-industry stocks current hits. "
    ),
    markdown=True,
)

analyze_agent = Agent(
    name="Mood Analyst",
    model=Gemini(
        id="gemini-2.5-flash", vertexai=True,
        project_id=PROJECT_ID, location=LOCATION
    ),
    tools=[mood_quadrant],
    instructions=(
        "You are a helpful AI-DJ assistant and analyst. Be concise. "
        "You help to reason about Spotify tracks "
        "for the AI-DJ Hit Predictor project. "
        "Use mood_quadrant when the user provides Spotify audio features "
        "valence and energy to determine in which mood of the four Russell quadrants "
        "(happy, angry, sad or calm) the track is categorized. "
    ),
    markdown=True,
)
