from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from rich.pretty import pprint


load_dotenv()

model = os.getenv("MODEL", "gemini-2.5-flash-lite")
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


class AgentState(TypedDict):
    sentence: str
    sentiment: Literal["positive", "negative", "neutral"]
    score: float


class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    score: float


def get_sentiment(state: AgentState):
    response = llm.with_structured_output(SentimentResponse).invoke(f"""
Calculate the sentiment of the given sentence.
## Sentence:
{state.get("sentence")}
""")
    response = SentimentResponse.model_validate(response)
    return {"sentiment": response.sentiment, "score": response.score}


def positive_sentiment(state: AgentState):
    print(
        f"The sentence {state.get('sentence')} is Positive with score = {state.get('score')}"
    )
    return {}


def negative_sentiment(state: AgentState):
    print(
        f"The sentence {state.get('sentence')} is Negative with score = {state.get('score')}"
    )
    return {}


def neutral_sentiment(state: AgentState):
    print(
        f"The sentence {state.get('sentence')} is Neutral with score = {state.get('score')}"
    )
    return {}


def _route(state: AgentState):
    sentiment = state.get("sentiment")
    if sentiment == "positive":
        return "positive_sentiment"
    if sentiment == "negative":
        return "negative_sentiment"
    if sentiment == "neutral":
        return "neutral_sentiment"


graph = StateGraph(AgentState)

graph.add_node("get_sentiment", get_sentiment)
graph.add_node("positive_sentiment", positive_sentiment)
graph.add_node("negative_sentiment", negative_sentiment)
graph.add_node("neutral_sentiment", neutral_sentiment)

graph.add_edge(START, "get_sentiment")
graph.add_conditional_edges("get_sentiment", _route)

graph.add_edge("positive_sentiment", END)
graph.add_edge("negative_sentiment", END)
graph.add_edge("neutral_sentiment", END)

agent = graph.compile()

result = agent.invoke({"sentence": "I had a very boring day at college today!"})


pprint(result)
