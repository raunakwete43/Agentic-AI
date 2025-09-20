from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import pandas as pd
from rich.pretty import pprint
import logging

load_dotenv()

model = os.getenv("MODEL", "gemini-2.5-flash-lite")
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatOpenAI(
    model=model,
    api_key=api_key,  # type: ignore
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

logger = logging.getLogger("gantt_chart_agent")
logger.setLevel(logging.DEBUG)
handler1 = logging.StreamHandler()
handler1.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler1.setFormatter(formatter)
logger.addHandler(handler1)
handler2 = logging.FileHandler("gantt_chart_agent.log")
handler2.setLevel(logging.DEBUG)
handler2.setFormatter(formatter)
logger.addHandler(handler2)


class AgentState(TypedDict):
    project_idea: str
    time_limit: int
    messages: list[BaseMessage]


class ProjectTimeLine(BaseModel):
    Task: str = Field(..., description="Name of the task")
    Start: int = Field(..., description="Start week of the task")
    Duration: int = Field(..., description="Duration in weeks of the task")


class ProjectData(BaseModel):
    timeline: list[ProjectTimeLine] = Field(
        ..., description="List of project tasks with start week and duration"
    )


def _generate_gantt_chart(
    data: ProjectData,
    img_path: str = "chart.png",
):
    logger.debug("Called _generate_gantt_chart")
    try:
        df = pd.DataFrame([t.model_dump() for t in data.timeline])

        fig, ax = plt.subplots(figsize=(10, 6))

        tasks = df["Task"]
        y_pos = range(len(tasks))

        logger.debug("Geneating Gantt chart.")
        for i, task in enumerate(tasks):
            start = df.loc[i, "Start"]
            duration = df.loc[i, "Duration"]
            ax.barh(i, duration, left=start, color="skyblue", edgecolor="black")  # type: ignore

        ax.set_yticks(y_pos)
        ax.set_yticklabels(tasks)
        ax.set_xlabel("Week")
        ax.set_title("12-Week Project Gantt Chart")
        ax.grid(axis="x")
        ax.invert_yaxis()
        plt.tight_layout()

        plt.savefig(img_path)

        logger.info(f"Gantt chart saved to {img_path}")
    except Exception as e:
        logger.error(f"Error generating Gantt chart: {e}")


def generate_gantt_chart(state: AgentState):
    logger.debug("Called generate_gantt_chart Node")
    project_idea = state["project_idea"]
    time_limit = state["time_limit"]
    messages = [
        SystemMessage(
            content="""You are a Gantt Chart Timeline Genetator. You will be
            given the  project idea and the total time in weeks to
            complete the entire project. You will break down the project
            in to tasks and assign each task a start week and reasonable
            duration in weeks."""
        ),
        HumanMessage(
            content=f"""Generate a {time_limit} week project timeline for
        building a {project_idea}"""
        ),
    ]

    try:
        logger.debug("Invoking LLM for project timeline.")
        response = llm.with_structured_output(ProjectData).invoke(messages)

        response = ProjectData.model_validate(response)
        logger.debug("LLM responded successfully")

        _generate_gantt_chart(response)
    except Exception as e:
        logger.error(f"Error in generate_gantt_chart: {e}")

    return {"messages": messages}


graph = StateGraph(AgentState)


graph.add_node("generate_gantt_chart", generate_gantt_chart)

graph.add_edge(START, "generate_gantt_chart")
graph.add_edge("generate_gantt_chart", END)

gantt_chart_agent = graph.compile()


result = gantt_chart_agent.invoke(
    {
        "project_idea": "webpage for advertising",
        "time_limit": 12,
        "messages": [],
    }
)


pprint(result)
