import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List
from rich.pretty import pprint


load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL", "gemini-2.5-flash-lite"),
    api_key=os.getenv("GEMINI_API_KEY"),  # type: ignore
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


class TodoItem(BaseModel):
    name: str = Field(..., description="The name of the todo item")
    is_completed: bool


class ResponseModel(BaseModel):
    todos: List[TodoItem]


messages = [
    SystemMessage(
        "You are a ToDo List Generator. Based on the given task you create a list of todo steps in order to complete the task."
    ),
    HumanMessage("Create a simple login page."),
]

response = llm.with_structured_output(ResponseModel).invoke(messages)

response = ResponseModel.model_validate(response)

pprint(response)
