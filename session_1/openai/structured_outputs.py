import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from rich.pretty import pprint


load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = os.getenv("MODEL", "gemini-2.5-flash-lite")


class TodoItem(BaseModel):
    name: str = Field(..., description="The name of the todo item")
    is_completed: bool


class ResponseModel(BaseModel):
    todos: List[TodoItem]


messages = [
    {
        "role": "system",
        "content": "You are a ToDo List Generator. Based on the given task you create a list of todo steps in order to complete the task.",
    },
    {"role": "user", "content": "Create a simple login page."},
]

response = client.chat.completions.parse(
    model=model,
    messages=messages,  # type: ignore
    response_format=ResponseModel,
)

todos = ResponseModel.model_validate(response.choices[0].message.parsed)

pprint(todos)
