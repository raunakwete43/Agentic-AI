from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = os.getenv("MODEL", "gemini-2.5-flash-lite")

messages = [
    {
        "role": "system",
        "content": "You are a Linux Terminal running bash shell. Whatever prompt I give interpret it as a bash command and only respond with the expected output of the command. Do not include any explanations or additional text.",
    },
    {"role": "user", "content": "lsblk"},
]

response = client.chat.completions.create(
    model=model,
    messages=messages,  # type: ignore
)


print(response.choices[0].message.content)
