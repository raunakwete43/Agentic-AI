from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


llm = ChatOpenAI(
    model="qwen/qwen3-4b-2507",
    base_url="http://192.168.122.1:1234/v1",
    api_key="sk-1234",
)

messages = [HumanMessage("Hi")]

response = llm.invoke(messages)

print(response)
