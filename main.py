import json
import os

from dotenv import load_dotenv

from langgraph.stream.run_stream import GraphRunStream
from langgraph.stream.stream_channel import StreamChannel
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, AIMessage

from tools.tools_registrar import TOOLS


load_dotenv()
MODEL = os.getenv("MODEL")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")

agent = create_agent(
    model=MODEL,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT
)

messages = [HumanMessage("Using the tools available to you, demo to me how you would compute 2 + 3 * 4.")]

result = agent.invoke({"messages": messages})

print(result["messages"][-1].content_blocks[0]["text"])
