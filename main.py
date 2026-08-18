import json
import os

from dotenv import load_dotenv

from langgraph.stream.run_stream import GraphRunStream
from langgraph.stream.stream_channel import StreamChannel
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from tools.tools_registrar import TOOLS


load_dotenv()
MODEL: str = os.getenv("MODEL")
SYSTEM_PROMPT: str = os.getenv("SYSTEM_PROMPT")
MAX_TOOL_CALLS: int = 3

agent = create_agent(
    model=MODEL,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT
)

messages = [HumanMessage("Using the tools available to you, demo to me how you would compute 2 + 3 * 4.")]

result = agent.invoke({"messages": messages})

# naive tool use search for printing
_start = len(result["messages"])-2
_end = max(0, _start-MAX_TOOL_CALLS+1)
for r in result["messages"][_start:_end:-1]:
    if isinstance(r, ToolMessage):
        print(f"[TOOL] {r.name}: {r.content}")
    

print(f"AI --- {result["messages"][-1].content_blocks[0]["text"]}")
