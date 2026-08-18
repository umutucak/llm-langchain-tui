import json
import os

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.utils.uuid import uuid7
from langgraph.graph.state import CompiledStateGraph
from langgraph.stream.run_stream import GraphRunStream
from langgraph.stream.stream_channel import StreamChannel
from langgraph.checkpoint.memory import InMemorySaver

from tools.tools_registrar import TOOLS
from util import tool_print


load_dotenv()
MODEL: str = os.getenv("MODEL")
SYSTEM_PROMPT: str = os.getenv("SYSTEM_PROMPT")

agent: CompiledStateGraph = create_agent(
    model=MODEL,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver()
)

agent_thread_config = {
    "configurable": {
        "thread_id": str(uuid7())
    }
}

messages = []
print("Type 'quit' to exit the loop.")
while True:
    # user prompt
    user_input: str = input(" > Input: ")
    if user_input == "quit":
        print("Ending loop.")
        break

    messages.append(HumanMessage(user_input))
    result = agent.invoke({"messages": messages}, config=agent_thread_config)

    # check if tool were used, if so, print what tool and its result
    tool_print(result)
    # ai response
    result["messages"][-1].pretty_print()

