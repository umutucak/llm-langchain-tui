import json
import os
import warnings

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

# announces that its beta, so we shush it
warnings.filterwarnings(
    "ignore", 
    message=".*v3 streaming protocol on Pregel is experimental.*"
)

messages = []
print("Type 'quit' to exit the loop.")
while True:
    try:
        # user prompt
        user_input: str = input(" > Input: ")
        if user_input == "quit":
            print("Ending loop.")
            break

        messages.append(HumanMessage(user_input))
        stream = agent.stream_events({"messages": messages}, config=agent_thread_config, version="v3")

        print("=== Assistant Response ===")
        for message in stream.messages:
            for delta in message.text:
                print(delta, end="", flush=True)
        print("\n")

        # check if tool were used, if so, print what tool and its result
        # tool_print(stream)
        # ai response
        # result["messages"][-1].pretty_print()
    except KeyboardInterrupt:
        print("\nctrl+c caught. terminating.")
        break

