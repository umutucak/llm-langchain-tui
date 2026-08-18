import json
import os
import warnings

from dotenv import load_dotenv

from langchain_core.utils.uuid import uuid7

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from langchain_ollama.chat_models import ChatOllama

from langgraph.graph.state import CompiledStateGraph
from langgraph.stream.run_stream import GraphRunStream
from langgraph.stream.stream_channel import StreamChannel
from langgraph.checkpoint.memory import InMemorySaver

from tools.tools_registrar import TOOLS
from util import tool_print

load_dotenv()
MODEL: str = os.getenv("MODEL")
SYSTEM_PROMPT: str = os.getenv("SYSTEM_PROMPT")
DISPLAY_THINKING: bool = bool(int(os.getenv("DISPLAY_THINKING")))
IS_REASONING: bool = bool(int(os.getenv("IS_REASONING")))
TEMPERATURE: float = float(os.getenv("TEMPERATURE"))
TOP_K: float = float(os.getenv("TOP_K"))
TOP_P: float = float(os.getenv("TOP_P"))
MIN_P: float = float(os.getenv("MIN_P"))
REPETITION_PENALTY: float = float(os.getenv("REPETITION_PENALTY"))
# PRESENCE_PENALTY: float = float(os.getenv("PRESENCE_PENALTY")) # not exposed through ChatOllama?


# values from .env, which are the recommended values from
# https://huggingface.co/Qwen/Qwen3.8-27B
model = ChatOllama(
    model=MODEL,
    reasoning=IS_REASONING,
    temperature=TEMPERATURE,
    validate_model_on_init=True,
    top_p=TOP_P,
    top_k=TOP_K,
    min_p=MIN_P,
    repeat_penalty=REPETITION_PENALTY
)

agent: CompiledStateGraph = create_agent(
    model=model,
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
            thinking_gate = False
            for event in message:
                # check if event is streamed data, and check if streamed delta has data
                delta = event.get("delta")
                if event.get("event") != "content-block-delta" or not isinstance(delta, dict):
                    continue
                # reasoning stream
                if delta.get("type") == "reasoning-delta" and DISPLAY_THINKING:
                    if not thinking_gate:
                        print("<THINKING>", end="", flush=True)
                        thinking_gate = True
                    print(delta.get("reasoning", ""), end="", flush=True)
                # output stream
                elif delta.get("type") == "text-delta":
                    if thinking_gate:
                        print("</THINKING>\n", flush=True)
                        thinking_gate = False
                    print(delta.get("text", ""), end="", flush=True)
        print("\n")
        # check if tool were used, if so, print what tool and its result
        tool_print(stream.output)
    except KeyboardInterrupt:
        print("\nctrl+c caught. terminating.")
        break
