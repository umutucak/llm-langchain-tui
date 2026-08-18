import os
import warnings

from dotenv import load_dotenv

from langchain_core.utils.uuid import uuid7

from langchain.agents import create_agent
from langchain.messages import HumanMessage

from langchain_ollama.chat_models import ChatOllama

from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

import sqlite3

from tools.tools_registrar import TOOLS
from util import print_tool_use, set_session_title, list_sessions, is_session_named, strip_tool_context, delete_session

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

sqlite_connection = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
sqlite_connection.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        thread_id TEXT PRIMARY KEY,
        title TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    )
""")
sqlite_connection.commit()

memory: SqliteSaver = SqliteSaver(sqlite_connection)
memory.setup()

agent: CompiledStateGraph = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory
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


print("Type 'quit' to exit the loop. Several / commands are available (/help to see).")
while True:
    try:
        # user prompt
        user_input: str = input(" > Input: ")
        if user_input.strip() == "":
            continue
        elif user_input == "quit":
            print("Ending loop.")
            break
        elif user_input.strip()[0] == '/':
            match user_input[1:]:
                case "help":
                    print("/name -- Name your current conversation and save to database.")
                    print("/list -- See all conversations so far.")
                    print("/load -- Resume a previous conversation.")
                    print("/delete -- Delete a previous conversation.")
                case "name":
                    session_title = input(" > Enter the title of the session: ")
                    set_session_title(sqlite_connection, agent_thread_config["configurable"]["thread_id"], session_title)
                case "list":
                    for thread_id, title, updated_at in list_sessions(sqlite_connection):
                        print(f"{title or '(untitled)'}  [{thread_id}]  updated {updated_at}")
                case "load":
                    sessions = list_sessions(sqlite_connection)
                    for i, (thread_id, title, updated_at) in enumerate(sessions, start=1):
                        print(f"{i}. {title or '(untitled)'}  updated {updated_at}")
                    selected = int(input(" > Enter index of session to resume: ")) - 1
                    agent_thread_config["configurable"]["thread_id"] = sessions[selected][0]
                case "delete":
                    sessions = list_sessions(sqlite_connection)
                    for i, (thread_id, title, updated_at) in enumerate(sessions, start=1):
                        # skip current one
                        if thread_id == agent_thread_config["configurable"]["thread_id"]:
                            continue
                        print(f"{i}. {title or '(untitled)'}  updated {updated_at}")
                    selected = int(input(" > Enter index of session to delete: ")) - 1
                    confirm = input("Confirm session deletion by typing YES: ")
                    if confirm == "YES":
                        delete_session(sqlite_connection, sessions[selected][0])
                        print("Delete successful.")
                    else:
                        print("Delete not confirmed.\n")
                case _:
                    print(f"{user_input[1:]} unrecognized. Try /help to see available commands.")
            continue
            
        stream = agent.stream_events({"messages": [HumanMessage(user_input)]}, config=agent_thread_config, version="v3")

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
        print_tool_use(stream.output)
        # self-name the session
        if not is_session_named(sqlite_connection, agent_thread_config["configurable"]["thread_id"]):
            self_naming_prompt = strip_tool_context(agent.get_state(agent_thread_config).values["messages"])
            p = """**You are a conversation-naming engine.** Given the first user message and the first assistant reply in a conversation, produce a short, human-friendly title (2-7 words) that a user would use to *find this conversation again* in a sidebar or history list.
                **Rules:**
                - Write it as a noun phrase or short label, not a sentence. No "How to…", "Question about…", or "The user asks…".
                - Capture the **user's intent or the subject matter**, not the mechanics of the exchange.
                - Be specific enough to distinguish from similar conversations, but general enough that it still fits if the chat meanders a bit.
                - Title Case. No punctuation at the end. No quotes.
                - If the topic is a task (e.g., debugging, writing a poem, planning a trip), name the *object of the task* ("Debugging React Hydration Error," "Trip Itinerary - Lisbon in June"), not the verb of help-seeking.
                - Avoid filler like "Chat about…", "Discussion on…", "Inquiry regarding…".
                - If the first exchange is ambiguous or very short, lean on the user's framing.

                **Examples of good titles:**
                - User: "Why is my Python script printing `None` instead of the list?" / Assistant explains the in-place mutation. → **None vs. List - Python Mutation Bug**
                - User: "Can you write a haiku about my cat knocking things off shelves?" / Assistant writes one. → **Cat Shelf-Knocker Haiku**
                - User: "I need a 3-week itinerary for Japan in October" / Assistant outlines days. → **Japan - October, 3 Weeks**
                - User: "Explain CRISPR like I'm 12" / Assistant does. → **CRISPR, Explained Simply**
                - User: "Help me name my bakery" / Assistant brainstorms. → **Bakery Name Brainstorm**

                **Now generate the title for this conversation:**
            """
            self_naming_prompt.append(HumanMessage(p))
            set_session_title(
                sqlite_connection,
                agent_thread_config["configurable"]["thread_id"],
                model.invoke(self_naming_prompt).content
            )

    except KeyboardInterrupt:
        print("\nctrl+c caught. terminating.")
        break
