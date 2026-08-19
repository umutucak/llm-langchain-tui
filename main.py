import os
import warnings

from dotenv import load_dotenv

from langchain_core.utils.uuid import uuid7

from langchain.agents import create_agent
from langchain.messages import HumanMessage

from langchain_ollama.chat_models import ChatOllama

from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.stream.run_stream import GraphRunStream

import sqlite3

from tools.tools_registrar import TOOLS
from util import print_tool_use, set_session_title, list_sessions, is_session_named, strip_tool_context, delete_session, stream_output

load_dotenv()
MODEL: str = os.getenv("MODEL")
SQLITE_DB_PATH=os.getenv("SQLITE_DB_PATH")
with open(os.getenv("SYSTEM_PROMPT_PATH"), 'r') as f:
    SYSTEM_PROMPT: str = f.read()
CONTEXT_SIZE: int = int(os.getenv("CONTEXT_SIZE"))
MAX_TOOL_CALLS: int = 3
IS_REASONING: bool = bool(int(os.getenv("IS_REASONING")))
TEMPERATURE: float = float(os.getenv("TEMPERATURE"))
TOP_K: int = int(os.getenv("TOP_K"))
TOP_P: float = float(os.getenv("TOP_P"))
REPETITION_PENALTY: float = float(os.getenv("REPETITION_PENALTY"))


# values from .env, which are the recommended values from
# https://huggingface.co/Qwen/Qwen3.8-27B
# ctx extended according to my 3090 with q8 kv caching
model = ChatOllama(
    model=MODEL,
    reasoning=IS_REASONING,
    num_ctx=CONTEXT_SIZE,
    temperature=TEMPERATURE,
    validate_model_on_init=True,
    top_p=TOP_P,
    top_k=TOP_K,
    repeat_penalty=REPETITION_PENALTY
)

sqlite_connection = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
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


print("Control+C to exit the loop. Several / commands are available (/help to see).")
while True:
    try:
        # user prompt
        user_input: str = input(" > Input: ")
        # empty input
        if user_input.strip() == "":
            continue
        # slash commands
        # big ass ugly ass switch case, look away before it turns you to stone
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
                    sessions = list_sessions(sqlite_connection)
                    for i, (thread_id, title, updated_at) in enumerate(sessions, start=1):
                        if thread_id == agent_thread_config["configurable"]["thread_id"]:
                            print("[CURRENT] ", end="")
                        print(f"{i}. {title or '(untitled)'}  updated {updated_at}")
                case "load":
                    sessions = list_sessions(sqlite_connection)
                    for i, (thread_id, title, updated_at) in enumerate(sessions, start=1):
                        if thread_id == agent_thread_config["configurable"]["thread_id"]:
                            print("[CURRENT] ", end="")
                        print(f"{i}. {title or '(untitled)'}  updated {updated_at}")
                    selected = int(input(" > Enter index of session to resume: ")) - 1
                    agent_thread_config["configurable"]["thread_id"] = sessions[selected][0]
                case "delete":
                    sessions = list_sessions(sqlite_connection)
                    current_idx: int = 0
                    for i, (thread_id, title, updated_at) in enumerate(sessions, start=1):
                        # skip current one
                        if thread_id == agent_thread_config["configurable"]["thread_id"]:
                            current_idx = i
                            print("[CURRENT] ", end="")
                        print(f"{i}. {title or '(untitled)'}  updated {updated_at}")
                    selected = input(" > Enter indices of sessions to delete separated with white space (1 2 4): ")
                    for i in selected.strip().split(' '):
                        idx = i.strip()
                        if idx.isnumeric() and int(idx) > 0 and int(idx) <= len(sessions) and int(idx) != current_idx:
                            idx = int(idx)-1
                            # commenting this out for now for testing... surely i wont forget and then get confused
                            # confirm = input("Confirm session deletion by typing YES: ")
                            confirm = "YES"
                            if confirm == "YES":
                                delete_session(sqlite_connection, sessions[idx][0])
                                print("Delete successful.")
                            else:
                                print("Delete not confirmed.\n")
                case _:
                    print(f"{user_input[1:]} unrecognized. Try /help to see available commands.")
            continue

        
        # prompt chatbot
        # HumanMessage returns dict with keys "role" and "content" which are "user" and their input string, respectively.
        # you then put the HumanMessage in a list, as stream_events takes list as arg
        # only current message can be sent and not the full context,
        # the agent already has the context inside agent.get_state(agent_thread_config)
        stream: GraphRunStream = agent.stream_events({"messages": [HumanMessage(user_input)]}, config=agent_thread_config, version="v3")

        print("=== Assistant Response ===")
        stream_output(stream)
        # self-name the session if not already named
        if not is_session_named(sqlite_connection, agent_thread_config["configurable"]["thread_id"]):
            print("Watch for slight freeze, naming conversation...")
            self_naming_prompt = strip_tool_context(agent.get_state(agent_thread_config).values["messages"])
            with open("self_name_prompt.txt", 'r') as f:
                p = f.read()
            self_naming_prompt.append(HumanMessage(p))
            set_session_title(
                sqlite_connection,
                agent_thread_config["configurable"]["thread_id"],
                model.invoke(self_naming_prompt).content
            )

    except KeyboardInterrupt:
        print("\nctrl+c caught. terminating.")
        break
