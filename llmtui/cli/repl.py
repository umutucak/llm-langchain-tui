"""The input loop: read a line, run it as a command or send it to the agent."""

from langchain.messages import HumanMessage

from langgraph.stream.run_stream import GraphRunStream

from llmtui.cli.commands import dispatch
from llmtui.cli.render import stream_output
from llmtui.naming import name_session_if_unnamed


def run(agent, model, sqlite_connection, agent_thread_config) -> None:
    print("Control+C to exit the loop. Several / commands are available (/help to see).")
    while True:
        try:
            # user prompt
            user_input: str = input(" > Input: ")
            # empty input
            if user_input.strip() == "":
                continue
            # slash commands
            elif user_input.strip()[0] == '/':
                dispatch(user_input, sqlite_connection, agent_thread_config)
                continue


            # prompt chatbot
            # HumanMessage returns dict with keys "role" and "content" which are "user" and their input string, respectively.
            # you then put the HumanMessage in a list, as stream_events takes list as arg
            # only current message can be sent and not the full context,
            # the agent already has the context inside agent.get_state(agent_thread_config)
            stream: GraphRunStream = agent.stream_events({"messages": [HumanMessage(user_input)]}, config=agent_thread_config, version="v3")

            print("=== Assistant Response ===")
            stream_output(stream)
            name_session_if_unnamed(agent, model, sqlite_connection, agent_thread_config)

        except KeyboardInterrupt:
            print("\nctrl+c caught. terminating.")
            break
