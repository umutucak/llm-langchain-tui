"""Turning the agent's event stream into something readable in a terminal."""

from langchain.messages import ToolMessage

from langgraph.stream.run_stream import GraphRunStream

from llmtui.config import DISPLAY_THINKING, MAX_TOOL_CALLS


def stream_output(stream:GraphRunStream):
    # guide i read to write this abomination:
    # https://docs.langchain.com/oss/python/deepagents/event-streaming#stream-messages

    # message type = ChatModelStream
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
        # a tool-calling turn ends on the tool call, with no text delta to
        # close the block, so it has to be closed out here instead
        if thinking_gate:
            print("</THINKING>\n", flush=True)
    print("\n")
    # check if tool were used, if so, print what tool and its result
    print_tool_use(stream.output)



def print_tool_use(result) -> None:
    # naive tool use search for printing
    _start = len(result["messages"])-2
    _end = max(0, _start-MAX_TOOL_CALLS+1)
    for r in result["messages"][_start:_end:-1]:
        if isinstance(r, ToolMessage):
            print(f"[TOOL] {r.name}: {r.content}")
