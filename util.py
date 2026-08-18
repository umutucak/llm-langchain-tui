from dotenv import load_dotenv

from langchain.messages import ToolMessage


load_dotenv()
MAX_TOOL_CALLS: int = 3


def tool_print(result):
    # naive tool use search for printing
    _start = len(result["messages"])-2
    _end = max(0, _start-MAX_TOOL_CALLS+1)
    for r in result["messages"][_start:_end:-1]:
        if isinstance(r, ToolMessage):
            print(f"[TOOL] {r.name}: {r.content}")
