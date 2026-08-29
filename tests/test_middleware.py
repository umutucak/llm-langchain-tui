"""End-to-end tests driving the real agent graph with a scripted model."""
import sys, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ToolErrorMiddleware
from langchain.tools import tool

from llmtui.middleware import on_search_error, repair_tool_calls

BAD_CONCAT = ('{"query": "setting of the game", "book": "Heart Beneath the City"}'
              '{"query": "what kind of adventures can you run", "book": "Heart Beneath the City"}')
TRUNCATED = '{"query": "grap'


class ScriptedModel(BaseChatModel):
    """Pops pre-built AIMessages so the graph runs deterministically."""
    responses: list

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self


def ai_invalid(args, name="search_books", cid=None):
    cid = cid or str(uuid.uuid4())
    m = AIMessage(
        content=[{"type": "reasoning", "reasoning": "thinking", "index": 0},
                 {"type": "invalid_tool_call", "id": cid, "name": name,
                  "args": args, "error": "Failed to parse tool call arguments as JSON"}],
        id=str(uuid.uuid4()))
    m.invalid_tool_calls = [{"name": name, "args": args, "id": cid,
                             "error": "Failed to parse tool call arguments as JSON",
                             "type": "invalid_tool_call"}]
    return m


def ai_valid(calls):
    """calls = [(name, args_dict), ...]"""
    tcs, blocks = [], [{"type": "reasoning", "reasoning": "thinking", "index": 0}]
    for name, args in calls:
        cid = str(uuid.uuid4())
        tcs.append({"name": name, "args": args, "id": cid, "type": "tool_call"})
        blocks.append({"type": "tool_call", "id": cid, "name": name, "args": args})
    return AIMessage(content=blocks, tool_calls=tcs, id=str(uuid.uuid4()))


def ai_invalid_many(argstrings, name="search_books"):
    """One AIMessage carrying several malformed calls at once."""
    blocks = [{"type": "reasoning", "reasoning": "thinking", "index": 0}]
    invalid = []
    for args in argstrings:
        cid = str(uuid.uuid4())
        blocks.append({"type": "invalid_tool_call", "id": cid, "name": name,
                       "args": args, "error": "Failed to parse tool call arguments as JSON"})
        invalid.append({"name": name, "args": args, "id": cid,
                        "error": "Failed to parse tool call arguments as JSON",
                        "type": "invalid_tool_call"})
    m = AIMessage(content=blocks, id=str(uuid.uuid4()))
    m.invalid_tool_calls = invalid
    return m


def ai_text(text):
    return AIMessage(content=[{"type": "text", "text": text}], id=str(uuid.uuid4()))


CALLS = []

def build(responses, raises=False, run_limit=3):
    CALLS.clear()

    @tool
    def search_books(query: str, book: str = "") -> str:
        """Search the book library."""
        CALLS.append(query)
        if raises:
            raise RuntimeError("simulated milvus GOAWAY")
        return f"PASSAGE for {query!r}"

    return create_agent(
        model=ScriptedModel(responses=list(responses)),
        tools=[search_books],
        middleware=[
            ToolCallLimitMiddleware(tool_name="search_books", run_limit=run_limit),
            ToolErrorMiddleware(on_error=on_search_error, tools=["search_books"]),
            repair_tool_calls,
        ],
    )


def trace(result):
    out = []
    for m in result["messages"]:
        kind = type(m).__name__
        if isinstance(m, AIMessage):
            types = [b.get("type") for b in m.content if isinstance(b, dict)]
            out.append(f"{kind}({','.join(types)}) tool_calls={len(m.tool_calls)}")
        elif isinstance(m, ToolMessage):
            out.append(f"{kind}[{m.status}] {str(m.content)[:52]!r}")
        else:
            out.append(f"{kind} {str(m.content)[:40]!r}")
    return out


def run(label, agent, expectations):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    result = agent.invoke({"messages": [HumanMessage("a two part question")]})
    for line in trace(result):
        print("   ", line)
    print(f"    tools actually executed: {CALLS}")
    ok = True
    for desc, passed in expectations(result):
        print(f"    {'PASS' if passed else 'FAIL'}  {desc}")
        ok = ok and passed
    return ok


results = {}

# ---- 1. concatenated JSON gets expanded into two real searches ----
agent = build([ai_invalid(BAD_CONCAT), ai_text("Both parts answered.")])
results["1 repair->2 calls"] = run(
    "CASE 1  malformed concatenated args -> two executed searches", agent,
    lambda r: [
        ("both queries executed", len(CALLS) == 2),
        ("two ToolMessages returned", sum(isinstance(m, ToolMessage) for m in r["messages"]) == 2),
        ("no invalid_tool_call block left in history",
         all(b.get("type") != "invalid_tool_call"
             for m in r["messages"] if isinstance(m, AIMessage)
             for b in m.content if isinstance(b, dict))),
        ("final message is text", r["messages"][-1].content[0].get("type") == "text"),
    ])

# ---- 2. unrepairable -> turn handed back -> model reissues valid call ----
agent = build([ai_invalid(TRUNCATED),
               ai_valid([("search_books", {"query": "grappling"})]),
               ai_text("Recovered after retry.")])
results["2 jump back + retry"] = run(
    "CASE 2  truncated args -> jump_to model -> model reissues correctly", agent,
    lambda r: [
        ("model got a second chance and searched", CALLS == ["grappling"]),
        ("an error ToolMessage explained the problem",
         any(isinstance(m, ToolMessage) and m.status == "error" for m in r["messages"])),
        ("run completed with an answer", r["messages"][-1].content[0].get("type") == "text"),
    ])

# ---- 3. loop guard: model keeps emitting garbage ----
agent = build([ai_invalid(TRUNCATED), ai_invalid(TRUNCATED),
               ai_invalid(TRUNCATED), ai_invalid(TRUNCATED)])
results["3 loop guard"] = run(
    "CASE 3  repeated unparseable args -> guard stops the loop", agent,
    lambda r: [
        ("never executed a tool", CALLS == []),
        ("stopped at 2 hand-backs",
         sum(isinstance(m, ToolMessage) and str(m.content).startswith("[malformed tool call]")
             for m in r["messages"]) == 2),
        ("terminated instead of looping forever", True),
    ])

# ---- 4. tool raises -> ToolErrorMiddleware feeds it back ----
agent = build([ai_valid([("search_books", {"query": "grapple"})]),
               ai_text("Told the user the library is unavailable.")], raises=True)
results["4 tool error"] = run(
    "CASE 4  search_books raises -> error fed back, run survives", agent,
    lambda r: [
        ("run did not crash", True),
        ("error surfaced as a ToolMessage",
         any(isinstance(m, ToolMessage) and m.status == "error" for m in r["messages"])),
        ("message names the exception type",
         any(isinstance(m, ToolMessage) and "RuntimeError" in str(m.content) for m in r["messages"])),
        ("message forbids answering from general knowledge",
         any(isinstance(m, ToolMessage) and "general knowledge" in str(m.content) for m in r["messages"])),
    ])

# ---- 5. call limit blocks the 4th search in one turn ----
agent = build([ai_valid([("search_books", {"query": f"q{i}"}) for i in range(4)]),
               ai_text("Answered with what I had.")], run_limit=3)
results["5 call limit"] = run(
    "CASE 5  four searches in one turn with run_limit=3", agent,
    lambda r: [
        ("at most 3 searches actually ran", len(CALLS) <= 3),
        ("run still produced an answer", r["messages"][-1].content[0].get("type") == "text"),
    ])

# ---- 6. well-formed single call passes through untouched ----
agent = build([ai_valid([("search_books", {"query": "grapple rules"})]),
               ai_text("Here are the rules.")])
results["6 valid passthrough"] = run(
    "CASE 6  normal single call -> repair does not interfere", agent,
    lambda r: [
        ("executed exactly once", CALLS == ["grapple rules"]),
        ("no repair message emitted",
         not any(isinstance(m, ToolMessage) and str(m.content).startswith("[malformed") for m in r["messages"])),
    ])

# ---- 7. several well-formed calls all run ----
agent = build([ai_valid([("search_books", {"query": "a"}), ("search_books", {"query": "b"})]),
               ai_text("Both answered.")])
results["7 multi valid"] = run(
    "CASE 7  two well-formed calls -> both execute, untouched", agent,
    lambda r: [
        ("both executed", sorted(CALLS) == ["a", "b"]),
        ("two ToolMessages", sum(isinstance(m, ToolMessage) for m in r["messages"]) == 2),
    ])

# ---- 8. mixed: one call recoverable, one not ----
agent = build([ai_invalid_many([BAD_CONCAT, TRUNCATED]), ai_text("Answered what I could.")])
results["8 partial recovery"] = run(
    "CASE 8  one recoverable + one unrecoverable call in the same message", agent,
    lambda r: [
        ("the recoverable call still ran, expanded into two searches", len(CALLS) == 2),
        ("the unrecoverable call was reported, not silently dropped",
         any(isinstance(m, ToolMessage) and str(m.content).startswith("[tool call dropped]")
             for m in r["messages"])),
        ("the dropped notice names a real tool_call_id",
         any(isinstance(m, ToolMessage) and str(m.content).startswith("[tool call dropped]")
             and m.tool_call_id for m in r["messages"])),
        ("no invalid_tool_call block survives in history",
         all(b.get("type") != "invalid_tool_call"
             for m in r["messages"] if isinstance(m, AIMessage)
             for b in m.content if isinstance(b, dict))),
        ("run still produced an answer", r["messages"][-1].content[0].get("type") == "text"),
    ])

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for k, v in results.items():
    print(f"  {'PASS' if v else 'FAIL'}   {k}")
print(f"\n{sum(results.values())}/{len(results)} cases passed")
