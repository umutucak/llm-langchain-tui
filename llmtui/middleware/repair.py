"""after_model hook that recovers search calls the model failed to format."""

import json
import uuid

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.agents.middleware import after_model, AgentState, Runtime


# the calculator catches its own errors and its arguments are a single string,
# so it does not share either failure mode. it can have its own handler if it
# ever needs one.
TOOL_NAME: str = "search_books"
# hand the turn back this many times before letting it end. an invalid call
# never reaches the tools node, so the tool call limit never counts it and
# nothing else bounds this loop.
MAX_REPAIR_ATTEMPTS: int = 2
# how the guard recognises its own hand-backs later. only this marker is
# counted, so it measures rounds of retrying rather than notices sent.
REPAIR_MARKER: str = "[malformed tool call]"
# marks a call that was thrown away rather than retried. deliberately not
# counted, so telling the model about a dropped call cannot spend its budget.
DROPPED_MARKER: str = "[tool call dropped]"


def _extract_objects(args: str) -> list[dict]:
    """Pull every complete JSON object out of a run-together argument string.

    '{"query": "a"}{"query": "b"}' is not valid JSON, but raw_decode reads one
    object at a time and reports where it stopped, so the pair can be walked
    out of the string.
    """

    decoder = json.JSONDecoder()
    objects: list[dict] = []
    text = args.strip()
    index = 0

    while index < len(text):
        try:
            obj, end = decoder.raw_decode(text, index)
        except ValueError:
            break
        # anything that is not a mapping cannot be tool arguments
        if isinstance(obj, dict):
            objects.append(obj)
        index = end
        while index < len(text) and text[index].isspace():
            index += 1

    return objects


def _repair_attempts(messages: list) -> int:
    """Count repair hand-backs already made this turn.

    Counting stops at the last user message, so the budget resets every turn
    rather than being spent once for the whole conversation.
    """

    attempts = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage) and str(message.content).startswith(REPAIR_MARKER):
            attempts += 1
    return attempts


@after_model(can_jump_to=["model"])
def repair_tool_calls(state: AgentState, runtime: Runtime) -> dict | None:
    """Recover search calls the model failed to format, before routing.

    Runs as a node between the model and the tools/end decision. Note that
    after_model hooks run in reverse of their order in the middleware list,
    so this one is listed last to run first -- the tool call limit then counts
    repaired calls rather than broken ones.
    """

    # check for messages
    messages = state.get("messages") or []
    if not messages:
        return None

    # check for AI message
    message = messages[-1]
    if not isinstance(message, AIMessage) or not message.invalid_tool_calls:
        return None

    # check for invalid tool calls
    broken = [c for c in message.invalid_tool_calls if c.get("name") == TOOL_NAME]
    if not broken:
        return None

    # decouple the dicts into multiple legal tool calls. a call whose arguments
    # yield nothing parseable is kept aside rather than quietly forgotten
    recovered: list[dict] = []
    unrepairable: list[dict] = []
    for call in broken:
        objects = _extract_objects(call.get("args") or "")
        if not objects:
            unrepairable.append(call)
            continue
        for n, args in enumerate(objects):
            recovered.append({
                "name": call["name"],
                "args": args,
                # the first keeps the model's own id, the rest need their own
                "id": call["id"] if n == 0 else str(uuid.uuid4()),
                "type": "tool_call",
            })

    # drop invalids from the original message, so that we can later put the
    # fixed versions back in. only the calls handled here are removed -- an
    # invalid call for some other tool is left for whoever owns that tool
    handled = {call["id"] for call in broken}
    blocks = []
    for block in message.content:
        # a content list can hold bare strings as well as block dicts, and
        # .get() would blow up on a string, so only dicts get type-checked
        if (isinstance(block, dict)
                and block.get("type") == "invalid_tool_call"
                and block.get("id") in handled):
            continue
        blocks.append(block)

    # every call that does not survive gets named against its own id, so the
    # model is never left to infer that one of its requests went missing
    def dropped_notice(call: dict) -> ToolMessage:
        return ToolMessage(
            content=(
                f"{DROPPED_MARKER} Your `{TOOL_NAME}` call was discarded and never "
                f"ran: {call.get('error', 'arguments were not valid JSON')}. Its "
                f"arguments started {str(call.get('args'))[:80]!r}. Treat this "
                f"request as unanswered -- no passages were searched for it."
            ),
            tool_call_id=call["id"],
            name=TOOL_NAME,
            status="error",
        )

    # pipe back out to the tool node the newly fixed tool calls
    if recovered:
        for call in recovered:
            blocks.append({
                "type": "tool_call",
                "id": call["id"],
                "name": call["name"],
                "args": call["args"],
            })
        print(f"[REPAIR] recovered {len(recovered)} search call(s) from malformed arguments")
        if unrepairable:
            print(f"[REPAIR] {len(unrepairable)} call(s) could not be recovered, reported to the model")
        return {
            "messages": [
                AIMessage(
                    content=blocks,
                    # any calls the model got right are kept alongside the repairs
                    tool_calls=list(message.tool_calls) + recovered,
                    id=message.id,
                ),
                # ToolMessage reports back to the ai for the dropped tools
                *[dropped_notice(call) for call in unrepairable],
            ]
        }

    # failed repair attempts dont count towards out ToolCallLimitMiddleware,
    # so we keep a separate counter here, and block tool attempts that exceed it
    # if limit = 1 and 4 tool calls are received, the first 3 make it through, and
    # its reported back to the agent that the 4th didnt make it through
    if _repair_attempts(messages) >= MAX_REPAIR_ATTEMPTS:
        print(f"[REPAIR] gave up after repeated malformed tool calls, {len(unrepairable)} dropped")
        return {
            "messages": [
                AIMessage(content=blocks, id=message.id),
                # ToolMessage reports back to the ai for the dropped tools
                *[dropped_notice(call) for call in unrepairable],
            ]
        }

    print("[REPAIR] arguments unparseable, handing the turn back")
    handbacks = []
    for call in unrepairable:
        handbacks.append(ToolMessage(
            content=(
                f"{REPAIR_MARKER} Your `{TOOL_NAME}` call could not be read: "
                f"{call.get('error', 'arguments were not valid JSON')}. "
                f"Reissue it as a single call whose arguments are one JSON "
                f"object, for example {{\"query\": \"...\", \"book\": \"...\"}}. "
                f"To run more than one search, make them in separate calls."
            ),
            tool_call_id=call["id"],
            name=TOOL_NAME,
            status="error",
        ))

    return {
        "messages": [AIMessage(content=blocks, id=message.id)] + handbacks,
        "jump_to": "model",
    }
