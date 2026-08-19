"""Render a checkpointed thread as a readable HTML page.

    python inspect_context.py                 # this usage guide
    python inspect_context.py --list          # what threads exist
    python inspect_context.py --select        # pick one by index and render it

Shows every message the agent would resend on its next turn: content blocks
kept separate, tool calls with their arguments, retrieved passages in full,
plus what share of the window each message kind is taking and whether the
answers cited anything that was never retrieved.

Each run writes its own file into resources/context_inspection/, named after
the session's own title -- draw_steel_hexcrawl_rules_inspection.html and so
on. Inspecting a session twice adds _2, _3 rather than replacing what was
there, so two views of the same thread can be compared. The page is
self-contained -- styles inline, nothing fetched -- so it opens straight from
disk in any browser, offline, with no server involved.
"""

import os
import re
import sys
import json
import html
import sqlite3
import argparse
import collections

from dotenv import load_dotenv

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH")
CONTEXT_DUMP_DIR: str = "resources/context_inspection"

# left border colour and heading label per message kind
ROLE = {
    "HumanMessage": ("human", "You"),
    "AIMessage": ("assistant", "Assistant"),
    "ToolMessage": ("tool", "Tool result"),
}


def e(value) -> str:
    return html.escape(str(value))


def thread_title(connection, thread_id: str) -> str | None:
    """The session's self-assigned name, if it ever got one."""

    row = connection.execute(
        "SELECT title FROM sessions WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    return row[0] if row and row[0] else None


def slugify(title: str) -> str:
    """Turn a session title into something safe to put in a filename."""

    slug = re.sub(r"[^\w\s-]", "", title).strip().lower()
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug or "untitled"


def output_path(connection, thread_id: str) -> str:
    """A file named after the session, kept distinct from earlier inspections.

    Sessions that were never named fall back to the head of their thread id,
    which is ugly but at least unambiguous.
    """

    title = thread_title(connection, thread_id)
    stem = slugify(title) if title else thread_id.split("-")[0]

    os.makedirs(CONTEXT_DUMP_DIR, exist_ok=True)
    candidate = os.path.join(CONTEXT_DUMP_DIR, f"{stem}_inspection.html")
    # a repeat inspection gets its own file rather than replacing the last one
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(CONTEXT_DUMP_DIR, f"{stem}_inspection_{n}.html")
        n += 1
    return candidate


def load_threads(connection) -> list[dict]:
    """Every thread in the checkpoint db, with whatever the session was named."""

    saver = SqliteSaver(connection)
    threads = []
    for (thread_id,) in connection.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall():
        tup = saver.get_tuple({"configurable": {"thread_id": thread_id}})
        if not tup:
            continue
        messages = tup.checkpoint["channel_values"].get("messages", [])
        threads.append({
            "ts": tup.checkpoint.get("ts", ""),
            "thread_id": thread_id,
            "count": len(messages),
            "title": thread_title(connection, thread_id),
        })
    return threads


def by_name(threads: list[dict]) -> list[dict]:
    """Alphabetical by session name, with the never-named ones after them."""

    return sorted(threads, key=lambda t: (t["title"] is None, (t["title"] or t["thread_id"]).lower()))


def message_size(message) -> int:
    """Roughly what this message costs in the window."""

    if isinstance(message, AIMessage):
        return sum(len(json.dumps(b, default=str)) for b in message.content)
    return len(str(message.content))


def render_ai(message) -> list[str]:
    """One labelled section per content block, so block structure stays visible."""

    parts = []
    for block in message.content:
        if not isinstance(block, dict):
            parts.append(f'<div class="blk"><pre>{e(block)}</pre></div>')
            continue

        kind = block.get("type")
        if kind == "reasoning":
            parts.append(
                '<div class="blk reasoning"><div class="blk-label">reasoning block</div>'
                f'<div class="prose dim">{e(block.get("reasoning", ""))}</div></div>')
        elif kind == "text":
            parts.append(
                '<div class="blk"><div class="blk-label">text block</div>'
                f'<div class="prose">{e(block.get("text", ""))}</div></div>')
        elif kind == "tool_call":
            parts.append(
                '<div class="blk call"><div class="blk-label">tool_call block</div>'
                f'<div class="kv"><span>name</span><code>{e(block.get("name"))}</code>'
                f'<span>id</span><code>{e(block.get("id"))}</code></div>'
                f'<pre class="args">{e(json.dumps(block.get("args"), indent=2))}</pre></div>')
        elif kind == "invalid_tool_call":
            parts.append(
                '<div class="blk bad"><div class="blk-label warn">invalid_tool_call block</div>'
                f'<div class="kv"><span>error</span><code class="no">{e(block.get("error"))}</code></div>'
                f'<pre class="args">{e(block.get("args"))}</pre></div>')
        else:
            parts.append(
                f'<div class="blk"><div class="blk-label">{e(kind)} block</div>'
                f'<pre>{e(json.dumps(block, default=str, indent=2))}</pre></div>')

    # these two counters are what tell a dead turn from a working one
    valid, invalid = len(message.tool_calls), len(message.invalid_tool_calls)
    parts.append(
        f'<div class="kv"><span>.tool_calls</span><code class="{"ok" if valid else ""}">{valid}</code>'
        f'<span>.invalid_tool_calls</span><code class="{"no" if invalid else "ok"}">{invalid}</code></div>')
    return parts


def render_tool(message) -> list[str]:
    """Tool results are split back into the passages the tool joined together."""

    parts = [
        f'<div class="kv"><span>status</span>'
        f'<code class="{"no" if message.status == "error" else "ok"}">{e(message.status)}</code>'
        f'<span>tool</span><code>{e(message.name)}</code>'
        f'<span>replies to</span><code>{e(message.tool_call_id)}</code></div>'
    ]
    passages = re.split(r"\n\n---\n\n", str(message.content))
    parts.append(f'<div class="blk-label">{len(passages)} passage(s) returned</div>')
    for passage in passages:
        head, _, rest = passage.partition("\n")
        parts.append(
            f'<div class="passage"><div class="cite">{e(head)}</div>'
            f'<pre>{e(rest)}</pre></div>')
    return parts


def build_page(thread_id: str, messages: list) -> str:
    totals = collections.Counter()
    for message in messages:
        totals[type(message).__name__] += message_size(message)
    grand = sum(totals.values()) or 1

    # a citation naming a page that was never retrieved is a fabricated one
    retrieved, cited = set(), set()
    for message in messages:
        if isinstance(message, ToolMessage):
            retrieved |= set(re.findall(r"p\.(\d+)", str(message.content)))
        elif isinstance(message, AIMessage):
            text = "".join(b.get("text", "") for b in message.content
                           if isinstance(b, dict) and b.get("type") == "text")
            cited |= set(re.findall(r"p\.(\d+)", text))
    fabricated = cited - retrieved

    cards = []
    for n, message in enumerate(messages):
        kind = type(message).__name__
        css, label = ROLE.get(kind, ("assistant", kind))
        if isinstance(message, AIMessage):
            body = render_ai(message)
            types = ",".join(b.get("type", "?") for b in message.content if isinstance(b, dict))
        elif isinstance(message, ToolMessage):
            body = render_tool(message)
            types = ""
        else:
            body = [f'<div class="prose">{e(message.content)}</div>']
            types = ""

        cards.append(
            f'<article class="msg {css}"><header class="msg-head">'
            f'<span class="idx">{n}</span><span class="role">{label}</span>'
            f'<span class="meta">{e(kind)}{" &middot; " + e(types) if types else ""}'
            f' &middot; {message_size(message):,} chars</span></header>'
            f'{"".join(body)}</article>')

    bar = "".join(
        f'<div class="seg {ROLE.get(k, ("assistant",))[0]}" style="width:{v / grand * 100:.2f}%"></div>'
        for k, v in totals.items())
    legend = "".join(
        f'<span class="lg"><i class="dot {ROLE.get(k, ("assistant",))[0]}"></i>'
        f'{ROLE.get(k, (None, k))[1]} <b>{v / grand * 100:.0f}%</b> <em>{v:,}</em></span>'
        for k, v in totals.items())

    return TEMPLATE.format(
        thread_id=e(thread_id),
        # the first uuid segment is enough to tell pages apart in a gallery
        short_id=e(thread_id.split("-")[0]),
        count=len(messages), grand=f"{grand:,}",
        bar=bar, legend=legend,
        retrieved=", ".join(sorted(retrieved, key=int)) or "none",
        cited=", ".join(sorted(cited, key=int)) or "none",
        fab_class="no" if fabricated else "pass",
        fabricated=", ".join(sorted(fabricated, key=int)) if fabricated else "none",
        cards="".join(cards),
    )


TEMPLATE = """<title>Agent Context {short_id}</title>
<style>
:root {{
  --ground:#F4F6F8; --surface:#FFFFFF; --sunk:#EDF1F4;
  --ink:#151A1F; --muted:#5C6773; --faint:#8794A1; --rule:#DCE2E8;
  --human:#2F6F4F; --assistant:#2B5C9B; --tool:#8A5A1E;
  --ok:#2F6F4F; --no:#9E3B2C;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
  --serif:Georgia,"Iowan Old Style","Palatino Linotype",serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#0F1418; --surface:#161C22; --sunk:#1B232A;
    --ink:#E3E8EC; --muted:#95A1AC; --faint:#6B7883; --rule:#242D35;
    --human:#6BBF8F; --assistant:#7BA8DC; --tool:#D9A05B;
    --ok:#6BBF8F; --no:#E28575;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0F1418; --surface:#161C22; --sunk:#1B232A;
  --ink:#E3E8EC; --muted:#95A1AC; --faint:#6B7883; --rule:#242D35;
  --human:#6BBF8F; --assistant:#7BA8DC; --tool:#D9A05B;
  --ok:#6BBF8F; --no:#E28575;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:60rem;margin:0 auto;padding:2.5rem 1.25rem 5rem;
  display:flex;flex-direction:column;gap:1.5rem}}
h1{{font-size:1.5rem;font-weight:600;letter-spacing:-0.02em;margin:0;text-wrap:balance}}
.sub{{color:var(--muted);font-size:0.9rem;margin:0}}
.hdr{{display:flex;flex-direction:column;gap:0.5rem;border-bottom:1px solid var(--rule);
  padding-bottom:1.25rem}}
.tid{{font-family:var(--mono);font-size:0.76rem;color:var(--faint);word-break:break-all}}
.panel{{background:var(--surface);border:1px solid var(--rule);border-radius:7px;
  padding:1rem 1.15rem;display:flex;flex-direction:column;gap:0.75rem}}
.panel h2{{font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
  color:var(--faint);margin:0}}
.barwrap{{display:flex;height:12px;border-radius:3px;overflow:hidden;background:var(--sunk)}}
.seg.human{{background:var(--human)}} .seg.assistant{{background:var(--assistant)}}
.seg.tool{{background:var(--tool)}}
.legend{{display:flex;flex-wrap:wrap;gap:0.4rem 1.4rem;font-size:0.82rem;color:var(--muted)}}
.lg b{{color:var(--ink);font-variant-numeric:tabular-nums}}
.lg em{{font-style:normal;color:var(--faint);font-family:var(--mono);font-size:0.76rem}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:0.4rem}}
.dot.human{{background:var(--human)}} .dot.assistant{{background:var(--assistant)}}
.dot.tool{{background:var(--tool)}}
.audit{{display:flex;flex-wrap:wrap;gap:0.35rem 1.5rem;font-size:0.85rem}}
.audit b{{font-family:var(--mono);font-size:0.8rem}}
.pass{{color:var(--ok);font-weight:600}} .no{{color:var(--no);font-weight:600}}
.msg{{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--rule);
  border-radius:6px;padding:1rem 1.15rem;display:flex;flex-direction:column;gap:0.8rem}}
.msg.human{{border-left-color:var(--human)}}
.msg.assistant{{border-left-color:var(--assistant)}}
.msg.tool{{border-left-color:var(--tool)}}
.msg-head{{display:flex;align-items:baseline;gap:0.7rem;flex-wrap:wrap}}
.idx{{font-family:var(--mono);font-size:0.72rem;color:var(--faint);
  background:var(--sunk);border-radius:3px;padding:0.1rem 0.42rem}}
.role{{font-size:0.78rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase}}
.msg.human .role{{color:var(--human)}}
.msg.assistant .role{{color:var(--assistant)}}
.msg.tool .role{{color:var(--tool)}}
.meta{{font-family:var(--mono);font-size:0.72rem;color:var(--faint);margin-left:auto}}
.prose{{font-family:var(--serif);font-size:1rem;line-height:1.7;white-space:pre-wrap;
  overflow-wrap:break-word;max-width:66ch}}
.prose.dim{{color:var(--muted);font-size:0.94rem}}
.blk{{border-left:2px solid var(--rule);padding-left:0.85rem;
  display:flex;flex-direction:column;gap:0.45rem}}
.blk.reasoning{{border-left-style:dashed}}
.blk.call{{border-left-color:var(--assistant)}}
.blk.bad{{border-left-color:var(--no)}}
.blk-label{{font-family:var(--mono);font-size:0.7rem;letter-spacing:0.06em;
  text-transform:uppercase;color:var(--faint)}}
.blk-label.warn{{color:var(--no)}}
.kv{{display:grid;grid-template-columns:auto 1fr;gap:0.28rem 0.8rem;
  align-items:baseline;font-size:0.8rem}}
.kv span{{font-family:var(--mono);font-size:0.72rem;color:var(--faint)}}
code{{font-family:var(--mono);font-size:0.78rem;background:var(--sunk);
  border-radius:3px;padding:0.08em 0.36em;word-break:break-all}}
code.ok{{color:var(--ok)}} code.no{{color:var(--no)}}
pre{{font-family:var(--mono);font-size:0.75rem;line-height:1.55;background:var(--sunk);
  border:1px solid var(--rule);border-radius:5px;padding:0.7rem 0.85rem;margin:0;
  overflow-x:auto;white-space:pre-wrap;overflow-wrap:break-word}}
pre.args{{white-space:pre}}
.passage{{display:flex;flex-direction:column;gap:0.3rem}}
.cite{{font-family:var(--mono);font-size:0.73rem;color:var(--tool);font-weight:600}}
@media(max-width:620px){{ .meta{{margin-left:0;width:100%}} .wrap{{padding:1.75rem 0.9rem 3rem}} }}
</style>

<div class="wrap">
  <header class="hdr">
    <h1>Agent Context</h1>
    <p class="sub">Every message the agent will resend on the next turn, exactly as checkpointed.</p>
    <div class="tid">thread {thread_id} &middot; {count} messages &middot; {grand} chars</div>
  </header>

  <section class="panel">
    <h2>What is filling the window</h2>
    <div class="barwrap">{bar}</div>
    <div class="legend">{legend}</div>
  </section>

  <section class="panel">
    <h2>Citation audit</h2>
    <div class="audit">
      <span>pages retrieved <b>{retrieved}</b></span>
      <span>pages cited <b>{cited}</b></span>
      <span>cited but never retrieved <b class="{fab_class}">{fabricated}</b></span>
    </div>
  </section>

  {cards}
</div>"""


def render(connection, thread_id: str) -> None:
    saver = SqliteSaver(connection)
    tup = saver.get_tuple({"configurable": {"thread_id": thread_id}})
    if not tup:
        sys.exit(f"no checkpoint for thread {thread_id}")

    messages = tup.checkpoint["channel_values"].get("messages", [])
    out = output_path(connection, thread_id)
    with open(out, "w") as f:
        f.write(build_page(thread_id, messages))
    print(f"wrote {out}  ({len(messages)} messages)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a checkpointed thread as a self-contained HTML page.")
    parser.add_argument("--list", action="store_true",
                        help="show every stored thread, oldest first")
    parser.add_argument("--select", action="store_true",
                        help="choose a thread by index and render it")
    args = parser.parse_args()

    # bare invocation explains itself rather than guessing at a thread
    if not (args.list or args.select):
        parser.print_help()
        return

    connection = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    threads = load_threads(connection)
    if not threads:
        sys.exit("no threads in the checkpoint database")

    if args.list:
        for thread in sorted(threads, key=lambda t: t["ts"]):
            name = thread["title"] or "(untitled)"
            print(f"{thread['ts'][:19]}  {name:34}  {thread['thread_id']}  "
                  f"{thread['count']} messages")
        return

    ordered = by_name(threads)
    for i, thread in enumerate(ordered, start=1):
        name = thread["title"] or f"(untitled) {thread['thread_id'][:13]}"
        print(f"{i:3}. {name:38}  {thread['ts'][:19]}  {thread['count']} messages")

    choice = input(" > Enter index of the session to inspect: ").strip()
    if not choice.isnumeric() or not 1 <= int(choice) <= len(ordered):
        sys.exit(f"'{choice}' is not one of 1-{len(ordered)}")

    render(connection, ordered[int(choice) - 1]["thread_id"])


if __name__ == "__main__":
    main()
