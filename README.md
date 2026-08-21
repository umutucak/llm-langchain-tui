Langchain practice with self-hosted model through ollama using `python3.12.13`

Most constants and hyperparameters are defined in `.env`.

works with ollama, so can install [models](https://ollama.com/library) through `ollama run`. im using the new qwen3.8-27b because why not.

to migrate dev milvus db to prod:

```
-rf milvus_prod.db && cp -r milvus_dev.db milvus_prod.db
```

ollama server systemd env variables:

```
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
```

more parameters (inside .env) injected through python

## running it

```
python app.py                               # the chat TUI
python scripts/ingest.py <pdf_dir>          # build the dev vector store
python scripts/inspect_context.py           # dump a thread's context as html
python tests/test_middleware.py             # end-to-end middleware tests
```

## layout

```
app.py                      entry point, wires everything together
llmtui/
  config.py                 every .env value, read and checked once
  agent.py                  model + tools + middleware + checkpointer
  sessions.py               the sessions table beside langgraph's threads
  naming.py                 letting the model title its own conversation
  middleware/
    errors.py               per-tool ToolErrorMiddleware handlers
    repair.py               after_model hook that fixes malformed tool calls
  tools/                    calculator and search_books, registered in __init__
  cli/
    repl.py                 the input loop
    commands.py             the / commands
    render.py               streaming output and [TOOL] lines
prompts/                    system prompt and the self-naming prompt
scripts/                    ingest and context inspection, run by hand
tests/                      end-to-end tests, no ollama or milvus needed
```

paths in `.env` are relative to the project root and resolved against it in
`config.py`, so scripts run the same from any directory.
