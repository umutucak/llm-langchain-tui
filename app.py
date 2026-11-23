"""Dev entry point shim: `uv run python app.py` — the real one is llmtui.cli."""

from llmtui.cli import main

if __name__ == "__main__":
    main()
