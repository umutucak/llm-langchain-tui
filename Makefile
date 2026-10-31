# every command here is identical to what ci.yml runs — hook, CI, and
# developer can never drift apart

format: ## reformat code + markdown
	uv run mdformat --wrap 88 *.md
	uv run ruff format .
	uv run ruff check --fix .

lint: ## check only (what CI runs)
	uv run ruff format --check .
	uv run ruff check .
	uv run mdformat --check --wrap 88 *.md

test: ## every test (what CI runs after lint)
	uv run --with pytest python -m pytest tests

.PHONY: format lint test
