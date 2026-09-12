.PHONY: venv seed dq test lint

venv:
	python -m venv .venv
	.venv/Scripts/pip install -U pip
	.venv/Scripts/pip install -r requirements-dev.txt

seed:
	python -m generator seed

dq:
	pytest tests/dq -v

test:
	pytest tests/unit -v

lint:
	ruff check .
	black --check .
