.PHONY: venv seed dq test lint features train

venv:
	python -m venv .venv
	.venv/Scripts/pip install -U pip
	.venv/Scripts/pip install -r requirements-dev.txt

seed:
	python -m generator seed

dq:
	pytest tests/dq -v

features:
	python -m features build

train:
	python -m ml train

test:
	pytest tests/unit tests/features -v

lint:
	ruff check .
	black --check .
