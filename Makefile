.PHONY: venv seed dq test lint features train api

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

api:
	uvicorn serving.app:app --reload

test:
	pytest tests/unit tests/features tests/dq tests/integration -v

lint:
	ruff check .
	black --check .
