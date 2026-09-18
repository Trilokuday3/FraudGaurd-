.PHONY: venv seed dq test lint features train api replay frontend dq-scored drift promote rollback

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

replay:
	python -m scripts.replay_transactions --loop

test:
	pytest tests/unit tests/features tests/dq tests/integration -v

lint:
	ruff check .
	black --check .

frontend:
	cd frontend && npm run dev

dq-scored:
	python -m mlops.check_scored_dq

drift:
	python -m mlops.drift_report

promote:
	python -m mlops.promote_model --run-id $(RUN_ID)

rollback:
	python -m mlops.rollback_model
