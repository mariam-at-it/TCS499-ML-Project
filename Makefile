API_DIR=api
TRAINER_DIR=trainer
WEB_DIR=web

.PHONY: api-dev trainer-dev web-dev api-test trainer-test web-test test fixture-stack fixture-evaluate cer-dev-stack cer-download cer-dev-bootstrap smart-meter-bootstrap

api-dev:
	cd $(API_DIR) && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

trainer-dev:
	cd $(TRAINER_DIR) && uv run python -m trainer.cli.main worker

web-dev:
	cd $(WEB_DIR) && npm run dev -- --host 0.0.0.0 --port 5173

api-test:
	cd $(API_DIR) && uv run pytest

trainer-test:
	cd $(TRAINER_DIR) && uv run pytest

web-test:
	cd $(WEB_DIR) && npm run test -- --run

test: api-test trainer-test web-test

fixture-stack:
	docker compose --profile fixture up --build

fixture-evaluate:
	cd $(TRAINER_DIR) && APP_PROFILE=fixture uv run python -m trainer.cli.main evaluate-fixture --publish-dp-on

cer-dev-stack:
	docker compose --profile cer-dev up --build

cer-download:
	cd $(TRAINER_DIR) && uv run python -m trainer.cli.main download-cer

cer-dev-bootstrap:
	cd $(TRAINER_DIR) && APP_PROFILE=cer-dev uv run python -m trainer.cli.main bootstrap-cer --profile cer-dev

smart-meter-bootstrap:
	cd $(TRAINER_DIR) && APP_PROFILE=smart-meters uv run python -m trainer.cli.main bootstrap-smart-meters
