.PHONY: venv test test-fast test-kafka fixture demo up down
venv:
	python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'
fixture:
	.venv/bin/python tests/make_fixture.py
test:
	.venv/bin/python -m pytest -q
test-fast:   # seconds, no models
	.venv/bin/python -m pytest -q tests/test_units.py tests/test_features.py
test-kafka:  # needs a broker: see README section 6
	KAFKA_BOOTSTRAP=localhost:19092 .venv/bin/python -m pytest -q tests/test_workers.py
demo: fixture
	.venv/bin/aikyam-video process tests/fixtures/sample-temple.mp4 -o output --temple-id temple_123
up:
	docker compose up -d --build
down:
	docker compose down
