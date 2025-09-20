
.PHONY: api test docker-build docker-run

api:
	uvicorn app.main:app --reload --port 8000

test:
	pytest -q

docker-build:
	docker build -t ml-pipeline-lite:latest .

docker-run:
	docker run -p 8000:8000 ml-pipeline-lite:latest
