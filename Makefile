.PHONY: install install-dev lint format test train serve docker-build docker-up docker-down clean help

# ── Variables ─────────────────────────────────────────────────────────────────
PYTHON   := python
UV       := uv
SRC      := src
TESTS    := tests
IMAGE    := kdrama-compass
TAG      := latest

# ── Setup ─────────────────────────────────────────────────────────────────────

install:          ## Install production dependencies via uv
	$(UV) pip install .

install-dev:      ## Install all dependencies including dev extras
	$(UV) pip install ".[dev]"
	pre-commit install

# ── Code quality ──────────────────────────────────────────────────────────────

lint:             ## Run Ruff linter
	ruff check $(SRC)/ $(TESTS)/

format:           ## Auto-format code with Ruff
	ruff format $(SRC)/ $(TESTS)/
	ruff check --fix $(SRC)/ $(TESTS)/

# ── Tests ─────────────────────────────────────────────────────────────────────

test:             ## Run the full test suite with coverage
	pytest $(TESTS)/ -v --tb=short --cov=$(SRC) --cov-report=term-missing

test-unit:        ## Run unit tests only
	pytest $(TESTS)/unit/ -v --tb=short

test-integration: ## Run integration tests only
	pytest $(TESTS)/integration/ -v --tb=short

# ── ML pipeline ───────────────────────────────────────────────────────────────

train:            ## Run the full training pipeline
	$(PYTHON) scripts/run_pipeline.py

# ── API server ────────────────────────────────────────────────────────────────

serve:            ## Start the API server (development, hot-reload)
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

serve-prod:       ## Start the API server (production)
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:     ## Build the Docker image
	docker build -f docker/Dockerfile -t $(IMAGE):$(TAG) .

docker-up:        ## Start all services (API + MLflow) via docker-compose
	docker compose -f docker/docker-compose.yml up -d

docker-down:      ## Stop all Docker services
	docker compose -f docker/docker-compose.yml down

docker-logs:      ## Tail logs from all Docker services
	docker compose -f docker/docker-compose.yml logs -f

# ── Utilities ─────────────────────────────────────────────────────────────────

clean:            ## Remove build artefacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage coverage.xml htmlcov/ dist/ build/

help:             ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
