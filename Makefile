.PHONY: help dev test lint build docker-up docker-down bench clean

GATEWAY_DIR := gateway
PROXY_DIR   := pqc-proxy

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Development ──────────────────────────────────────────────
dev: ## Start all services with hot reload
	docker compose up --build

dev-gateway: ## Run gateway locally (requires postgres + redis)
	cd $(GATEWAY_DIR) && uvicorn llm_security_gateway.main:app --reload --host 0.0.0.0 --port 8000

dev-proxy: ## Run PQC proxy locally (requires liboqs)
	cd $(PROXY_DIR) && go run ./cmd/proxy

# ── Testing ───────────────────────────────────────────────────
test: test-python test-go ## Run all tests

test-python: ## Run Python tests with coverage
	cd $(GATEWAY_DIR) && python -m pytest tests/ -v --cov=llm_security_gateway --cov-report=term-missing --cov-fail-under=80

test-go: ## Run Go tests (requires liboqs — use Docker in CI)
	cd $(PROXY_DIR) && go test -race -count=1 ./...

test-go-docker: ## Run Go tests inside Docker (for CI / Windows)
	docker compose run --rm pqc-proxy go test -race ./...

bench: ## Run benchmarks
	cd $(GATEWAY_DIR) && python -m pytest benchmarks/ -v --benchmark-only
	cd $(PROXY_DIR) && go test -bench=. -benchmem ./...

# ── Linting ───────────────────────────────────────────────────
lint: lint-python lint-go ## Run all linters

lint-python: ## Lint Python code
	cd $(GATEWAY_DIR) && ruff check src/ tests/
	cd $(GATEWAY_DIR) && mypy src/
	cd $(GATEWAY_DIR) && bandit -r src/ -c pyproject.toml

lint-go: ## Lint Go code
	cd $(PROXY_DIR) && golangci-lint run ./...

fmt: ## Format all code
	cd $(GATEWAY_DIR) && ruff format src/ tests/
	cd $(PROXY_DIR) && gofmt -w .

# ── Build ─────────────────────────────────────────────────────
build: build-python build-go ## Build all artifacts

build-python: ## Build Python wheel
	cd $(GATEWAY_DIR) && python -m build

build-go: ## Build Go binary (Linux, requires liboqs)
	cd $(PROXY_DIR) && CGO_ENABLED=1 go build -o bin/pqc-proxy ./cmd/proxy

build-docker: ## Build all Docker images
	docker compose build

# ── Docker ────────────────────────────────────────────────────
docker-up: ## Start all services
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-logs: ## Follow logs for all services
	docker compose logs -f

docker-clean: ## Remove containers, volumes, and images
	docker compose down -v --rmi local

# ── Database ──────────────────────────────────────────────────
migrate: ## Run Alembic migrations
	cd $(GATEWAY_DIR) && alembic upgrade head

migrate-new: ## Create a new migration (usage: make migrate-new MSG="add table")
	cd $(GATEWAY_DIR) && alembic revision --autogenerate -m "$(MSG)"

migrate-down: ## Rollback last migration
	cd $(GATEWAY_DIR) && alembic downgrade -1

# ── Utilities ─────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	cd $(PROXY_DIR) && rm -rf bin/
