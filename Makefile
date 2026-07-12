.PHONY: help build up down restart test lint format logs check-dags clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build all Docker container images for Lakehouse DataOps
	docker compose build

up: ## Start the full production DataOps Lakehouse stack in detached mode
	docker compose up -d

down: ## Stop and tear down all running services and volumes
	docker compose down -v

restart: down up ## Restart the entire Docker Compose stack cleanly

test: ## Run unit, integration, and DAG integrity tests via pytest
	pytest tests/ -v --tb=short

lint: ## Run Ruff linter across the entire project
	ruff check dags/ src/ api/ tests/

format: ## Auto-format Python codebase via Ruff
	ruff check --fix dags/ src/ api/ tests/
	ruff format dags/ src/ api/ tests/

check-dags: ## Validate Airflow DAGs for cycles and syntax errors
	pytest tests/dags/test_dag_integrity.py -v

logs: ## Tail real-time logs from Airflow Scheduler and API
	docker compose logs -f airflow-scheduler fastapi

clean: ## Remove Python cache files, Pytest artifacts, and build folders
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf dist build *.egg-info
