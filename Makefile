.PHONY: help install install-dev test test-unit test-integration test-e2e test-coverage lint format type-check clean

PYTHON := python
PIP := pip

help:
	@echo "PyUT Agent - Available commands:"
	@echo ""
	@echo "  install         Install the package"
	@echo "  install-dev     Install the package with development dependencies"
	@echo ""
	@echo "  test            Run all tests (unit + integration)"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-e2e        Run end-to-end tests"
	@echo "  test-coverage   Run tests with coverage report"
	@echo "  test-slow       Run slow tests"
	@echo ""
	@echo "  lint            Run all linters (ruff, black, mypy)"
	@echo "  format          Format code with black and ruff"
	@echo "  type-check      Run type checking with mypy"
	@echo ""
	@echo "  clean           Clean build artifacts"
	@echo ""

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	$(PIP) install ruff black mypy pytest-cov pytest-html

# Test commands
test: test-unit test-integration

test-unit:
	pytest tests/unit -v --tb=short -m "not slow"

test-integration:
	pytest tests/integration -v --tb=short -m "not slow"

test-e2e:
	pytest tests/test_e2e.py -v --tb=short

test-coverage:
	pytest tests/unit tests/integration -v --tb=short \
		--cov=pyutagent \
		--cov-report=xml \
		--cov-report=html \
		--cov-report=term-missing

test-slow:
	pytest tests -v --tb=short -m "slow"

test-all:
	pytest tests -v --tb=short

# Lint and format commands
lint:
	ruff check pyutagent tests
	black --check pyutagent tests
	mypy pyutagent --ignore-missing-imports

format:
	black pyutagent tests
	ruff check --fix pyutagent tests

type-check:
	mypy pyutagent --ignore-missing-imports

# Clean command
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -f .coverage
	rm -f coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
