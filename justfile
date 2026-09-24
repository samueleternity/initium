# initium developer tasks. `just` lists them.

default:
    @just --list

# ---- read-only checks (exactly what CI runs) ----------------------------------
# Ruff lint
lint:
    ruff check .

# Fail if any file is not ruff-formatted (never writes)
format-check:
    ruff format --check .

# Static type check (mypy)
typecheck:
    mypy

# lint + format-check + typecheck
check: lint format-check typecheck

# ---- local only, these WRITE files --------------------------------------------
# Sort imports, then format
format:
    ruff check --select I --fix --exit-zero --quiet .
    ruff format .

# Apply every safe ruff autofix
fix:
    ruff check --fix .

# Format, then lint
format-lint: format lint

# ---- tests (reserved: no-op until tests/ exists) -------------------------------
test *args:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -d tests ]; then
        pytest {{args}}
    else
        echo "tests/ not found - test suite not implemented yet, skipping."
    fi

# ---- environment ----------------------------------------------------------------
# Install CUDA-matched mamba-ssm / causal-conv1d and git pytorch-dnc (args: --dry-run, --skip ...)
setup *args:
    python -m setup_wheels {{args}}