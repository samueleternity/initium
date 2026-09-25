# Test scope notes

Tests use the existing `src/model` source tree through pytest's `pythonpath` setting because production modules currently import sibling packages as top-level names. The planned move to `src/initium` is intentionally deferred; this suite does not attempt that package migration or introduce a `conftest.py` path mutation.

Mypy remains scoped to `src/model` for now. Test code is runtime-checked by pytest and linted/formatted with the repository-wide Ruff commands; expanding strict mypy coverage to tests is a separate decision because most assertions intentionally exercise third-party DNC APIs with loose typing.
