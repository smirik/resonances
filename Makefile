install:
	uv sync

test: install
	uv run flake8 --count
	uv run black . --check
	uv run pytest -v -m "not benchmark" tests

test-only: install
	uv run pytest -v -m "not benchmark" tests

test-fast: install
	uv run pytest -v -m "not slow and not benchmark" tests

test-slow: install
	uv run pytest -v -m "slow" tests

test-benchmark: install
	uv run pytest -v -m "benchmark" tests

run-docs:
	uv run mkdocs serve

publish-docs:
	rm -Rf docs/cache/*
	uv run mkdocs gh-deploy

publish-test:
	uv build
	uv publish --index-url https://test.pypi.org/legacy/

publish:
	uv build
	uv publish

coverage: install
	uv run coverage run -m pytest -v tests/resonances
	uv run coverage report -m

clean:
	rm -f cache/allnum.cat
	rm -f cache/solar.bin
	rm -f cache/*.csv
	rm -f cache/*.png

cache-clear:
	rm -f cache/*.csv
	rm -f cache/*.png
