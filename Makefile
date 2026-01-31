install:
	pip uninstall resonances -y && poetry install

test: install
	poetry run flake8 --count
	poetry run black . --check
	poetry run pytest -v -m "not benchmark" tests

test-only: install
	poetry run pytest -v -m "not benchmark" tests

test-fast: install
	poetry run pytest -v -m "not slow and not benchmark" tests

test-slow: install
	poetry run pytest -v -m "slow" tests

test-benchmark: install
	poetry run pytest -v -m "benchmark" tests

run-docs:
	poetry run mkdocs serve

publish-docs:
	rm -Rf docs/cache/*
	poetry run mkdocs gh-deploy

publish-test:
	poetry build
	poetry config repositories.testpypi https://test.pypi.org/legacy/
	poetry publish -r testpypi

publish:
	poetry publish --build

coverage: install
	poetry run coverage run -m pytest -v tests/resonances
	poetry run coverage report -m

clean:
	rm -f cache/allnum.cat
	rm -f cache/solar.bin
	rm -f cache/*.csv
	rm -f cache/*.png

cache-clear:
	rm -f cache/*.csv
	rm -f cache/*.png
