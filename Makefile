test:
	poetry run flake8 --count
	poetry run black . --check
	poetry run pytest -v tests/resonances

test-only:
	poetry run pytest -v tests/resonances

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

