test:
	poetry run flake8 --count
	poetry run black . --check
	poetry run pytest -v tests/resonances

docs:
	poetry run mkdocs serve

publish-docs:
	poetry run mkdocs gh-deploy

publish-test:
	poetry build
	poetry config repositories.testpypi https://test.pypi.org/legacy/
	# poetry config pypi-token.testpypi your-test-api-token
	poetry publish -r testpypi

publish:
	poetry publish --build

set-token:
	poetry config pypi-token.pypi your-api-token

