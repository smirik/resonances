test:
	poetry run flake8 --count
	poetry run black . --check
	poetry run pytest -v tests/resonances