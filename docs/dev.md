# Development Documentation

## To release a new version

1. Set a new version in `pyproject.toml`.
2. Set the production token:

    ```bash
    poetry config pypi-token.pypi TOKEN
    ```

3. Run tests

    ```bash
    make test
    ```

4. Publish

    ```bash
    make publish
    ```

5. Update docs if necessary

    ```bash
    make publish-docs
    ```

For pypi test, the test token should be set:

    ```bash
    poetry config pypi-token.testpypi TEST-TOKEN
    ```
