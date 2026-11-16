# Development Documentation

## To release a new version

1. Set a new version in `pyproject.toml`.
2. Run tests

    ```bash
    make test
    ```

3. Build and publish

    ```bash
    make publish
    ```

    This runs `uv build` and `uv publish`. You will need to set the `UV_PUBLISH_TOKEN` environment variable with your PyPI token, or pass it via `--token`:

    ```bash
    UV_PUBLISH_TOKEN=your-token make publish
    ```

4. Update docs if necessary

    ```bash
    make publish-docs
    ```

For pypi test:

```bash
UV_PUBLISH_TOKEN=your-test-token make publish-test
```
