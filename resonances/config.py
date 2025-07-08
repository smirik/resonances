from pathlib import Path
from dotenv import dotenv_values
import os


def static_init(cls):
    if getattr(cls, "static_init", None):
        cls.static_init()
    return cls


@static_init
class config:
    config = {}

    @classmethod
    def get(cls, key, default=None):
        try:
            value = cls.config[key]
        except KeyError:
            if default is not None:
                return default
            raise Exception(f'There is no config with key = {key}. The full config: {cls.config}')
        return value

    @classmethod
    def has(cls, key):
        return key in cls.config

    @classmethod
    def set(cls, key, value):
        if not cls.has(key):
            raise Exception(f'There is no config with key = {key}. The full config: {cls.config}')
        cls.config[key] = value

    @classmethod
    def static_init(cls):
        """
        1) Load default values from .env.dist (in the package).
        2) Override with real OS environment variables.
        2) Override them with .env (in current working directory), if present.
        """
        package_env_path = Path(__file__).parent / ".env.dist"
        user_env_path = Path.cwd() / ".env"

        # 1. Load defaults from your package's .env.dist
        if not package_env_path.exists():  # pragma: no cover
            raise FileNotFoundError(f"Missing .env.dist at: {package_env_path}")
        default_config = dotenv_values(package_env_path)

        # 2. Load user-local .env if available
        user_config = {}
        if user_env_path.exists():  # pragma: no cover
            user_config = dotenv_values(user_env_path)

        # 3. Get actual environment variables
        # (os.environ is a live mapping; turn it into a dict copy here)
        env_vars = dict(os.environ)

        # Merge them: left to right means the rightmost wins in conflicts
        #   default_config  <  env_vars < user_config
        merged = {**user_config, **default_config, **env_vars, **user_config}
        if merged.get("LOG_LEVEL") == "debug":
            print(f"Merged config: {merged}")

        cls.config = merged
