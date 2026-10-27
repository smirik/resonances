from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from dotenv import dotenv_values
import os
import sys

from resonances.data.util import static_init


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
        merged = {**default_config, **env_vars, **user_config}
        if merged.get("LOG_LEVEL") == "debug":
            print(f"Merged config: {merged}")

        cls.config = merged


# The low-memory allocator policy of spawned worker processes (LOW_MEMORY_ALLOCATOR = auto | on | off, docs/config.md).
# macOS libmalloc keeps freed large blocks in its large cache, in the process' footprint, and reuses them only for blocks
# of a matching size: a reclassify worker's footprint p90 0.90 GB, max 2.34 GB, of which 1.26 GB was live at most
# (W23 perf: 37087 at 1 Myr). MallocLargeCache=0 returns them (p90 0.30 GB, max 1.42 GB, the same bytes) at ~+29 % CPU
# before the W23 levers (fewer allocations: L-MEMO, L-TABLE). libmalloc reads it when a process starts, so it is set in
# the environment the workers inherit. Linux (glibc) has other knobs, not measured: no policy there.
LOW_MEMORY_ALLOCATOR_MODES = ('auto', 'on', 'off')
LOW_MEMORY_ALLOCATOR_ENV = {'darwin': {'MallocLargeCache': '0'}}


def low_memory_allocator(ram_limited: bool, mode: Optional[str] = None) -> bool:
    """Whether workers get the low-memory allocator: LOW_MEMORY_ALLOCATOR (`mode`) on or off, or auto: on when memory,
    not the CPUs, limits the worker count (`ram_limited`). Always False where no policy is known (not macOS)."""
    mode = str(mode if mode is not None else config.get('LOW_MEMORY_ALLOCATOR', 'auto')).strip().lower()
    if mode not in LOW_MEMORY_ALLOCATOR_MODES:
        raise ValueError(f'LOW_MEMORY_ALLOCATOR must be one of {LOW_MEMORY_ALLOCATOR_MODES}, not {mode!r}')
    if sys.platform not in LOW_MEMORY_ALLOCATOR_ENV:
        return False
    return mode == 'on' or (mode == 'auto' and ram_limited)


@contextmanager
def worker_allocator(enabled: bool):
    """Inside: processes started from here (spawned workers) inherit the low-memory allocator environment when `enabled`;
    this process keeps its allocator. The environment is restored at the exit."""
    variables = LOW_MEMORY_ALLOCATOR_ENV.get(sys.platform, {}) if enabled else {}
    saved = {name: os.environ.get(name) for name in variables}
    os.environ.update(variables)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
