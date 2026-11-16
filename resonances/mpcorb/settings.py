"""Settings of the catalogue: which file, where it is cached, and the checks on it.

Every public function takes them as `settings` (and `catalog`/`cache_dir`, which override those fields). Without
them it uses the defaults, `Settings()` unless the host set others with `configure`: a `Settings`, or a callable that
returns one at every call (a host reading its own configuration file, e.g. `resonances` and its MPC_* keys).
"""

import dataclasses
from dataclasses import dataclass
from typing import Callable, Optional, Union


@dataclass(frozen=True)
class Settings:
    # A preset of `CATALOGS` (mpcorb, nea, neam00, distant, unusual) or a URL (http(s):// or file://) of a file in the
    # MPCORB format
    catalog: str = 'mpcorb'
    # The catalogue lives in <cache_dir>/<catalog key>/ (a relative path is taken from the working directory)
    cache_dir: str = 'cache/mpc'
    # When set, any other snapshot is refused (the sha256 of the downloaded file)
    sha256: str = ''
    # A snapshot older than this (days since its Last-Modified) warns; 0 = never
    max_age_days: float = 30.0
    # `quality_warnings` reports orbits with an uncertainty parameter U at least this
    warn_u: int = 6
    # How the old-snapshot warning tells the user to refresh it
    update_hint: str = '`mpcorb.update_catalog()`'
    user_agent: str = 'mpcorb (python)'
    timeout_s: float = 120.0


_defaults: Union[Settings, Callable[[], Settings]] = Settings()


def configure(settings: Union[Settings, Callable[[], Settings], None] = None, **changes) -> None:
    """Set the defaults: a `Settings`, a callable returning one (read at every call), or changes of some fields
    (`configure(catalog='nea')`; applied to the current defaults, a callable's value included)."""
    global _defaults
    if settings is not None and changes:
        raise TypeError("configure takes a Settings (or a callable) or field changes, not both")
    if settings is None:
        settings = dataclasses.replace(current(), **changes)
    if not (isinstance(settings, Settings) or callable(settings)):
        raise TypeError(f"Not mpcorb settings: {settings!r}")
    _defaults = settings


def current() -> Settings:
    """The default settings now."""
    return _defaults() if callable(_defaults) else _defaults


def resolve(settings: Optional[Settings] = None, catalog: Optional[str] = None, cache_dir: Optional[str] = None) -> Settings:
    """`settings` (else the defaults) with `catalog` and `cache_dir` when given."""
    base = settings if settings is not None else current()
    changes = {key: value for key, value in (('catalog', catalog), ('cache_dir', cache_dir)) if value}
    return dataclasses.replace(base, **changes) if changes else base
