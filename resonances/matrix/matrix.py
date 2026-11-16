import json
import os
from pathlib import Path

import pandas as pd

import resonances
from resonances.config import config
from resonances.data import const
from resonances.logger import logger
from resonances.resonance.factory import create_mmr


class Matrix:
    catalog_file = ''
    # Bump when build() changes what the catalog holds: a cached file with another stamp is rebuilt.
    catalog_version = 1
    matrix = None
    planets = None

    # Subclasses define which columns hold planet names for filtering
    planet_columns = []
    # The config keys whose values shape the catalog (its limits); recorded in the build record
    limit_keys = ()
    # (file, message) pairs already warned about in this process: one warning per cached file
    _warned = set()

    @classmethod
    def catalog_planets(cls) -> list:
        """The planets the catalog is built for: `planets`, or the whole Solar system when it is unset or empty."""
        return list(cls.planets) if cls.planets else list(const.SOLAR_SYSTEM)

    @classmethod
    def build_parameters(cls) -> dict:
        """Everything that shapes the catalog (limits, planets, catalog version) and the package version (provenance)."""
        return {
            'catalog': cls.__name__,
            'catalog_version': cls.catalog_version,
            'limits': {key: int(config.get(key)) for key in cls.limit_keys},
            'planets': cls.catalog_planets(),
            'package_version': resonances.__version__,
        }

    @staticmethod
    def meta_filename(catalog_file) -> Path:
        """The build record next to a cached catalog: `<file>.meta.json`."""
        catalog_file = Path(catalog_file)
        return catalog_file.with_name(catalog_file.name + '.meta.json')

    @classmethod
    def dump(cls):
        """Write the stamped catalog to a temporary file and move it into place, so a reader never sees half a table."""
        if cls.matrix is None:
            cls.build()

        target = Path(cls.catalog_full_filename())
        tmp = target.with_name(f'{target.name}.{os.getpid()}.tmp')
        try:
            with open(tmp, 'w') as f:
                f.write(cls._stamp() + '\n')
                cls.matrix.to_csv(f)
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        cls._write_meta(target)

    @classmethod
    def _write_meta(cls, catalog_file: Path):
        """Write the build record of a freshly dumped catalog, atomically (temporary file, then moved into place)."""
        meta = cls.meta_filename(catalog_file)
        tmp = meta.with_name(f'{meta.name}.{os.getpid()}.tmp')
        try:
            with open(tmp, 'w') as f:
                json.dump(cls.build_parameters(), f, indent=2)
                f.write('\n')
            os.replace(tmp, meta)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    @classmethod
    def build_mismatches(cls, catalog_file) -> list | None:
        """The settings in which a cached catalog's build record differs from the current ones, as
        (setting, cached, current); None when the file has no readable record (its limits are unknown).
        The package version alone is provenance, not a mismatch: the catalog version stamp covers format changes."""
        try:
            with open(cls.meta_filename(catalog_file)) as f:
                cached = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(cached, dict):
            return None
        current = cls.build_parameters()
        cached_limits = cached.get('limits') or {}
        diffs = []
        for key, value in current['limits'].items():
            if cached_limits.get(key) != value:
                diffs.append((key, cached_limits.get(key), value))
        if sorted(cached.get('planets') or []) != sorted(current['planets']):
            diffs.append(('planets', cached.get('planets'), current['planets']))
        if diffs and cached.get('package_version') != current['package_version']:
            diffs.append(('package_version', cached.get('package_version'), current['package_version']))
        return diffs

    @classmethod
    def check_cached(cls, catalog_file) -> str | None:
        """Warn (once per file and message) when a cached catalog was built with other settings, or has no build
        record. Only a warning: the file is read as it is, nothing is rebuilt or deleted. Returns the message."""
        catalog_file = Path(catalog_file)
        diffs = cls.build_mismatches(catalog_file)
        rebuild = (
            f'It is read as it is. To rebuild it with the current settings, delete {catalog_file} '
            f'(or call {cls.__name__}.load(reload=True)); or point {cls.catalog_file} to another file.'
        )
        if diffs is None:
            current = ', '.join(f'{key}={value}' for key, value in cls.build_parameters()['limits'].items())
            message = (
                f'MMR catalog {catalog_file} has no build record ({cls.meta_filename(catalog_file).name}): the limits '
                f'it was built with are unknown and may differ from the current {current}. {rebuild}'
            )
        elif diffs:
            changes = '; '.join(f'{key}: cached {cached}, current {value}' for key, cached, value in diffs)
            message = f'MMR catalog {catalog_file} was built with other settings ({changes}). {rebuild}'
        else:
            return None
        if (str(catalog_file), message) not in Matrix._warned:
            Matrix._warned.add((str(catalog_file), message))
            logger.warning(message)
        return message

    @classmethod
    def _stamp(cls) -> str:
        """First line of the cached catalog: the class and its catalog version."""
        return f'# {cls.__name__} v{cls.catalog_version}'

    @classmethod
    def _cache_is_current(cls, catalog_file: Path) -> bool:
        if not catalog_file.exists():
            return False
        with open(catalog_file) as f:
            return f.readline().rstrip('\n') == cls._stamp()

    @classmethod
    def catalog_full_filename(cls) -> str:
        """
        If the config value is an absolute path (starts with '/'), use it as is.
        Otherwise, interpret it relative to the current working directory.
        """
        filename = config.get(cls.catalog_file)
        path_obj = Path(filename)

        # If it's not already absolute, prepend current working directory
        if not path_obj.is_absolute():
            path_obj = Path(os.getcwd()) / path_obj

        return str(path_obj)

    @classmethod
    def load(cls, reload=False):
        """Read the cached catalog, or build and cache it when missing, stale or `reload` is set.

        A cached catalog is checked against its build record (`<file>.meta.json`, see `check_cached`): other limits
        or planets, or no record, give a warning; the file is still the one read."""
        catalog_file = Path(cls.catalog_full_filename())
        if reload or not cls._cache_is_current(catalog_file):
            cls.matrix = None
            cls.dump()
        else:
            cls.matrix = pd.read_csv(catalog_file, skiprows=1)
            cls.check_cached(catalog_file)

    @classmethod
    def find_resonances(cls, a, sigma=0.1, planets=None):
        """Find resonances near semi-major axis `a` within `sigma` AU."""
        if cls.matrix is None:
            cls.load()

        df = cls.matrix[(cls.matrix['a'] >= (a - sigma)) & (cls.matrix['a'] <= (a + sigma))]

        if isinstance(planets, list) and cls.planet_columns:
            for col in cls.planet_columns:
                df = df[df[col].isin(planets)]

        return [create_mmr(mmr) for mmr in df['mmr'].tolist()]
