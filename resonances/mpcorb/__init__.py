"""The Minor Planet Center's orbit catalogue MPCORB (or a subset in its format), cached locally, searchable.

    import mpcorb                           # inside the resonances package: from resonances import mpcorb

    mpcorb.search(463)                      # elements of (463) Lola: a, e, angles in radians, epoch MJD TT
    mpcorb.search([1, '2001 ND13'])         # {identifier: elements}
    mpcorb.search_by_axis(2.4, sigma=0.01)  # DataFrame, a within 2.39..2.41 AU (as astdys.search_by_axis)
    mpcorb.query(a=(0, 1), H=(None, 20))    # DataFrame, every criterion of `query`
    mpcorb.update_catalog()                 # the only refresh (a conditional GET on the ETag)

The bulk file MPCORB.DAT.gz (or NEA.txt, Distant.txt, ...) is downloaded on first use into
<cache_dir>/<catalog>/ with its provenance (snapshot.json: URL, ETag, Last-Modified, sha256, standard epoch), see
`cache`. Elements (https://minorplanetcenter.net/iau/info/MPOrbitFormat.html): heliocentric osculating, J2000
ecliptic, mu = k^2, epoch in TT at 0h, returned as MJD TT. Most rows share the standard epoch; the one-opposition
orbits at the end of MPCORB do not. Designations: `designations`.

Self-contained (the standard library, numpy and pandas): settings come as arguments or `Settings` (`configure` sets
the defaults), messages go to the standard logger 'mpcorb'.
"""

from .settings import Settings, configure, current as current_settings, resolve as resolve_settings
from .designations import normalise_provisional, pack_number, pack_provisional, unpack_number, unpack_provisional
from .parser import (
    DAYS_PER_YEAR,
    FLAG_BITS,
    GAUSSIAN_K,
    GAUSSIAN_K_DEG,
    MJD_ZERO,
    MU_CHECK_FACTOR,
    ORBIT_TYPES,
    PARSER_VERSION,
    TABLE_COLUMNS,
    arc_years,
    build_table,
    datetime_from_mjd,
    decode_flags,
    flags_value,
    mu_ratio,
    parse_line,
    unpack_epoch,
)
from .cache import (
    CATALOGS,
    LOAD_COLUMNS,
    PROVENANCE_KEYS,
    STALE_PART_S,
    catalog_datetime,
    catalog_folder,
    catalog_url,
    clear_cache,
    load_table,
    provenance,
    query_columns,
    read_snapshot,
    snapshot,
    snapshot_age_days,
    update_catalog,
)
from .lookup import (
    elements_from_row,
    get_catalog_datetime,
    load,
    lookup,
    quality_warnings,
    query_keys,
    rebuild,
    row_label,
    search,
)
from .query import RANGE_COLUMNS, get_catalog, identifiers, query, search_by_axis

__version__ = '0.1.0'
