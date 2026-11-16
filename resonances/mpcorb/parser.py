"""The MPCORB format: one orbit per fixed-column line, packed epochs, the 4-hex flags, the compact table.

Format: "Export Format for Minor-Planet Orbits", https://minorplanetcenter.net/iau/info/MPOrbitFormat.html (also
https://docs.minorplanetcenter.net/mpc-ops-docs/orbits/minor-planet-orbit-format/); MPCORB's own header points there.
Elements: heliocentric osculating, J2000 ecliptic, mu = k^2 (n = k a^-1.5 in the file), epoch in TT at 0h.
"""

import csv
import datetime
import gzip
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from .designations import BASE62, normalise_provisional, unpack_number, unpack_provisional

# The Gaussian gravitational constant (IAU), rad/day: MPC's elements have n = k a^-1.5, mu = k^2.
GAUSSIAN_K = 0.01720209895
GAUSSIAN_K_DEG = math.degrees(GAUSSIAN_K)
MJD_ZERO = datetime.datetime(1858, 11, 17)
DAYS_PER_YEAR = 365.25

# Bump when the table's columns or their meaning change: an older table is rebuilt from the raw file
PARSER_VERSION = 1
TABLE_COLUMNS = (
    'packed', 'number', 'designation', 'name', 'epoch_mjd', 'a', 'e', 'inc_deg', 'Omega_deg', 'omega_deg', 'M_deg',
    'n_deg_day', 'H', 'U', 'n_obs', 'n_opp', 'arc', 'rms', 'perturbers', 'flags', 'last_obs',
)  # fmt: skip
# Rounding of the file: n to 1e-8 deg/d, a to 1e-7 AU. A row whose |n - k a^-1.5| exceeds this many times its
# rounding bound (n 5e-9 + 1.5 n 5e-8 / a) is not a mu = k^2 orbit
MU_CHECK_FACTOR = 4.0

# Columns 162-165, "4-hexdigit flags" (MPOrbitFormat.html above). Bits 0-5 hold the orbit type ("other values are
# undefined"; 7 is "Unused or internal MPC use only", yet the 2026-10-25 MPCORB gives it to 16,289 rows, all near
# a = 2 AU, q = 2 AU). Bits 6-10 are "Unused or internal MPC use only". The classification "is based on cuts in
# osculating element space and is not 100% reliable".
ORBIT_TYPE_MASK = 0x3F
ORBIT_TYPES = {
    1: 'atira',
    2: 'aten',
    3: 'apollo',
    4: 'amor',
    5: 'q_lt_1665',  # "Object with q < 1.665 AU"
    6: 'hungaria',
    7: 'mpc_internal',  # "Unused or internal MPC use only"
    8: 'hilda',
    9: 'jupiter_trojan',
    10: 'distant',  # "Distant object"
}
FLAG_BITS = {
    'neo': 11,  # 2048, "Object is NEO"
    'km_neo': 12,  # 4096, "Object is 1-km (or larger) NEO"
    'one_opp_earlier': 13,  # 8192, "1-opposition object seen at earlier opposition"
    'critical_list': 14,  # 16384, "Critical list numbered object"
    'pha': 15,  # 32768, "Object is PHA"
}

_ARC_YEARS = re.compile(r'(\d{4})-(\d{4})')
_ARC_DAYS = re.compile(r'(\d+) +days')


# ----------------------------------------------------------------------------------------------- time
def unpack_epoch(packed: str) -> float:
    """A packed date (https://minorplanetcenter.net/iau/info/PackedDates.html) as MJD: 'K2669' = 2026-06-09 =
    MJD 61200.0, 'K01AM138303' = 2001-10-22.138303. MPCORB epochs are TT; the scale is the file's, not changed."""
    s = str(packed).strip()
    if len(s) < 5 or s[0] not in 'IJKL' or not s[1:3].isdigit() or not s[5:].isdigit() and s[5:]:
        raise ValueError(f"Not a packed date: {packed!r}")
    year = BASE62.index(s[0]) * 100 + int(s[1:3])
    day = datetime.date(year, BASE62.index(s[3]), BASE62.index(s[4]))  # ValueError on a month/day out of range
    fraction = float('0.' + s[5:]) if len(s) > 5 else 0.0
    return float(day.toordinal() - MJD_ZERO.toordinal()) + fraction


def datetime_from_mjd(mjd: float) -> datetime.datetime:
    """The calendar value of MJD `mjd` (in its own time scale; naive). Not through astropy: Time(naive) is UTC and
    `.tt` would shift a TT value by 69 s."""
    return MJD_ZERO + datetime.timedelta(days=float(mjd))


# ----------------------------------------------------------------------------------------------- fields
def arc_years(arc: str) -> float:
    """The observed arc of columns 128-136 in years: last minus first year for a multi-opposition orbit ('1801-2026'
    -> 225), days / 365.25 for a one-opposition one ('  10 days'); nan when blank or unreadable."""
    s = str(arc).strip()
    m = _ARC_YEARS.fullmatch(s)
    if m:
        return float(int(m.group(2)) - int(m.group(1)))
    m = _ARC_DAYS.fullmatch(s)
    if m:
        return int(m.group(1)) / DAYS_PER_YEAR
    return math.nan


def flags_value(flags: str) -> int:
    """The 4-hex flags as an integer ('8803' -> 34819); 0 when blank or unreadable."""
    try:
        return int(str(flags).strip(), 16)
    except ValueError:
        return 0


def decode_flags(flags) -> dict:
    """The meaning of the 4-hex flags (a string as in the file, or its integer): {'orbit_type': name or '', 'neo',
    'km_neo', 'one_opp_earlier', 'critical_list', 'pha': bool}. '8803' (2022 LX1) is an Apollo, NEO and PHA."""
    value = flags if isinstance(flags, int) else flags_value(flags)
    out = {'orbit_type': ORBIT_TYPES.get(value & ORBIT_TYPE_MASK, '')}
    out.update({name: bool(value >> bit & 1) for name, bit in FLAG_BITS.items()})
    return out


# ----------------------------------------------------------------------------------------------- parsing
def parse_line(line: str) -> Optional[dict]:
    """One MPCORB record as a dict (angles in degrees, as in the file), or None if the line is not a record."""
    if len(line.rstrip('\r\n')) < 103:
        return None
    packed = line[0:7].strip()
    try:
        record = {
            'epoch_mjd': unpack_epoch(line[20:25]),
            'M_deg': float(line[26:35]),
            'omega_deg': float(line[37:46]),
            'Omega_deg': float(line[48:57]),
            'inc_deg': float(line[59:68]),
            'e': float(line[70:79]),
            'n_deg_day': float(line[80:91]),
            'a': float(line[92:103]),
        }
    except (ValueError, IndexError):
        return None
    if not all(math.isfinite(value) for value in record.values()):
        return None  # 'nan' and 'inf' parse as floats
    number = unpack_number(packed)
    designation = '' if number else (unpack_provisional(packed) or '')
    readable = line[166:194].strip()
    name = ''
    m = re.fullmatch(r'\((\d+)\)\s*(.*)', readable)
    if m:
        rest = m.group(2).strip()
        if normalise_provisional(rest):
            designation = normalise_provisional(rest)  # a numbered object without a name: its principal designation
        else:
            name = rest
    elif readable and not number:
        designation = readable  # MPC's own text (also old-style ones such as 'A904 OA')
    record.update(
        packed=packed,
        number=number or 0,
        designation=designation,
        name=name,
        H=line[8:13].strip(),
        U=line[105:106].strip(),
        n_obs=line[117:122].strip(),
        n_opp=line[123:126].strip(),
        arc=line[127:136].strip(),
        rms=line[137:141].strip(),
        perturbers=f"{line[142:145].strip()} {line[146:149].strip()}".strip(),
        flags=line[161:165].strip(),
        last_obs=line[194:202].strip(),
    )
    return record


def _looks_like_record(line: str) -> bool:
    return len(line.rstrip()) >= 103 and not line.startswith(("Des'n", '-'))


def mu_ratio(record: dict) -> float:
    """|n - k a^-1.5| over the file's rounding bound for this row (<= 1 for mu = k^2 up to rounding)."""
    n, a = record['n_deg_day'], record['a']
    bound = 5e-9 + 1.5 * n * 5e-8 / a
    return abs(n - GAUSSIAN_K_DEG * a**-1.5) / bound


def _open_text(path: Path):
    with open(path, 'rb') as probe:
        gzipped = probe.read(2) == b'\x1f\x8b'
    if gzipped:
        return gzip.open(path, 'rt', encoding='ascii', errors='replace')
    return open(path, 'rt', encoding='ascii', errors='replace')


def build_table(raw_path, table_path) -> dict:
    """Parse the raw file (gzip or text) into the compact table; return its statistics.

    ValueError when the file holds no record, or its rows are not mu = k^2 orbits (`mu_ratio` > MU_CHECK_FACTOR on
    more than 0.1 % of them): a wrong file or a changed format, not a catalogue to integrate.
    """
    rows, unparsed, no_perturbers, mu_bad, mu_max = 0, 0, 0, 0, 0.0
    epochs = Counter()
    table_path = Path(table_path)
    with _open_text(Path(raw_path)) as source, open(table_path, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(TABLE_COLUMNS)
        for line in source:
            record = parse_line(line)
            if record is None:
                unparsed += _looks_like_record(line)
                continue
            rows += 1
            epochs[record['epoch_mjd']] += 1
            no_perturbers += record['perturbers'] == ''
            ratio = mu_ratio(record)
            mu_max = max(mu_max, ratio)
            mu_bad += ratio > MU_CHECK_FACTOR
            writer.writerow([record[column] for column in TABLE_COLUMNS])
    if rows == 0:
        raise ValueError(f"{raw_path} holds no MPCORB-format record")
    if mu_bad > 1e-3 * rows:
        raise ValueError(f"{raw_path}: {mu_bad} of {rows} rows do not satisfy n = k a^-1.5 (mu = k^2) within rounding")
    standard_epoch, standard_rows = epochs.most_common(1)[0]
    return {
        'rows': rows,
        'standard_epoch_mjd': standard_epoch,
        'standard_epoch_rows': standard_rows,
        'other_epoch_rows': rows - standard_rows,
        'other_epochs': len(epochs) - 1,
        'unparsed_lines': unparsed,
        'no_perturber_rows': no_perturbers,
        'mu_max_ratio': round(mu_max, 3),
        'mu_bad_rows': mu_bad,
    }
