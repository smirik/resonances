"""MPC's packed and unpacked minor-planet designations.

Numbers 00001..99999, A0000..z9999 (100000..619999), ~0000..~zzzz (620000 + base-62); provisional K01N13D =
2001 ND13, the extended scheme _QC0000 = 2026 CA620 (order > 15500 in a half-month), surveys PLS2040 = 2040 P-L
(https://minorplanetcenter.net/iau/info/PackedDes.html and the MPC's provisional-designations page).
"""

import re
from typing import Optional

BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# Second letters of a provisional designation (no I): order = 25 * cycle + index + 1
SECOND_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
# The original packed scheme holds 25 * 620 = 15500 designations per half-month; the extended one counts from 15501
EXTENDED_ORDER_START = 15501
SURVEYS = {'PLS': 'P-L', 'T1S': 'T-1', 'T2S': 'T-2', 'T3S': 'T-3'}


def _base62(value: int, width: int) -> str:
    digits = []
    for _ in range(width):
        value, rest = divmod(value, 62)
        digits.append(BASE62[rest])
    if value:
        raise ValueError("too large for base-62 of this width")
    return ''.join(reversed(digits))


def _from_base62(text: str) -> int:
    value = 0
    for ch in text:
        value = value * 62 + BASE62.index(ch)
    return value


def pack_number(number: int) -> str:
    """(463) -> '00463', (101429) -> 'A1429', (630872) -> '~02pM'."""
    number = int(number)
    if number < 1:
        raise ValueError(f"A minor-planet number is positive, got {number}")
    if number < 100000:
        return f"{number:05d}"
    if number < 620000:
        return BASE62[number // 10000] + f"{number % 10000:04d}"
    return '~' + _base62(number - 620000, 4)


def unpack_number(packed: str) -> Optional[int]:
    """'00463' -> 463, 'A1429' -> 101429, '~02pM' -> 630872; None if `packed` is not a packed number."""
    s = str(packed).strip()
    if len(s) != 5:
        return None
    if s[0] == '~':
        if not all(ch in BASE62 for ch in s[1:]):
            return None
        return 620000 + _from_base62(s[1:])
    if s.isdigit():
        return int(s) if int(s) > 0 else None
    if s[0].isalpha() and s[0].isascii() and s[1:].isdigit():
        return BASE62.index(s[0]) * 10000 + int(s[1:])
    return None


_UNPACKED_PROVISIONAL = re.compile(r'([12]\d{3}|A\d{3}) ?([A-HJ-Y])([A-HJ-Z])(\d*)')
_UNPACKED_SURVEY = re.compile(r'(\d{1,4}) ?(P-L|T-1|T-2|T-3)')
_PACKED_PROVISIONAL = re.compile(r'([I-L])(\d\d)([A-HJ-Y])([0-9A-Za-z])(\d)([A-HJ-Z])')
_PACKED_EXTENDED = re.compile(r'_([A-Z])([A-HJ-Y])([0-9A-Za-z]{4})')
# Before 1925 the year's first digit is written 'A': 1904 OA is 'A904 OA' (packed J04O00A)
FIRST_MODERN_YEAR = 1925


def _year_text(year: int) -> str:
    return f"A{year % 1000:03d}" if year < FIRST_MODERN_YEAR else str(year)


def _year_value(text: str) -> int:
    return 1000 + int(text[1:]) if text[0] == 'A' else int(text)


def normalise_provisional(designation: str) -> Optional[str]:
    """MPC's own spelling of a provisional designation: '2001ND13' or '2001 ND13' -> '2001 ND13', '1900 UK' ->
    'A900 UK', '4150T-3' -> '4150 T-3'; None if `designation` is not one."""
    s = str(designation).strip()
    m = _UNPACKED_PROVISIONAL.fullmatch(s)
    if m:
        cycle = m.group(4).lstrip('0')
        if m.group(4) and not cycle:  # '2001 ND0' is not a designation
            return None
        return f"{_year_text(_year_value(m.group(1)))} {m.group(2)}{m.group(3)}{cycle}"
    m = _UNPACKED_SURVEY.fullmatch(s)
    if m:
        return f"{int(m.group(1))} {m.group(2)}"
    return None


def pack_provisional(designation: str) -> Optional[str]:
    """'2001 ND13' -> 'K01N13D', '2026 CA620' -> '_QC0000', '2040 P-L' -> 'PLS2040'; None if not packable."""
    s = normalise_provisional(designation)
    if s is None:
        return None
    m = _UNPACKED_SURVEY.fullmatch(s)
    if m:
        prefix = {v: k for k, v in SURVEYS.items()}[m.group(2)]
        return f"{prefix}{int(m.group(1)):04d}"
    m = _UNPACKED_PROVISIONAL.fullmatch(s)
    year, half_month, letter = _year_value(m.group(1)), m.group(2), m.group(3)
    cycle = int(m.group(4) or 0)
    if not 1800 <= year <= 2199:
        return None
    if cycle < 620:
        return f"{BASE62[year // 100]}{year % 100:02d}{half_month}{BASE62[cycle // 10]}{cycle % 10}{letter}"
    order = 25 * cycle + SECOND_LETTERS.index(letter) + 1
    if not 2010 <= year <= 2035:  # the extended scheme: '_' means 20xx, year letter A..Z = 10..35
        return None
    return f"_{BASE62[year - 2000]}{half_month}{_base62(order - EXTENDED_ORDER_START, 4)}"


def unpack_provisional(packed: str) -> Optional[str]:
    """'K01N13D' -> '2001 ND13', '_QC0000' -> '2026 CA620', 'PLS2040' -> '2040 P-L', 'J00U00K' -> 'A900 UK'; None
    if `packed` is not a packed provisional designation."""
    s = str(packed).strip()
    if len(s) != 7:
        return None
    if s[:3] in SURVEYS and s[3:].isdigit():
        return f"{int(s[3:])} {SURVEYS[s[:3]]}"
    m = _PACKED_EXTENDED.fullmatch(s)
    if m:
        order = EXTENDED_ORDER_START + _from_base62(m.group(3))
        cycle, index = divmod(order - 1, 25)
        return f"{2000 + BASE62.index(m.group(1))} {m.group(2)}{SECOND_LETTERS[index]}{cycle}"
    m = _PACKED_PROVISIONAL.fullmatch(s)
    if m:
        year = BASE62.index(m.group(1)) * 100 + int(m.group(2))
        cycle = BASE62.index(m.group(4)) * 10 + int(m.group(5))
        return f"{_year_text(year)} {m.group(3)}{m.group(6)}{cycle if cycle else ''}"
    return None
