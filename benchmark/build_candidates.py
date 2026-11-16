"""Merge the raw sweep inventories (raw/*.json) into candidates.csv.

One row per (name, resonance). Author verdicts win over algorithmic labels.
Near-identical algorithmic rows (same rtype, resonance, status, no author verdict)
are thinned to KEEP per group; the dropped ones go to dropped.csv.
"""

import csv
import glob
import json
import os
import re
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KEEP = 5

SOURCE_ORDER = [
    "tests",
    "old-benchmark",
    "claude-sessions-late-sep",
    "codex-and-remember",
    "nea-mmrs",
    "claude-sessions-oct",
    "claude-sessions-sep",
    "docs-misc",
    "secular",
    "zlk",
]
REL_RANK = {"author": 5, "adjudicated": 4, "literature": 3, "algorithmic": 2, "model": 1, "unknown": 0}

PLACEHOLDERS = {"2016 AA165x", "2054509", "2054509x"}

NAME_MAP = {
    "1999 UJ7 (121514)": "121514",
    "2014 EK24 (459872)": "459872",
    "2014 EK24": "459872",
    "2008 NP3": "613995",
    "1998 UP1": "85770",
    "624 Hektor": "624",
    "2010 TK7": "706765",
    "26420 (1999 XL103) / 2012 TS209": "26420",
    "80218 (1999 VO123) / 213471": "80218",
    "4765 / 350716 (pair 5)": "4765",
    "9068 / 455327 (pair 12)": "9068",
    "19289 and 278067 (test-v5 20 Myr)": "19289",
    "52773 / 279865": "52773",
    "313701 / 2012 KL9": "313701",
}

GROUP_HINTS = (
    "legacy-3",
    "set",
    "objects",
    "pilot10",
    "Toro/",
    "lkr-ias15",
    "NEA ZLK",
    "18916 (and",
    "v4 ",
    "v1 ",
    "v5 ",
    "planetary mode",
    "Lidov-Kozai in",
    "_elements_sample",
    "other angles",
    "old run",
)

LK_NAMES = {"lk", "lkr", "omega", "omega (lk)", "lk (omega)", "omega (lidov-kozai)"}

SECULAR_ALIASES = {"z1": "g-g6+s-s6", "z2": "2(g-g6)+(s-s6)", "nu6/z (astdys reference)": "g-3g6+2g5"}

RESONANCE_FIX = {
    ("9068", "all 132 secular angles"): "all secular angles",
    ("35762", "unknown"): "36M-73+0+37",
    ("1", ""): "g-g6+s-s6",
    ("3527", ""): "2(g-g6)+(s-s6)",
    ("2005 GQ107", ""): "g-2g2+g8",
}

R2 = re.compile(r"^(\d+)([A-Z])([+-]\d+)(?:([+-]\d+)([+-]\d+))?$")
R3 = re.compile(r"^(\d+)([A-Z])([+-]\d+)([A-Z])([+-]\d+)(?:([+-]\d+)([+-]\d+)([+-]\d+))?$")


def sgn(x):
    return ("+" if x >= 0 else "-") + str(abs(x))


def norm_resonance(name, res):
    res = (res or "").strip()
    if (name, res) in RESONANCE_FIX:
        return RESONANCE_FIX[(name, res)]
    low = res.lower()
    if low in LK_NAMES:
        return "LK"
    res = re.sub(r"\s*\((z1|z2|nu6|nu16|nu5)\)\s*$", "", res)
    if res.lower() in SECULAR_ALIASES:
        return SECULAR_ALIASES[res.lower()]
    m = R2.match(res)
    if m:
        a, p, b = int(m.group(1)), m.group(2), int(m.group(3))
        # d'Alembert: a + b + c = 0, the eccentricity term carries -(a+b)
        return "%d%s%s+0%s" % (a, p, sgn(b), sgn(-(a + b)))
    m = R3.match(res)
    if m:
        a, p1, b, p2, c = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4), int(m.group(5))
        return "%d%s%s%s%s+0+0%s" % (a, p1, sgn(b), p2, sgn(c), sgn(-(a + b + c)))
    return res


def norm_name(name):
    name = name.strip()
    if name in NAME_MAP:
        return NAME_MAP[name]
    m = re.match(r"^(\d+|\d{4} [A-Z]{2}\d*) \(([^)]+)\)$", name)
    if m:
        return m.group(1)
    m = re.match(r"^(\d+|\d{4} [A-Z]{2}\d*) / (\d+|\d{4} [A-Z]{2}\d*)( \(pair \d+\))?$", name)
    if m:
        return m.group(1)
    return name


def is_group(name):
    if name in NAME_MAP or name.startswith("model:"):
        return False
    return any(h in name for h in GROUP_HINTS) or (" / " in name and "(" in name)


def rtype_of(res, given):
    if given in ("model", "other"):
        return given
    if res == "LK":
        return "zlk"
    if re.search(r"(^|[^A-Za-z])[gs][0-9]|^[-+]?\d*\(?[gs]", res) or res.startswith(
        ("g", "s", "-g", "2(", "2g", "2s", "all secular", "nu")
    ):
        return "secular"
    letters = re.findall(r"[A-Z]", res)
    if re.match(r"^1[A-Z]-1\+0\+0$", res):
        return "coorbital"
    if len(letters) == 1:
        return "mmr2"
    if len(letters) == 2:
        return "mmr3"
    return given or "other"


def fix_reliability(it):
    rel = it.get("reliability", "unknown")
    src = it.get("source", "")
    if "gt:A" in src or "author:2026-07-04" in src:
        return "author"
    if "gt:V" in src or "verdict:" in src:
        return "adjudicated"
    return rel


def resolve_path(p):
    p = (p or "").strip()
    if not p:
        return ""
    p = p.split(";")[0].strip()
    p = re.sub(r"\s*\(.*\)$", "", p)
    if p.startswith("B/"):
        p = os.path.join(ROOT, "experiments/nea-mmrs/research/2026-09-classifier", p[2:])
    elif p.startswith(("experiments/", "output/", "tests/")):
        p = os.path.join(ROOT, p)
    elif p.startswith("combined-review-en-png/"):
        p = os.path.join(ROOT, "experiments/nea-mmrs/research/2026-09-classifier", p)
    return p


def path_rank(p):
    if not p:
        return -1
    exists = os.path.exists(p)
    score = 0 if not exists else 10
    if p.endswith((".csv", ".csv.gz")):
        score += 5
    if "long-200k" in p:
        score += 4
    if "coorbital-1000-forward" in p or "fixtures/free-omega" in p:
        score += 3
    if "production-T1T2" in p:
        score += 2
    if "secular-reference" in p or "secular-10myr-v4/data" in p:
        score += 2
    if p.endswith(".png") or p.endswith("/") or os.path.isdir(p):
        score -= 6
    return score


HEADER_CACHE = {}


def header(p):
    if p not in HEADER_CACHE:
        try:
            if p.endswith(".gz"):
                import gzip

                with gzip.open(p, "rt") as f:
                    HEADER_CACHE[p] = f.readline().strip().split(",")
            else:
                with open(p) as f:
                    HEADER_CACHE[p] = f.readline().strip().split(",")
        except Exception:
            HEADER_CACHE[p] = []
    return HEADER_CACHE[p]


def has_angle(p, res, rtype):
    if not p or not p.endswith((".csv", ".csv.gz")) or not os.path.exists(p):
        return None
    cols = header(p)
    if not cols:
        return None
    if res and res in os.path.basename(p) and "angle" in cols:
        return True
    if res in ("all secular angles", "z1/z2 angles"):
        return any(c.endswith("_angle") for c in cols)
    if rtype == "zlk":
        return any(c.startswith("LK_angle") for c in cols) or "omega" in cols
    if rtype == "secular":
        return any(c.startswith(res + "_angle") for c in cols)
    return any(c.startswith(res + "_angle") for c in cols)


def data_desc(p):
    if not p:
        return ""
    rules = [
        (
            "production-T1T2+rubin-20000-forward",
            "NEA production: 20 kyr fwd, ias15, step 5 yr (4001 rows); cols a,e,inc,Omega,omega,M,longitude,varpi,a_filtered,times,<res>_angle[_unwrapped|_filtered...],LK_angle*; planets not saved",
        ),
        ("2026-09-classifier/input", "atlas copy of the NEA production series: 20 kyr ias15, 5 yr step"),
        ("combined-review", "atlas plot_data.npz (+cycles.csv, metadata.json), derived from the 20 kyr NEA series"),
        ("alias_1865_fine", "fine-cadence rerun of 1865 (first ~200 yr) proving the alias"),
        (
            "coorbital-1000-forward",
            "co-orbital run: 1 kyr fwd, ias15 + Moon, step 0.05 yr (20001 rows); cols as package data CSV with 1X-1+0+0_angle*",
        ),
        ("cadence_test/runs", "co-orbital cadence test runs (1 kyr / 20 kyr, steps 0.02-5 yr)"),
        ("run-50000_2026-09-30", "asteroid pairs: 50 kyr fwd, ias15, step 0.6 yr (83334 rows); planets saved (data-planet-*.csv)"),
        ("long-200k", "asteroid pairs rerun: 200 kyr fwd, ias15, step 0.6 yr"),
        ("partners-50k", "asteroid-pair partners: 50 kyr ias15 step 0.6 yr"),
        (
            "secular-10myr-v4/data",
            "secular v4: 10 Myr SABA dt 0.5, step 100 yr (100001 rows), ~66-150 angles/body with _osculating/_proper cols; planets NOT saved",
        ),
        ("secular-10myr-saba-v4", "secular v4 SABA dir (same data as secular-10myr-v4)"),
        ("secular-10myr-ias15-pair12", "secular v4 ias15 rerun of chaotic pair 12, 10 Myr"),
        (
            "secular-reference-10myr",
            "secular reference run: 10 Myr SABA dt 0.5, step 100 yr, 9 angles; summary.csv statuses are OLD (osculating) - reclassify",
        ),
        ("test-v5-20myr", "secular test v5: 20 Myr, step 100 yr, measured planet phases, planets saved"),
        (
            "fixtures/free-omega",
            "test fixture: first 500 kyr of mercurius run thinned to 100 yr (5001 rows); cols times,a,e,inc,Omega,omega,LK_angle_unwrapped",
        ),
        ("lkr-ias15", "LK scan: 100 kyr ias15 dt 1, AstDyS 2025-11-21, ~16 yr step (6283 rows), pre-gate statuses"),
        ("omega-free.csv", "free-omega series from the NEA 20 kyr run (times, |z_e|,|z_i| free/forced, free angles)"),
        ("batch_mmr_test", "secular 2nu6-nu5 test: 100 kyr SABA dt 1, cols angle/_osculating/_proper"),
        ("pilot10", "NEA pilot (20/100 kyr) - mostly PNGs, series not located"),
        ("resonances-main-check", "old main-code check run (100 kyr SABA), PNG/periodograms only"),
        ("planetary-frequencies", "planet-only NAFF runs (npz)"),
    ]
    for key, d in rules:
        if key in p:
            return d
    return ""


def code_to_bucket(code):
    return {
        "2": "resonant",
        "-2": "kinematic-libration",
        "1": "transient",
        "-1": "transient",
        "-3": "near-separatrix",
        "-4": "slow-circulation",
        "-5": "non-resonant",
        "0": "non-resonant",
        "-6": "insufficient-time",
        "-9": "uncertain",
        "-99": "chaotic",
    }.get(code)


def first_status(s):
    m = re.search(r"-?\d+", s or "")
    return m.group(0) if m else ""


def has(t, *ws):
    return any(w in t for w in ws)


def bucket_from_text(t):
    """Map an author verdict text to (bucket, confidence). None if nothing recognisable.

    Rules are tried in order; the first that fires wins. Phrases are the author's own wording
    in the sweeps (English paraphrases plus a few Russian quotes)."""
    t = t.lower()
    if "->" in t:  # a later, longer run overrides an earlier verdict
        t = t.split("->")[-1]
    if t.startswith("transient"):
        return "transient", "medium" if has(t, "but", "almost") else "high"
    if t.strip() in ("resonant", "non-resonant"):
        return t.strip(), "high"
    if has(t, "after 200 kyr", "at 200 kyr") and has(t, "non-resonant"):
        return "non-resonant", "high"
    if has(t, "kinematic libration", "-2 (kinematic"):
        return "kinematic-libration", "high"
    if t.startswith("controversial"):
        if has(t, "transient for me", "closer to transient", "several captures", "short capture", "one capture", "very short capture"):
            return "transient", "low"
        if has(t, "non-resonant for me", "closer to non-resonant", "inclined to non-resonant", "or non-resonant"):
            return "non-resonant", "low"
        return "uncertain", "low"
    if has(t, "resonant (maybe", "resonant after"):
        return "resonant", "medium"
    if has(
        t,
        "extend time",
        "extend integration",
        "insufficient baseline",
        "needs longer",
        "longer integration",
        "lack of integration time",
        "integration time insufficient",
        "не хватает времени",
        "надо на 30 млн",
        "longer baseline",
        "needs extended integration",
        "need extending",
        "baseline_too_short",
        "-6",
    ):
        return "insufficient-time", "low" if "slow circulation" in t else "medium"
    if has(t, "resonant or transient", "pure or transient"):
        return "uncertain", "low"
    if has(
        t,
        "no actual librations",
        "non-resonant",
        "nonresonant",
        "not resonant",
        "not zlk",
        "non-librator",
        "not co-orbital",
        "not a co-orbital",
        "not a qs",
        "plainly 0",
        "obvious 0",
        "definite circulation",
        "closer to 0",
        "0, at worst",
        "захвата нет",
        "not libration",
        "control, 0",
    ):
        if has(t, "likely", "rather", "closer to", "seems", "inclined"):
            return "non-resonant", "medium"
        if has(t, "unclear", "do not know", "dunno"):
            return "non-resonant", "low"
        return "non-resonant", "high"
    m = re.search(
        r"(closer to|likely|most likely(?: still)?|would say|clearer that it is|obviously|undoubtedly|"
        r"clearly|despite[^;]*;)?\s*\bresonant\b",
        t,
    )
    if m or has(t, "должна же быть 2", "явный резонанс", "+2!"):
        strong = has(t, "obviously", "undoubtedly", "clearly resonant", "clearer that it is", "+2!", "явный")
        weak = has(t, "likely", "closer to", "would say", "unclear", "needs check", "nuances", "drift", "weak", "disputed", "despite")
        if strong:
            return "resonant", "high"
        return "resonant", "medium" if weak else "high"
    if has(t, "slow circulation"):
        if has(t, "close to", "closer to", " or ", "likely", "would bet", "seems", "nuances"):
            return "slow-circulation", "medium"
        return "slow-circulation", "high"
    if has(t, "near separatrix", "near-separatrix", "separatrix"):
        return "near-separatrix", "low" if "?" in t else "medium"
    if re.search(r"(^|[^0-9])-3\b", t):
        if "clearly" in t:
            return "near-separatrix", "high"
        if has(t, "between", "arguably", "or 1", "theoretically 1", "very much"):
            return "near-separatrix", "medium"
        return "near-separatrix", "high"
    if has(t, "staircase"):
        return "near-separatrix", "low"
    if has(
        t,
        "clean circulation",
        "circulates cleanly",
        "uniform slow circulation",
        "clean turns",
        "structureless",
        "uniform everywhere",
        "uniform in all",
        "uniform angle",
    ):
        return "non-resonant", "high"
    if has(t, "transient", "capture", "stickiness", "escape"):
        if has(t, "unclear", "do not know", "no idea"):
            return "uncertain", "low"
        if has(
            t,
            "excellent",
            "model transient",
            "purest",
            "perfect",
            "textbook",
            "clear transient",
            "classical capture",
            "clearly a transient",
            "true status 1",
        ):
            return "transient", "high"
        if has(t, "closer to 1", "more like"):
            return "transient", "medium"
        if has(t, "likely", "rather", "disputed", "maybe", "could be", "possible"):
            return "transient", "low"
        return "transient", "medium"
    if has(t, "circulation"):
        return "non-resonant", "medium"
    if has(t, "do not know", "no idea", "dunno", "unclear", "unknown", "how to interpret", "strange", "disputed"):
        return "uncertain", "low"
    if has(t, "librator", "libration", "in resonance", "status 2", "либраций много", "e-i exchange confirmed"):
        if has(t, "likely", "drift", "weak", "emerging", "perturbed"):
            return "resonant", "medium"
        return "resonant", "high"
    if t.strip() == "2":
        return "resonant", "medium"
    return None


# Hand overrides for objects whose verdict text does not parse cleanly or that carry known caveats.
OVERRIDES = {
    ("591986", "LK"): ("kinematic-libration", "high", "free-omega gate demotes to -2; reference false-positive guard"),
    ("162474", "LK"): ("resonant", "high", "genuine ZLK, confirmed by gate; date/step sensitive"),
    ("4257", "LK"): ("resonant", "high", "gate-confirmed reference"),
    ("3040", "LK"): ("resonant", "high", "gate-confirmed reference"),
    ("3752", "LK"): ("resonant", "low", "confirmed at 500 kyr but representative choice moves it to -9; chaotic Earth-crosser"),
    ("88899", "LK"): ("uncertain", "medium", "300 kyr: 2 -> gate -9; 500 kyr: not 2"),
    ("10563", "LK"): ("non-resonant", "low", "author false-positive list; SABA ejects"),
    ("373136", "LK"): ("non-resonant", "medium", "does not reach 2 at 500 kyr"),
    ("797204", "LK"): ("non-resonant", "low", "does not reach 2 at 500 kyr; irregular ZLK-like"),
    ("555527", "LK"): ("non-resonant", "low", "near-planar, does not reach 2 at 500 kyr"),
    ("1373", "LK"): ("resonant", "low", "Nov 2025 label, c2>0: doubtful"),
    ("15527", "LK"): ("non-resonant", "medium", "slow-test control; Nov 2025 labels conflict"),
    ("622577", "LK"): ("uncertain", "low", "Nov 2025 'should be trapped', never validated"),
    ("1981", "LK"): ("insufficient-time", "low", "NEA 20 kyr: -9 short_window; Nov 2025 candidate"),
    ("1866", "LK"): ("insufficient-time", "medium", "gate -9 short_window at 20 kyr"),
    ("463", "LK"): ("non-resonant", "high", "omega bounded by geometry; negative control"),
    ("624", "LK"): ("non-resonant", "high", "negative control"),
    ("592056", "LK"): ("non-resonant", "high", "coherent circulation"),
    ("651871", "LK"): ("non-resonant", "high", "coherent circulation"),
    ("15544", "LK"): ("near-separatrix", "medium", "legacy omega angle; staircase at e minima"),
    ("88710", "LK"): ("non-resonant", "high", ""),
    ("189865", "LK"): ("chaotic", "high", "e>1.3"),
    ("477492", "LK"): ("chaotic", "high", "e>1.3"),
    ("78201", "LK"): ("resonant", "medium", "ZLKR antiphase validated"),
    ("1020", "g-g6+s-s6"): ("resonant", "high", "author chose 3 turning points on this case"),
    ("363", "g-g6+s-s6"): ("resonant", "medium", "author: 'должна же быть 2'; current rule gives -6; decision pending"),
    ("3701", "g-g6+s-s6"): ("insufficient-time", "medium", "8 Myr period, 2 turning points at 10 Myr"),
    ("847", "g-g6+s-s6"): ("slow-circulation", "high", "just outside z1"),
    ("1222", "g-g6"): ("resonant", "high", "Tina, must stay 2"),
    ("4305", "g-3g6+2g5"): ("resonant", "high", "author: 'явный резонанс'"),
    ("2110", "g-g6-2g7+2g8"): ("insufficient-time", "high", "author: bounded but not enough time -> -6"),
    ("2005 GQ107", "g-2g2+g8"): ("insufficient-time", "high", "author: needs 30 Myr, should be -6"),
    ("2005 GQ107", "s-g5+g8-s6"): ("insufficient-time", "medium", "same dynamics as g-2g2+g8"),
    ("2008 UZ220", "g-g1+s3-s8"): ("insufficient-time", "low", "author asked how to read it; assistant -6"),
    ("19289", "g-g6"): ("kinematic-libration", "low", "Tina vs kinematic unresolved; author never ruled"),
    ("278067", "g-g6"): ("kinematic-libration", "low", "partner of 19289"),
    ("26420", "g-g6"): ("kinematic-libration", "medium", "free g-g6 4.1 arcsec/yr, period = 2pi/(g-g6)"),
    ("80218", "g-g6"): ("kinematic-libration", "medium", "free g-g6 5.3 arcsec/yr"),
    ("127504", "s-s6"): ("transient", "medium", "intermittent 3.6-7.8 Myr"),
    ("9068", "s-2s3+s4"): ("chaotic", "medium", "only chaotic pair; package chaos check misses it"),
    ("481085", "7E-10M-6+0+0+9"): ("uncertain", "low", "author: controversial; kinematic-looking (sigma period = e period)"),
    ("285637", "3M+7J-6+0+0-4"): ("uncertain", "low", "author: controversial; kinematic-looking"),
    ("7343", "1M+1J-2+0+0+0"): ("resonant", "high", "confirmed at 200 kyr (7 cycles)"),
    ("154634", "1M+1J-2+0+0+0"): ("resonant", "high", "confirmed at 200 kyr (5 cycles)"),
    ("501710", "1E+7S-4+0+0-4"): ("insufficient-time", "medium", "author: extend time; Claude 200 kyr: slow circulation (unconfirmed)"),
    ("503955", "1V-6M+6+0+0-1"): ("slow-circulation", "high", "author final: slow circulation at 200 kyr"),
    ("26416", "1M-3J-1+0+0+3"): ("non-resonant", "medium", "200 kyr: ~5 slow turns; author non-resonant"),
    ("214954", "1M-3J-1+0+0+3"): ("non-resonant", "medium", "200 kyr: ~5 slow turns"),
    ("42355", "3N-4+0+1"): ("transient", "medium", "slow test expects |1| (stickiness); date-dependent, currently gets 0"),
    ("2026 SZ2", "1V-1+0+0"): ("uncertain", "low", "WRONG INPUT elements - exclude from truth"),
    ("2010 CG236", "1E-1+0+0"): ("resonant", "low", "orbit undetermined (U9); label is a property of the fit"),
    ("2010 NT134", "1E-1+0+0"): ("resonant", "low", "orbit undetermined; author: drift"),
    ("2010 DH65", "1E-1+0+0"): ("uncertain", "low", "orbit undetermined"),
    ("88710", "9V-16+0+7"): ("transient", "high", "capture 0-22 kyr then escape (100 kyr)"),
    ("6489", "3J-1+0-2"): ("transient", "low", "test case disabled"),
    ("1863", "7J-2+0-5"): ("transient", "medium", "author: clear transient; theory: near-separatrix intermittency (class B)"),
    ("2013 TW135", "8M-4S-7+0+0+3"): ("slow-circulation", "low", "author: drift = slow circulation, asks if near separatrix"),
    ("2013 VA14", "2J+6S-1+0+0-7"): ("uncertain", "low", "author: staircase, slow circulation or separatrix - do not know"),
    ("719", "2M+3S-5+0+0+0"): ("uncertain", "low", "author unsure; theory: stall"),
    ("2003 TX9", "3E+2J-10+0+0+5"): ("uncertain", "low", "author: resonant or transient"),
    ("1997 GK3", "3M+5S-3+0+0-5"): ("uncertain", "low", "author: libration with trend, pure or transient unclear"),
    ("2011 ET4", "4M+8S-5+0+0-7"): ("slow-circulation", "medium", ""),
    ("2012 DE61", "1E+5S-4+0+0-2"): ("slow-circulation", "medium", ""),
    ("2014 QH365", "2M+2S-3+0+0-1"): ("slow-circulation", "low", "like 2013 TW135"),
    ("2017 UA8", "1R-10M+2+0+0+7"): ("slow-circulation", "low", "like 2013 TW135"),
    ("2018 TX1", "3M-5S-2+0+0+4"): ("slow-circulation", "low", "like 2013 TW135"),
    ("1685", "1V-1E-1+0+0+1"): ("uncertain", "low", "author: drifting loops, do not know"),
    ("1917", "6M-4J-9+0+0+7"): ("uncertain", "low", "author: like 1685 1V-1E-1"),
    ("1998 QA62", "1V-5M+3+0+0+1"): ("uncertain", "low", "author: interesting, short oscillation matching a"),
    ("1685", "9M+6S-8+0+0-7"): ("uncertain", "low", "author: do not know"),
    ("1992 SZ", "5M-5S-8+0+0+8"): ("uncertain", "low", "author: drift with libration at start"),
    ("2000 YG4", "2R-8E-1+0+0+7"): ("uncertain", "low", "author: do not know"),
    ("2001 PG14", "33S-2+0-31"): ("uncertain", "low", "author: tricky two-scale case"),
    ("1221", "1V-2E+1+0+0+0"): ("uncertain", "low", "author: do not know"),
    ("1566", "1R-7J-4+0+0+10"): ("uncertain", "low", "author: no idea"),
    ("1862", "5J+4S-1+0+0-8"): ("uncertain", "low", "author: unknown"),
    ("1864", "51M-50+0-1"): ("non-resonant", "low", "alias (class X); author: no idea, maybe transient"),
    ("1865", "10E-11+0+1"): ("non-resonant", "medium", "alias confirmed by fine-cadence rerun"),
    ("1865", "21E-22+0+1"): ("non-resonant", "low", "alias (class X); author unsure"),
    ("1865", "5E+4M-8+0+0-1"): ("uncertain", "low", "author: tiny late capture, do not know"),
    ("1999 UQ", "7E+2M-9+0+0+0"): ("non-resonant", "low", "alias (class X); author thought insufficient baseline"),
    ("2019 UK", "7E+1S-6+0+0-2"): ("non-resonant", "low", "alias (class X); author: slow circulation"),
    ("1620", "3V+2S-7+0+0+2"): ("non-resonant", "medium", "alias; author likely nonresonant"),
    ("1620", "7E+1S-10+0+0+2"): ("non-resonant", "medium", "alias; author nonresonant"),
    ("2004 TK14", "31M-34+0+3"): ("uncertain", "low", "author: irregular libration, do not know"),
    ("2003 ND", "6J-6S-1+0+0+1"): ("uncertain", "low", "author: very unclear"),
    ("2001 HW7", "5J+6S-2+0+0-9"): ("uncertain", "low", "author: do not know how to classify"),
    ("2003 EH16", "1M-3J-1+0+0+3"): ("slow-circulation", "medium", "author: closer to slow circulation"),
    ("1980", "1M-1J-1+0+0+1"): ("slow-circulation", "high", "author: clearly slow circulation"),
    ("2003 TG2", "4E-1M-3+0+0+0"): ("insufficient-time", "low", "author: slow circulation then maybe libration, extend"),
    ("2003 KQ18", "6J+2S-1+0+0-7"): ("resonant", "low", "author: resonant, needs checking; one V arc"),
    ("1627", "3V-4M-7+0+0+8"): ("transient", "low", "author: disputed transient; theory: stall"),
    ("1627", "11E-28+0+17"): ("transient", "medium", "author: excellent transient; theory: stall (class D)"),
    ("1627", "13E-33+0+20"): ("transient", "medium", "author: excellent transient; theory: stall (class D)"),
    ("433", "1R-5V+7+0+0-3"): ("transient", "low", "author: transient 4-12 kyr; theory: common a event"),
    ("433", "47M-44+0-3"): ("transient", "low", "author: transient; theory: stall"),
    ("1862", "12V-35+0+23"): ("transient", "low", "author: rather transient"),
    ("1999 TJ10", "19M-22+0+3"): ("transient", "medium", "author: double transient"),
    ("24293", "19M-35+0+16"): ("transient", "medium", "closer to 1, theoretically -3"),
    ("16017", "7S-1+0-6"): ("uncertain", "low", "author: between -3 and 1 (primary 1)"),
    ("887", "3J-1+0-2"): (
        "transient",
        "high",
        "author: model transient in the NEA 20 kyr ias15 run; old benchmark (100 kyr SABA) labels it -99 chaotic",
    ),
    ("2002 XA", "5J-2+0-3"): ("resonant", "medium", "author: like 1995 HM, librates with a's period"),
    ("1999 UQ", "8M-9J-4+0+0+5"): ("resonant", "low", "author: like 1995 HM-type, possible apocentric arc; theory class E"),
    ("1980", "5J-1+0-4"): ("non-resonant", "high", "atlas control, ~517 turns"),
    ("1995 HM", "6J-1+0-5"): ("non-resonant", "high", "atlas control, ~1227 turns"),
    ("2000 SF8", "5J-1+0-4"): ("non-resonant", "high", "atlas control, ~1017 turns"),
    ("84203", "3M+7J-6+0+0-4"): ("non-resonant", "medium", "clean circulation; contrast partner of kinematic 285637"),
    ("92336", "1R-3M-7+0+0+9"): ("non-resonant", "high", "author non-resonant; kinematic-looking libration with status 2"),
    ("19289 / 278067 other angles", "g-g6+s-s6"): ("uncertain", "low", "pre-fix osculating artifact - do not use as label"),
    ("26420 / 2012 TS209 old run", "g-g6"): ("uncertain", "low", "pre-fix osculating artifact - superseded"),
    ("lkr-ias15 low-inclination status-2 set: 80923, 525487, 544232, 320511, 460870, 550108, 694503, 207978", "LK"): (
        "kinematic-libration",
        "low",
        "forced-geometry suspects; expect -2/-9 at 500 kyr",
    ),
    (
        "lkr-ias15 status-2 high-e set: 858130, 385402, 143637, 523630, 504033, 504034, 184990, 267494, 658883, 153957, 307005, 530871",
        "LK",
    ): ("uncertain", "low", "unvetted pre-gate 100 kyr status 2"),
    ("NEA ZLK status-2 set: 2010 HY22, 2020 PN1, 2020 PP1, 2023 FW13, 2023 RS2", "LK"): (
        "uncertain",
        "low",
        "20 kyr confirmation is outside the gate calibration",
    ),
    (
        "18916 (and 24 other lkr chaotic: 119039 139359 237838 248590 312978 405058 434707 405562 461912 490171 513776 529456 529465 551023 584530 612937 670295 725997 761729 829876 850809 850547 841810 848043)",
        "LK",
    ): ("chaotic", "medium", "chaos short-circuit negative controls"),
    ("39984", "37S-5+0-32"): ("uncertain", "low", "author: disputed (primary -3)"),
}


def main():
    raw = {}
    for f in glob.glob(os.path.join(HERE, "raw", "*.json")):
        d = json.load(open(f))
        raw[d["key"]] = d

    # ground truth file (authoritative for the benchmark rows)
    gt = {}
    with open(os.path.join(ROOT, "benchmark/raw/old_ground_truth.csv")) as f:
        for r in csv.DictReader(f):
            gt[(r["name"], r["resonance"])] = r

    rows = OrderedDict()
    for key in SOURCE_ORDER:
        for it in raw[key]["items"]:
            name0 = it["name"].strip()
            if name0 in PLACEHOLDERS:
                continue
            group = is_group(name0)
            name = name0 if group else norm_name(name0)
            res = norm_resonance(name, it.get("resonance", ""))
            rtype = "group:" + rtype_of(res, it.get("rtype", "")) if group else rtype_of(res, it.get("rtype", ""))
            rel = fix_reliability(it)
            k = (name, res)
            r = rows.setdefault(
                k,
                {
                    "name": name,
                    "resonance": res,
                    "rtype": rtype,
                    "statuses": [],
                    "verdicts": [],
                    "other_verdicts": [],
                    "rels": [],
                    "sources": [],
                    "tags": [],
                    "paths": [],
                    "notes": [],
                    "src_keys": [],
                },
            )
            if r["rtype"].startswith("other") and not rtype.startswith("other"):
                r["rtype"] = rtype
            st = (it.get("old_status") or "").strip()
            if st:
                r["statuses"].append("%s [%s]" % (st, key))
            v = (it.get("verdict") or "").strip()
            if v:
                (r["verdicts"] if rel == "author" else r["other_verdicts"]).append((key, v))
            r["rels"].append(rel)
            r["sources"].append("%s: %s" % (key, it.get("source", "")))
            r["src_keys"].append(key)
            for t in it.get("tags", []):
                if t and t not in r["tags"]:
                    r["tags"].append(t)
            p = resolve_path(it.get("data_path", ""))
            if p:
                r["paths"].append(p)
            n = (it.get("notes") or "").strip()
            if n and n not in r["notes"]:
                r["notes"].append("[%s] %s" % (key, n))

    out = []
    for (name, res), r in rows.items():
        best_rel = max(r["rels"], key=lambda x: REL_RANK.get(x, 0))
        g = gt.get((name, res))
        if g and g["source"].startswith("author"):
            best_rel = "author"
        # author verdict text, deduplicated, in source order
        seen, av = set(), []
        for _, v in r["verdicts"]:
            if v.lower() not in seen:
                seen.add(v.lower())
                av.append(v)
        author_verdict = " | ".join(av)
        # data path: best ranked
        rt0 = r["rtype"].replace("group:", "")
        paths = sorted(set(r["paths"]), key=lambda p: (has_angle(p, res, rt0) is True, path_rank(p)), reverse=True)
        dp = paths[0] if paths else ""
        angle_ok = has_angle(dp, res, rt0)
        status_codes = [first_status(s.split(" [")[0]) for s in r["statuses"]]
        status_codes = [c for c in status_codes if c]

        bucket, conf, why = None, None, ""
        if (name, res) in OVERRIDES:
            bucket, conf, why = OVERRIDES[(name, res)]
        if bucket is None and g and g["primary"] and (g["source"].startswith("author") or g["source"].startswith("verdict")):
            bucket = code_to_bucket(g["primary"])
            acc = [a for a in g["acceptable"].split(";") if a]
            famap = {"2": "R", "-2": "R", "1": "T", "-1": "T", "-3": "N", "-4": "S", "0": "Z", "-5": "Z", "-99": "C"}
            fam = {famap.get(a, a) for a in acc}
            if g["source"].startswith("author"):
                conf = "high" if len(fam) <= 1 else ("medium" if len(fam) == 2 else "low")
                tb = bucket_from_text(g["comment"]) if g["comment"] else None
                if tb and tb[1] == "high" and tb[0] == bucket:
                    conf = "high"
            else:
                conf = "medium" if "high" in g["source"] and len(fam) <= 2 else "low"
            why = "ground_truth primary %s (acceptable %s, %s)" % (g["primary"], g["acceptable"], g["source"])
        if bucket is None and author_verdict:
            # prefer the first author verdict that parses
            for v in av:
                tb = bucket_from_text(v)
                if tb:
                    bucket, conf = tb
                    why = "author verdict"
                    break
        if bucket is None and g and g["primary"]:
            bucket = code_to_bucket(g["primary"])
            conf = "medium" if best_rel == "author" else "low"
            if re.match(r"^1J-1\+0\+0$", res) and g["primary"] == "2":
                conf = "high"
            why = "legacy dataset label %s" % g["primary"] + (" + author-asserted test" if best_rel == "author" else "")
        if bucket is None:
            for k, v in r["other_verdicts"]:
                tb = bucket_from_text(v)
                if tb:
                    bucket, conf = tb[0], "low" if best_rel not in ("adjudicated", "literature") else "medium"
                    if conf == "medium" and tb[1] == "low":
                        conf = "low"
                    why = "non-author verdict (%s)" % k
                    break
        if bucket is None and status_codes:
            bucket = code_to_bucket(status_codes[0]) or "uncertain"
            conf = "high" if best_rel == "model" else ("medium" if best_rel == "author" else "low")
            if best_rel == "author":
                why = "author-asserted test status"
            why = "status %s" % status_codes[0]
        if bucket is None:
            bucket, conf, why = "uncertain", "low", "no label"
        # poor orbits and wrong inputs cannot carry high confidence
        if "poor-orbit" in r["tags"] and conf == "high" and r["rtype"] == "coorbital":
            conf = "medium"

        # integration need
        rt = r["rtype"].replace("group:", "")
        if rt == "model":
            need = "no"
        elif not dp or not os.path.exists(dp):
            need = "yes"
        elif dp.endswith(".png") or os.path.isdir(dp):
            need = "yes"
        elif angle_ok is False:
            need = "yes"
        else:
            need = "no"
            if bucket == "insufficient-time" or ("insufficient-baseline" in r["tags"] and rt in ("coorbital", "secular")):
                need = "partial"
            if "lkr-ias15" in dp:
                need = "partial"
            if "coorbital-1000-forward" in dp and res.startswith("1M"):
                need = "partial"
        note_bits = []
        if why:
            note_bits.append("bucket from " + why)
        if len(paths) > 1:
            note_bits.append("alt data: " + "; ".join(paths[1:3]))
        if angle_ok is False:
            note_bits.append("angle column missing in data file")
        if r["other_verdicts"]:
            note_bits.append("other verdicts: " + " | ".join("%s: %s" % kv for kv in r["other_verdicts"][:3]))
        note_bits += r["notes"][:4]
        out.append(
            {
                "name": name,
                "resonance": res,
                "rtype": r["rtype"],
                "old_statuses": "; ".join(OrderedDict.fromkeys(r["statuses"])),
                "author_verdict": author_verdict,
                "best_reliability": best_rel,
                "sources": "; ".join(OrderedDict.fromkeys(r["src_keys"])),
                "tags": "|".join(r["tags"]),
                "data_path": dp,
                "data_desc": data_desc(dp),
                "needs_integration": need,
                "proposed_bucket": bucket,
                "confidence": conf,
                "notes": " || ".join(note_bits)[:1500],
                "_first_status": status_codes[0] if status_codes else "",
            }
        )

    # thin near-identical algorithmic/literature rows with no author verdict
    kept, dropped = [], []
    grp = defaultdict(list)
    for row in out:
        if (
            row["best_reliability"] in ("algorithmic", "literature", "unknown")
            and not row["author_verdict"]
            and not row["rtype"].startswith("group")
        ):
            grp[(row["rtype"], row["resonance"], row["_first_status"], row["proposed_bucket"])].append(row)
        else:
            kept.append(row)
    for gk, lst in grp.items():
        if len(lst) <= KEEP:
            kept += lst
            continue
        lst.sort(key=lambda x: (x["needs_integration"] != "no", x["name"]))
        keep, drop = lst[:KEEP], lst[KEEP:]
        names = ", ".join(x["name"] for x in drop)
        for x in keep:
            x["notes"] = ("[thinned: %d similar rows dropped (%s)] " % (len(drop), names[:300])) + x["notes"]
        kept += keep
        for x in drop:
            x["dropped_group"] = "|".join(gk)
            dropped.append(x)

    order = {"mmr2": 0, "mmr3": 1, "coorbital": 2, "secular": 3, "zlk": 4, "model": 5, "other": 6}
    kept.sort(
        key=lambda x: (
            x["rtype"].startswith("group"),
            order.get(x["rtype"].replace("group:", ""), 9),
            x["proposed_bucket"],
            x["name"],
            x["resonance"],
        )
    )
    cols = [
        "id",
        "name",
        "resonance",
        "rtype",
        "old_statuses",
        "author_verdict",
        "best_reliability",
        "sources",
        "tags",
        "data_path",
        "data_desc",
        "needs_integration",
        "proposed_bucket",
        "confidence",
        "notes",
    ]
    with open(os.path.join(HERE, "candidates.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(kept, 1):
            row["id"] = "c%04d" % i
            w.writerow(row)
    with open(os.path.join(HERE, "dropped.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dropped_group"] + cols[1:], extrasaction="ignore")
        w.writeheader()
        for row in dropped:
            w.writerow(row)
    print("rows", len(out), "kept", len(kept), "dropped", len(dropped))


if __name__ == "__main__":
    main()
