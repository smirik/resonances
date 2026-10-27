"""DATA_SOURCE=mpc on the package's side of `resonances.mpcorb`: its settings from the MPC_* keys of .env, and the
epoch policy (MPC_EPOCH_POLICY) with its refusal message. The catalogue itself (download, cache, provenance, search)
is `resonances.mpcorb`, which imports nothing from the package.

Importing this module makes the MPC_* keys the defaults of `resonances.mpcorb` (read at every call), so
`resonances.mpcorb.query(...)` uses the catalogue a run would use.
"""

import datetime
from typing import Optional

from resonances import mpcorb
from resonances.config import config as c

EPOCH_POLICIES = ('strict', 'propagate')
UPDATE_HINT = '`resonances mpc-update` (or resonances.mpcorb.update_catalog())'


def _setting(key: str, default: str) -> str:
    value = c.get(key) if c.has(key) else None
    return default if value in (None, '') else str(value)


def mpc_settings(catalog: Optional[str] = None) -> mpcorb.Settings:
    """The catalogue's settings from MPC_CATALOG (or `catalog`), MPC_CACHE_DIR, MPC_SHA256, MPC_MAX_AGE_DAYS and
    MPC_WARN_U."""
    return mpcorb.Settings(
        catalog=catalog or _setting('MPC_CATALOG', 'mpcorb'),
        cache_dir=_setting('MPC_CACHE_DIR', 'cache/mpc'),
        sha256=_setting('MPC_SHA256', ''),
        max_age_days=float(_setting('MPC_MAX_AGE_DAYS', '30')),
        warn_u=int(_setting('MPC_WARN_U', '6')),
        update_hint=UPDATE_HINT,
        user_agent='resonances (python)',
    )


mpcorb.configure(mpc_settings)


def epoch_refusal(
    row: dict,
    date: datetime.datetime,
    date_mjd: float,
    standard_epoch_mjd: Optional[float] = None,
    policy: str = 'strict',
    max_days: Optional[float] = None,
) -> str:
    """The message of an MPC row whose epoch is not the simulation date: refused by MPC_EPOCH_POLICY=strict, or by
    propagate when the span exceeds `max_days` (MPC_PROPAGATE_MAX_DAYS). It offers only the ways out that apply: an
    unset date when the body is at the catalogue's standard epoch (else a date at the body's epoch), and the
    propagate policy when it is not the one in force."""
    epoch = float(row['epoch_mjd'])
    why = (
        " It is a one-opposition orbit with an unperturbed solution (the last section of MPCORB, each at its own epoch)."
        if not row['perturbers']
        else ''
    )
    if policy == 'propagate' and max_days is not None:
        why += f" The span, {abs(epoch - date_mjd):.1f} days, exceeds MPC_PROPAGATE_MAX_DAYS = {max_days:g}."
    if standard_epoch_mjd is not None and abs(epoch - standard_epoch_mjd) < 1e-9:
        ways = [f"leave date unset (the catalogue's standard epoch, {mpcorb.datetime_from_mjd(epoch):%Y-%m-%d}, is the body's epoch)"]
    else:
        ways = ["set date to the body's epoch"]
    if policy != 'propagate':
        ways.append("set MPC_EPOCH_POLICY=propagate to integrate it to the date in the package's model")
    else:
        ways.append("raise MPC_PROPAGATE_MAX_DAYS (the propagation was checked over 138 days only)")
    advice = ', '.join(ways[:-1]) + ' or ' + ways[-1] if len(ways) > 1 else ways[0]
    return (
        f"The MPC elements of {mpcorb.row_label(row)} are at epoch MJD {epoch:.5f} TT "
        f"({mpcorb.datetime_from_mjd(epoch):%Y-%m-%d %H:%M}), but the simulation starts at {date} (MJD {date_mjd:.5f}).{why} "
        f"{advice[0].upper()}{advice[1:]}."
    )
