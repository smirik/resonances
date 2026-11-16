# mmr and matrix — MMR classes and catalogues

User docs: `docs/matrix.md`. The analysis of an MMR angle is in `resonances/resonance/analysis/mmr/`.

## Catalogues (`matrix/matrix.py` base, `mmr/*_matrix.py`)

- `TwoBodyMatrix`, `ThreeBodyMatrix`, `RetrogradeTwoBodyMatrix` list resonances with their resonant semi-major axes.
  Two-body axes match Standish within ≤ 0.0012 AU (Neptune 0.016 AU); three-body axes match Nesvorný & Morbidelli 1998, Table 1, within
  1.6e-3 AU (`dev/verified.md`).
- A cached catalogue carries a build record `<file>.meta.json` (limits, planets, versions). A cache built with other
  settings, or without a record, is read as it is with a warning (`Matrix.check_cached`), never rebuilt or deleted
  silently: rebuilding changes which resonances a finder proposes.
- `TwoBodyMatrix` has no m > 0 "retrograde twins" any more; retrograde resonances are their own class.

## Retrograde resonances

- `RetrogradeTwoBody` (`mmr/retrograde_two_body.py`): the p/−q resonance of an orbit with i > 90° (`1J-3r0` = p, planet, q, k), the angle of Li,
  Huang & Gong 2019 eq. 2 (Morais & Namouni: λ* = M + ω − Ω, ϖ* = ω − Ω) from Ω, ω, M only (`calc_angle`).
  `RetrogradeTwoBodyMatrix` lists all k; for i > 90° bodies `find` gives only these.
- REBOUND's `orbit.l` and `orbit.pomega` are −λ* and −ϖ* there and jump when i crosses 90°: never build an angle from
  them (root rule). The `longitude`/`varpi` data columns of such bodies are in different conventions, hence no FAIR.

## Traps

- An MMR key starting with `2S…` once went to the secular parser (`create_resonance`, restore, CLI); keep the type
  detection tested.
- The finder warns about an alias before integrating; the output step must resolve the fastest angle (`dry_run`).
