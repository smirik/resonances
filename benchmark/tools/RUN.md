# Benchmark v2: интеграция пачек A–D

Скрипт: `tools/integrate.py` (описание режимов в его docstring). Пишет примеры в `data/examples/<id>/` по `LAYOUT.md`,
конфиг прогона в `data/runs/<run_id>/run.json` (+ `planets.csv.gz`), вывод самого пакета (summary.csv, segments.csv,
simulation.json, data-*.csv.gz) в `runs/<batch>/<run_id>/<task>/`, отметки готовности в `runs/<batch>/done/`.
`labels.csv` дополняет только главный процесс, под файловой блокировкой (`labels.lock`), поэтому пачки можно
запускать одновременно. Пачка E (вековые) здесь не запускается.

## Команды

Из корня репозитория. План (без интеграции) пишет `runs/<batch>/plan.csv` и `excluded.csv`:

```bash
uv run python benchmark/tools/integrate.py --batch A --plan
```

Полные прогоны (на Mac 18 ядер / 64 ГБ; 12 воркеров, как у коорбиталов):

```bash
caffeinate -is uv run python benchmark/tools/integrate.py --batch A --n-cores 12 2>&1 | tee benchmark/runs/A.log
caffeinate -is uv run python benchmark/tools/integrate.py --batch B --n-cores 12 2>&1 | tee benchmark/runs/B.log
caffeinate -is uv run python benchmark/tools/integrate.py --batch C --n-cores 12 2>&1 | tee benchmark/runs/C.log
caffeinate -is uv run python benchmark/tools/integrate.py --batch D --n-cores 12 2>&1 | tee benchmark/runs/D.log
```

После каждой пачки: переклассификация из сохранённых файлов и сравнение со статусом самой интеграции (для LK вместе с
free-omega gate). Должно быть N из N:

```bash
uv run python benchmark/tools/integrate.py --batch A --verify   # -> runs/A/verify.csv
```

Повторный запуск той же команды пропускает готовые тела (докрутка после сбоя). Упавшая задача пишет traceback в
`runs/<batch>/failed/<task>.txt`, остальные идут дальше. `--force` пересчитывает готовые тела, `--only "a,b"`
ограничивает список тел. Пробный прогон: `--dry-run --years 2000 --only ...`, пишет в `benchmark/dryrun/`
и никогда не трогает `data/` и `labels.csv`.

Планеты на эпохи всех прогонов уже лежат в `cache/` (MJD 58800 клонов скачана при пробном прогоне). Если кэша нет,
главный процесс скачивает его из Horizons до старта воркеров.

## Начальные условия и режимы

Эпоха элементов каждого тела проверяется на равенство дате `Simulation(date=...)` (assert в воркере). Тела всегда
добавляются словарём элементов. `finder.find` не вызывается, `DATA_SOURCE` и дата «сегодня» не используются.

| пачка | run | тела / примеры | элементы, эпоха | интегратор, span, шаг | классификатор |
|---|---|---|---|---|---|
| A | 2 | 212 / 215 | AstDyS osculating, MJD 61200 (2026-06-09) | SABA(10,6,4) dt 1, 100 тыс. лет; шаг 0.5 года при P < 4.2 года и двухтельном угле (63 тела), иначе 1 год | пакет по умолчанию |
| B | 7 | 19 / 27 | NEA: MPC neam00, JD 2461297.5 (как NEA production); клоны 88710: clones.json, MJD 58800; пары: AstDyS | ias15, 100 тыс. лет; шаг P/4.2 → 0.2–0.5 года (FAIR) у всех АСЗ, в том числе только с трёхтельными углами (при шаге 1 год орбита с P ≈ 0.9–1.5 года сэмплируется реже раза-двух за оборот: M, λ и короткопериодика a алиасятся); пары с трёхтельными углами — 1 год; 501710: 500 тыс. лет, шаг 2 года | NEA: CLASSIFIER_NEA (cutoff 0.01); пары: пакет |
| C | 1 | 32 / 32 | `coorbitals/data/sample_coorbitals.parquet`, JD 2461319.5 (MPC, у 8 тел JPL Horizons) | ias15 + Луна, 20 тыс. лет, шаг 0.1 года (как `run_20kyr.sh`) | CLASSIFIER_NEA, coherence off |
| D | 3 | 23 / 25 | AstDyS; 3 перепроверки: MPC, JD 2461297.5 | 500 тыс. лет, шаг 10 лет; SABA dt 0.5 при q ≥ 1.3 а.е. (12 тел), ias15 (11 тел) | пакет (free-omega gate включён) |

Планеты (`save_planets`) сохраняются в A, B, C (FAIR), один `planets.csv.gz` на run. Если планеты задачи расходятся с
файлом run больше чем на 1e-6 рад по λ, пример получает свой файл. В пробах расхождение 5e-12, это округление CSV.
SABA-задачи объединяют до 8 тел в одну Simulation: тестовые частицы не меняют интеграцию с постоянным шагом. ias15 —
одно тело на Simulation, так что орбита не зависит от соседей по пачке (у хаотических тел зависела).

Решения, которые стоит проверить:
- **Из A в B (ias15) перенесены 4 тела с q < 1.3 а.е.**: 620082 4J-1, 5646 3M-5, 6489 3J-1, 99942 7E-6. SABA не
  справляется с пересекающими орбиту Земли. Элементы для них взяты из выборки MPC.
- Пары 21436 (11J-3, 2M-4S-3) и 26420 (4J-5N-1) считаются ias15 на 100 тыс. лет: ias15 как в прогоне пар, а 100 тыс. лет —
  потому что старые метки -1 получены на 100 тыс. лет. 501710 продлён до 500 тыс. лет (автор: «extend time»; 200 тыс. лет
  уже извлечены).
- **2026 SZ2 и 2026 PC6 в пачку C не входят**: среди строк C их нет, а их 1-kyr ряды уже извлечены. Если они появятся
  среди тел, скрипт подставит JPL-элементы на ту же эпоху из `research/input_check/jpl_compare.csv` вместо ошибочных
  MPC (`ELEMENT_OVERRIDES`).
- **ZLK-перепроверки** 2023 RS2, 2010 HY22, 2020 PP1 записаны с `candidate_id` c0767. У 2023 RS2 (a = 0.979) и 2020 PP1
  (a = 1.006) дополнительно интегрируется угол 1:1 с Землей (`coorbital__2023_RS2__1E-1+0+0`,
  `coorbital__2020_PP1__1E-1+0+0__500kyr`, без метки). Он показывает, коорбиталь ли тело: для коорбиталей метод
  free-omega неприменим. 2010 HY22 (a = 3.66) — не коорбиталь, угла 1:1 у него нет.
- id: строки `partial` (данные другого режима уже извлечены) получают вариант-span (`__100kyr`, `__20kyr`,
  `__500kyr`), строки `yes` — без варианта, если id не занят другим прогоном.

## Пробный прогон (13.10.2026) и оценка стоимости

Все пачки прошли на 2–4 телах: 2–5 тыс. лет, а у A ещё полные 100 тыс. лет на 3 + 8 телах. Переклассификация
из файлов совпала со статусом интеграции во всех примерах: A 12/12, B 9/9, C 2/2, D 5/5. Повторный прогон 4/4
(после правок). Combined-рисунок (с FAIR) строится из файлов. `free_omega.csv.gz` копируется (проверено на 463,
50 тыс. лет).

Измерено:
- SABA dt 1, 100 тыс. лет, шаг 0.5 года, 8 тел в задаче: 80 с (интеграция 31 с, остальное — классификация и запись).
  Пиковая RSS 0.86 ГБ. 2 тела с шагом 1 год: 23 с.
- ias15: 1.5–2 с на тело на тысячу лет почти независимо от шага вывода. 88710 1.7, 99942 2.0, 6489 1.7, 21436 1.55,
  501710 1.45, 622577 и 189865 1.55 (на 20 тыс. лет). С Луной 3.8 с на тысячу лет.
- SABA dt 0.5, 50 тыс. лет, 1 тело: 3.1 с.
- series.csv.gz: 56.5 байт на строку (100 тыс. строк ≈ 5.7 МБ). Пакетный data-*.csv.gz примерно вдвое больше,
  планеты 9 × ~10 МБ на 200 тыс. строк.

| пачка | задач | CPU | стена на 12 воркерах | диск (series + пакет + планеты) |
|---|---|---|---|---|
| A | 27 (8 по 0.5 года ≈ 80 с, 19 по 1 году ≈ 50 с) | ≈ 25 мин | ≈ 5 мин | 1.6 + 3.1 + 0.1 ≈ 4.8 ГБ |
| B | 19 | ≈ 70 мин (100 тыс. лет ≈ 3–3.5 мин на тело; 501710 ≈ 12 мин) | ≈ 15 мин (держит 501710) | 0.4 + 0.6 + 0.9 ≈ 2 ГБ |
| C | 32 | ≈ 50 мин (≈ 1.5 мин на тело) | ≈ 5 мин | 0.4 + 0.7 + 0.1 ≈ 1.2 ГБ |
| D | 13 (11 ias15 по одному телу, 2 SABA по 8/4) | ≈ 2.5–3 ч (ias15 ≈ 12–15 мин на тело на 500 тыс. лет) | ≈ 15–20 мин | ≈ 0.2 ГБ |

Итого ≈ 5 CPU-ч, около 45 мин стены, если пачки идут подряд, и ≈ 8 ГБ на диске (свободно 1.4 ТБ). Самый большой
`series.csv.gz` — 28 МБ (99942, 500 тыс. строк с шагом 0.2 года). Оговорки:
- Стоимость ias15 растёт при тесных сближениях. Экстраполяция с 2–20 тыс. лет на 100–500 тыс. лет может занизить её
  для NEA пачек B и D (99942 в 2029 г., 189865 и 477492 с e > 1.3 у SABA).
- `source_package_commit` в meta: текущий коммит; `+dirty(resonances/)`, если пакет правится в рабочей копии.
  Интеграция идёт тем пакетом, что лежит в этом дереве, не в worktree с исправлениями.

## Строки, не интегрированные здесь (39, `runs/<batch>/excluded.csv`)

- 22 вековые строки (`needs_integration` yes/partial) — пачка E, сейчас не запускается.
- 17 сводных строк (`group:*`), из них одно тело × резонанс не следует: c0752 legacy-3, c0753 463_elements_sample,
  c0754–c0759 (вековые наборы), c0760–c0762, c0764–c0768 (ZLK-наборы; из c0767 взяты только три перепроверки),
  c0770 (Toro/Cruithne/...). Как их раскрывать, решать отдельно.

## Примеры из готовых прогонов (без интеграции)

Пример можно добавить из уже посчитанного прогона пакета (data-<тело>.csv, simulation.json): читатель и meta
`extract_existing.py` (`resolve_run`, `read_source`, `build_example`), `bench_io.write_example`. Строка candidates.csv
и labels.csv дописываются в конец; `label_source=author:<дата>`. `data_path` в candidates.csv пишется относительно
репозитория; данные на внешнем диске — как `<external>/...`, точка монтирования задаётся `RESONANCES_EXTERNAL_DATA`
(`extract_existing.resolve_data_path`). Абсолютные пути в отслеживаемые файлы не пишутся: репозиторий публичный.

- 2026-10-17 (`tmp/classifier-v2-2026-10-15/impl/w16/rename/add_examples.py`):
  - `mmr2__564160__3N-4+0+1__300kyr` (c0791): 300 тыс. лет, SABA(10,6,4) dt 1, шаг 15.92 года, Horizons на
    2026-10-16, 7 планет прогона. Ложный алиас, когда e проходит около нуля. Вердикт автора ожидается.
  - `zlk__162474__LK__1500kyr` (c0792): прогон test_zlk_coherence (Horizons 2026-06-09, SABA(10,6,4) dt 0.5,
    1.5 млн лет, шаг 2 года), хранится через 10 лет (LAYOUT.md, прореживание с проверкой). Автор: transient
    (смена острова 270° → 90°).
- 2026-10-18 (`tmp/classifier-v2-2026-10-15/impl/w18/integrate/add_as27.py`):
  - `mmr2__2013_AS27__31E-36+0+5` (c0793): ряд примера `mmr3__2013_AS27__4E+9S-5+0+0-8` (NEA production, ias15,
    20 тыс. лет, шаг 5 лет) с углом 31λ_E − 36λ + 5ϖ; λ_E из планет прогона R
    (`bench2-R-ias15-19.9999kyr-5.00122yr-jd2461297.5`, та же модель и те же моменты), угол хранится как
    np.unwrap(σ mod 2π). Автор: resonant, libration, подтип moving_centre, high (author:2026-10-17, принято при
    условии литературы: Gallardo 2006 с. 30, Wisdom 1985 с. 274). Метка также в author_label_decisions_2026-10-15.csv.

## Переклассификация и `make test-benchmark`

`tools/reclassify.py` классифицирует сохранённые примеры без интеграции (docs/libration-analysis.md). `--workers auto`
(по умолчанию) = min(CPU − 2, доступная память / 3 ГБ, 8): пик воркера ≈ 2.6 ГБ (v2-w21a, 37087 на 1 млн лет).
Если число воркеров ограничивает память, а не ядра, воркеры запускаются с экономным аллокатором и считаются по 1.6 ГБ
(`LOW_MEMORY_ALLOCATOR`, docs/config.md; на macOS это `MallocLargeCache=0`). Измерено 2026-10-25 на всех 719 при
6 воркерах (`run_perf.py`, W23): footprint воркера p50/p90/max 406/901/2336 МБ → 174/285/912 МБ, CPU 1523 → 1779 с,
результаты те же. Примеры идут от долгих к коротким по времени прошлого прогона (`bench_io.longest_first`): стена при
6 воркерах 268 → 236 с вместе с остальными рычагами W23 (memo, gmpy2).
CSV читаются через двоичную копию `data/cache/columns/` (`tools/column_cache.py`, пишется при первом чтении;
`--no-column-cache` или `BENCHMARK_COLUMN_CACHE=0` отключает; копия ≈ 7.4 ГБ, по `.npy` на столбец, столбцы планет
отдаются как read-only memory map; старые `.npz` формата 1 не читаются, их можно удалить). Измерено 2026-10-23: на 20
примерах загрузка 23.8 → 2.9 с; на всех 719 при 7 воркерах загрузка 411 → 34 с CPU, стена 332 → 249 с (первый прогон,
который пишет копию, 270 с). summary.csv и mmr_numbers.csv те же (кроме времени `sep_zone_seconds`). Тесты читают
серии через ту же копию (`tests/conftest.py`).

`make test-benchmark` гонит reclassify в `data/test-benchmark` и reclassify_secular_zlk в
`data/test-benchmark-secular-zlk` (87 примеров, секунды; `BENCHMARK_WORKERS` задаёт число воркеров обоих) и сверяет
расхождения с автором (MMR, коорбитали, вековые, ZLK) со списком `KNOWN_DISAGREEMENTS.csv` (LAYOUT.md). После правки,
которая сдвигает строку автора:

```bash
uv run python benchmark/tools/known_disagreements.py test-benchmark --secular-zlk test-benchmark-secular-zlk           # new / resolved / changed
uv run python benchmark/tools/known_disagreements.py test-benchmark --secular-zlk test-benchmark-secular-zlk --write   # переписать список
```

Новые строки получают `category=unreviewed` без причины; причину пишут руками.

Те же прогоны (по одному на сессию, фикстуры `reclassified_run` и `secular_zlk_run` в
`tests/resonances/benchmark/conftest.py`) читает второй шлюз — эпизоды, которых просил автор (`EPISODE_EXPECTATIONS.csv`,
цитаты в `EPISODE_QUOTES.json`, LAYOUT.md). Вековые и ZLK-случаи (E06, E56, E57, D01) он берёт из векового прогона (без
него, из командной строки, инструмент анализирует их сам, `reclassify_secular_zlk.analyse_one`, ≈ 1 с на все четыре). Шлюз падает на невыполненном случае
уровня A/B, которого нет в `known_fail`, на выполненном случае из `known_fail` (снять отметку) и на записи, которой нет в
прогоне; уровни C и D только печатаются. Проверить любой прогон:

```bash
uv run python benchmark/tools/episode_expectations.py test-benchmark                       # таблица и проблемы, код 1 при проблеме
uv run python benchmark/tools/episode_expectations.py v2-w23 --secular-zlk v2-w23-secular-zlk   # вековые/ZLK из готового прогона
```

На 26.10.2026 (W23, прогоны `v2-w23` и `v2-w23-secular-zlk`; то же с S4): A 45 PASS, 6 FAIL (все в `known_fail`: E15, E16, E17, E21, E30, E43; E16 удалён по ответу автора 26.10, Q4; E26 и E27 проходят с правилом a),
B 12/12, C 12 PASS и 4 FAIL, D 1 FAIL. Синтетика тех же механизмов: `tests/resonances/analysis/test_episode_detection.py`.

## Дерево фигур для просмотра (`tools/sort_figures.py`)

Раскладывает комбинированные фигуры примеров (`data/examples/<id>/combined_v2.png`; у производной строки
`combined_v2__<ключ>.png`; у вековых и ZLK то же имя) по вердиктам прогона переклассификации в
`data/figures/<пресет>/<уровни>/<example_id>.png` (`/` пресета пишется `-`, например
`data/figures/confidence-status/high/resonant/mmr2__12218__4M-7+0+3.png`; производная строка —
`<example_id>__<ключ>.png`). Уровни те же, что у пакета (`resonances/simulation/layout.py`, `PLOT_LAYOUT`): status,
confidence (пустой уровень — `none`), type, resonance, body; пресет по умолчанию `confidence/status`. Читает
`<run>/summary.csv` и, если есть, `<run>-secular-zlk/summary.csv`. Файлы — жёсткие ссылки (копия, если ссылку сделать
нельзя), в `data/examples` ничего не пишется; каждый запуск очищает и строит заново только свой
`data/figures/<пресет>/`. Папка — вердикт прогона, фигура — то, что нарисовал последний `reclassify.py --figures`
(может быть старше прогона); строка без фигуры печатается как missing.

```bash
uv run python benchmark/tools/sort_figures.py --run v2-w23 --dry-run --limit 20   # план без записи
uv run python benchmark/tools/sort_figures.py --run v2-w23                         # confidence/status
uv run python benchmark/tools/sort_figures.py --run v2-w23 --preset status/confidence
```

## Рисунки документации (`make docs-figures`)

`docs/tools/build_figures.py` заново строит все рисунки документации из сохранённых рядов бенчмарка, без интеграции:
галерею `docs/img/gallery/<example_id>.png` (38 примеров, по одному на статус и путь к нему; у 82831 ещё рисунок члена
мультиплета 13M-24+2+9) и схемы физических страниц `docs/img/physics/*.png`. Каждый пример анализируется заново
установленным пакетом (`analyse_mmr` для MMR и коорбиталей; `Simulation` из ряда и `identify_librations` для вековых и
ZLK, как в `reclassify_secular_zlk.py`) и рисуется кодом пакета. Скрипт пишет `docs/img/gallery/verdicts.json`,
переписывает `docs/physics/gallery.md` и печатает расхождения свежих вердиктов с последними сохранёнными прогонами
(`SAVED_RUNS` в скрипте: сейчас `v2-w24-tx9b` и `v2-w23-secular-zlk`; после нового прогона их стоит обновить).
В `data/` ничего не пишется (ни `combined_v2.png`, ни кэш столбцов); файл с теми же байтами не перезаписывается, так что
повторный запуск ничего не меняет. Без `data/` рисунки галереи остаются как есть, перестраиваются только схемы.
Список примеров и подписи — `CASES` в скрипте.

```bash
make docs-figures                                   # всё: ~2 воркера, nice -n 10
make docs-figures ONLY=alias,pendulum_portrait      # отдельные рисунки (якоря галереи или имена схем)
uv run python docs/tools/build_figures.py --list    # имена
```
