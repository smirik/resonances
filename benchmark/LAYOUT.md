# Benchmark v2: раскладка данных на диске

Корень данных: `benchmark/data/` (вне git). Код: `tools/bench_io.py` (чтение/запись, загрузчик; одна таблица старых
корзин `BUCKET_STATUSES` и списки статусов/уверенностей/слов угла для инструментов меток, W23),
`tools/extract_existing.py` (нарезка уже посчитанных прогонов). Всё, что ниже, читается без повторной интеграции:
классификация и все графики строятся из этих файлов.

```
benchmark/
  labels.csv                    единственный источник правды о корзинах (правится руками/скриптами разметки)
  extract_report.csv            что не удалось извлечь и почему
  data/
    examples/<example_id>/
      series.csv.gz             ряды одного примера (тело × резонанс)
      meta.json                 режим, происхождение, метки
      planets.csv.gz            (необязательно) symlink на ../../runs/<run_id>/planets.csv.gz
      free_omega.csv.gz         (необязательно) копия <name>-omega-free.csv источника, если была
      combined.png              combined-рисунок (tools/render.py, только из сохранённых файлов)
      render.json               настройки рисунка: cutoff и почему, FAIR или почему нет, время, коммит
    runs/<run_id>/
      run.json                  конфиг исходного прогона (simulation.json/run.json, скаляры) + список файлов
                                (tools/rerun.py дописывает тело и источник элементов в bodies/elements_sources)
      planets.csv.gz            элементы планет этого прогона, если источник их сохранял
    cache/columns/<путь файла>.columns/<столбец>.npy
                                двоичная копия series.csv.gz / planets.csv.gz для reclassify и тестов
                                (tools/column_cache.py): float64 без сжатия, по файлу на столбец (заглавная буква
                                пишется как ^ и строчная: Omega и omega на macOS иначе один файл), по реальному пути
                                файла, сверяется по mtime и размеру (__source__.npy); столбцы планет отдаются как
                                read-only memory map; можно удалить (старые <путь>.npz формата 1 больше не читаются)
    review/<rtype>/<bucket>/<confidence>/<example_id>.png
                                symlink на combined.png; генерируется из labels.csv (bench_io.build_review_tree),
                                руками не трогать; index.html в каждой папке (миниатюры) и в review/
```

Рисунки: `uv run python benchmark/tools/render.py [--only ID,..] [--rtype ..] [--sample N] [--force]
[--show-labels] [--workers N] [--tree-only]`. Пропускает готовые (если не `--force`), затем перестраивает `review/`.
Старые метки на рисунке только с `--show-labels` (по умолчанию нет, чтобы не влиять на пересмотр).

## example_id

`<rtype>__<name>__<resonance>[__<variant>]`. Пробелы → `_`, скобки и прочие символы кроме `[A-Za-z0-9+-.]` → `_`,
знаки `+` и `-` резонанса сохраняются. Примеры: `mmr2__1864__51M-50+0-1`, `coorbital__2017_XF62__1M-1+0+0`,
`secular__10__2_g-g6_+_s-s6_`, `zlk__3752__LK`. `variant` нужен только для второго ряда того же тела и резонанса
при другом режиме (база, направление, шаг, интегратор), например `zlk__3752__LK__500kyr`. Настоящие имя и резонанс
лежат в meta.json; id только стабильный ключ папки.

## series.csv.gz

Одна строка на момент вывода, шаг исходного прогона (вековые ряды прорежены, см. ниже). Значения как в
интеграции, без фильтров. `%.12g`.

| колонка | единицы | что это |
|---|---|---|
| `t_yr` | год | время от эпохи элементов (назад — отрицательное). Перевод из единиц REBOUND (`times/(2π)`) сделан один раз при извлечении. Год здесь гауссов: 2π единиц REBOUND при G=1, а.е., M☉ = 365.2569 сут, не юлианский (365.25). Дата строки: `epoch_jd + t_yr·365.2569` (проверено по средней долготе Земли: с 365.25 за 20 тыс. лет набегает > 100°) |
| `a` | а.е. | большая полуось |
| `e` | — | эксцентриситет |
| `inc` | рад | наклон к эклиптике J2000 |
| `Omega`, `omega` | рад | долгота узла, аргумент перицентра (как в источнике, обычно [0, 2π)) |
| `varpi` | рад | Omega + omega, не свёрнут |
| `M`, `mean_longitude` | рад | средняя аномалия, средняя долгота (пусто, если источник их не хранил: фикстуры free-omega). До 2026-10-15 колонка звалась `lam`; загрузчик читает оба имени |
| `angle_raw` | рад | развёрнутый угол резонанса, без фильтра. MMR и ZLK: `<key>_angle_unwrapped` источника. Если источник угла не хранил, угол строится из λ, ϖ тела и планет прогона и пишется как np.unwrap(σ mod 2π); это записано в `meta.angles` и `provenance.derived` (`mmr2__2013_AS27__31E-36+0+5`). Вековой: угол из оскулирующих varpi и Omega тела (Omega от инвариантной плоскости) против мод-долгот планет g_j·t + φ_j, без вычитания вынужденных членов (`secular_angles_osculating` пакета) |
| `angle_reduced` | рад | только вековой: классифицируемый угол пакета (reduced: вычтены вынужденные члены на фундаментальных частотах, которых нет в угле), развёрнут |
| `src_angle` | рад | только вековой: угол, который классифицировал исходный прогон (`<key>_angle_unwrapped`); его смысл зависит от коммита, см. `meta.angles` |

Углы `angle_raw`/`angle_reduced` вековых примеров пересчитаны при извлечении текущим пакетом из элементов тела
на полном шаге (фазы планет: линейный закон или измеренные по `planets.csv.gz`, см. `meta.regime.secular_phases`),
затем прорежены. Расхождение с углом источника записано в `meta.angles.check`.

Прореживание (только вековые ряды на 10–20 млн лет): простое взятие каждой k-й строки, без сглаживания. Шаг не
больше 500 лет и не больше 1/20 самого короткого периода среди varpi, Omega и угла тела. Факторы и периоды —
в `meta.regime.decimation`.

Другой ряд можно хранить прореженным, если вердикт на прореженном ряду проверен и совпадает с вердиктом на исходном
шаге (все колонки вердикта summary.csv, принятые эпизоды). Проверка и её скрипт записываются в
`meta.regime.decimation.verified`. Пример: `zlk__162474__LK__1500kyr`, 1.5 млн лет с шагом 2 года хранится через
10 лет (каждая 5-я строка, `impl/w16/rename/verify_thinning.py`: вердикт тот же, числа отличаются меньше чем на 0.2 %).

## planets.csv.gz

Широкая таблица: `t_yr` и `<Planet>.<поле>` для полей `a, e, inc, Omega, omega, M, mean_longitude, varpi` (единицы
как выше; до 2026-10-15 `mean_longitude` звалась `l`, загрузчик читает оба и отдаёт пакету ключ `l` planets_data). Один файл на прогон, в примере только symlink. Для MMR-прогонов хранятся планеты, которые
встречаются в резонансах извлечённых примеров (их хватает для FAIR); для вековых — все (измеренные фазы мод).

Если прогон примера планет не сохранял, `meta.provenance.planets_run_id` указывает прогон той же модели (эпоха,
интегратор, планеты, та же сетка), чьи планеты воспроизводят сохранённый угол; загрузчик пакета
(`resonances.resonance.analysis.mmr.io`) читает его раньше `run_id`. Так подключены 118 примеров NEA production
(2026-10-16, прогон R 19.9999 тыс. лет, проверка в `provenance.planets_link`). Планетам, которых нет и там,
загрузчик даёт каталожные долготы (`MMRSeries.catalogue_longitudes`, только поиск владельца): λ из первой строки
файла планет прогона, иначе из кэша Солнечной системы на эпоху (`provenance.solar_system_cache` или run.json).

## meta.json

```
example_id, candidate_id, name, resonance, rtype, resonance_type (mmr | secular | lidov_kozai)
regime:     integrator, dt (в единицах REBOUND, год/2π = 58.13 сут; у ias15 только первый шаг), span_yr, step_yr, n_rows, direction, epoch_jd, epoch_date, elements_source,
            planets, moon, secular_phases, decimation {factor, source_step_yr, step_yr, shortest_period_yr} | null
provenance: run_id, source_files [{path, size, sha256}], source_columns {колонка series: колонка источника},
            source_package_commit, extracted {script, package_commit, created},
            planets (откуда планеты), planets_run_id + planets_link {why, check, read_by, date} (необязательно)
labels:     old_statuses, author_verdict, reliability, sources, tags, proposed_bucket, confidence,
            source_status (статус в самом исходном прогоне, если есть summary.csv)
angles:     описание angle_raw / angle_reduced / src_angle и check (макс. расхождение в градусах)
classifier_config: скалярные настройки исходного прогона (cutoff, периодограмма, classify_*, secular_*, free_*)
source_extra: дополнительные сведения источника (например, режим коорбитала из regimes_summary.csv)
notes
```

## labels.csv

`example_id, candidate_id, rtype, proposed_bucket, confidence, author_bucket, claude_bucket, comment`.
`proposed_bucket` и `confidence` приходят из `candidates.csv`. `author_bucket` ставит только автор,
`claude_bucket` — только Claude/LLM-судья (никогда не считается вердиктом автора). Скрипты добавляют строки и
обновляют `proposed_bucket`/`confidence`, но заполненные `author_bucket`, `claude_bucket`, `comment` не трогают.
Папки `review/` строятся по `author_bucket`, если он есть, иначе по `proposed_bucket`.

Разметка автора: `tools/build_review_page.py` собирает `review_page/index.html` (+ `data.js`; открывается из file://,
метки в localStorage браузера). Экспорт страницы `author_labels_<дата>.csv` кладётся в `benchmark/`, затем
`tools/import_author_labels.py [файл] [--dry-run] [--tree]` пишет колонки `author_bucket` (статус в написании корзин:
`non_resonant` → `non-resonant`), `author_confidence`, `author_angle_status`, `author_comment`, `author_discuss`,
`author_agreed_with_claude`, `author_labelled_at`. `claude_*` и прочие колонки не трогает; более старый экспорт
не перекрывает более новую метку.

Известные расхождения: `KNOWN_DISAGREEMENTS.csv` (в git) — по строке на (example_id, field), field из status,
angle_status, confidence (метки автора, только где он дал значение) и author_verdict (candidates.csv; не сравнивается
для записи с собственной меткой автора со статусом, R-AV 2026-10-24); колонки
`example_id, field, author, package, category, reason, since`. `category` и `reason` — прочтение Claude (open_question,
step_aliasing, cs12, q1d, calibration, old_verdict), а не метка автора. `make test-benchmark` падает на любом
расхождении не из списка, на исчезнувшем из прогона и на изменившемся (`tools/known_disagreements.py <run>
[--secular-zlk <run>] [--write]`). Вековые и ZLK-примеры (свой прогон `tools/reclassify_secular_zlk.py`) сверяются так же:
метки автора и, как author_verdict, вердикты автора `reclassify_secular_zlk.AUTHOR_VERDICTS` (14 записей, прочтение
Claude слов автора с источником; запись без строки в прогоне = `missing`). Без векового прогона его строки списка не
проверяются и не удаляются.

Эпизоды, которых просил автор: `EPISODE_EXPECTATIONS.csv` (в git, W23) — 79 случаев, по строке на запись: `case,
example_id, tier, scope, mechanism, expected, checks, sources, known_fail, known_reason`. `tier`: A — слова автора об
эпизодах этой записи, B — вывод разбора, который он принял, C — старый `author_verdict` с интервалом (подсказка), D —
аналогия с его решением по другому объекту; проверяют только A и B. `checks` — JSON-список операций
(`tools/episode_expectations.py`, `OPS`), `sources` — file:line цитат; сами цитаты дословно, с файлом и строкой, в
`EPISODE_QUOTES.json`. `known_fail=1` и `known_reason` — случаи A/B, которые сейчас не выполняются, и прочтение Claude
почему (не метка автора). `make test-benchmark` падает на невыполненном случае A/B не из списка, на выполненном из
списка и на отсутствующей записи (`tools/episode_expectations.py <run>`).

Калибровка уверенности: `tools/calibrate.py [--run v2] [--report FILE]` считает ячейки (status, angle_status,
reason) по вердиктам автора на прогоне и пишет `resonances/data/confidence_cells.json` и `<run>/calibration_rows.csv`
(docs/physics/confidence.md).

Корзины: `resonant, transient, non-resonant, near-separatrix, slow-circulation, insufficient-time, uncertain,
chaotic, kinematic-libration`. Уверенность: `high, medium, low`.

## Загрузка

```python
from bench_io import load_example, to_body
ex = load_example('mmr2__1864__51M-50+0-1')       # ex.meta, ex.series (DataFrame), ex.planets (или None)
body, resonance, sim = to_body(ex)                 # resonances.Body + объект с .times/.config/.integration_engine
```
`sim` — заглушка с тем, что читают плоттеры (`MMRPlotter.from_body(body, resonance, sim)`,
`SecularPlotter.from_body(body, resonance, sim)`, `FreeOmegaPlotter.from_body(body)` после
`body.build_free_elements(...)`) и классификатор (`classify_resonance(body, resonance, sim.config)`).
