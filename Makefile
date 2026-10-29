# Pass KEEP=1 to keep cache/tests/ results after a test run, e.g. `make test-benchmark KEEP=1`
KEEP ?=
# The tests run with the low-memory allocator on macOS (LOW_MEMORY_ALLOCATOR, docs/config.md: MallocLargeCache=0 returns
# freed large blocks; fast suite peak 2.98 -> 1.71 GB with the module caches released). LOW_MEMORY_ALLOCATOR=off: as before.
LOW_MEMORY_ALLOCATOR ?= on
ifeq ($(shell uname -s),Darwin)
ifneq ($(LOW_MEMORY_ALLOCATOR),off)
ALLOCATOR_ENV = MallocLargeCache=0
endif
endif

install:
	uv sync

test: install
	uv run flake8 --count
	uv run black . --check
	$(ALLOCATOR_ENV) KEEP_TEST_CACHE=$(KEEP) uv run pytest -v -m "not benchmark and not long" tests

test-only: install
	$(ALLOCATOR_ENV) KEEP_TEST_CACHE=$(KEEP) uv run pytest -v -m "not benchmark and not long" tests

# test-fast and test-slow need nothing outside git (tests/conftest.py hides benchmark/data from them); a test of more
# than 1 s is `slow`, one that reads local-only data (benchmark/data) is `benchmark` (tests/CLAUDE.md)
test-fast: install
	$(ALLOCATOR_ENV) KEEP_TEST_CACHE=$(KEEP) uv run pytest -v -m "not slow and not benchmark and not long" tests

test-slow: install
	$(ALLOCATOR_ENV) KEEP_TEST_CACHE=$(KEEP) uv run pytest -v -m "slow and not benchmark and not long" tests

# test-fast as on a fresh clone: the files git knows (tracked and new, not ignored; not experiments/ or tmp/) copied into
# a temporary folder, without benchmark/data and cache/
test-portable: install
	@tmp=$$(mktemp -d) && git ls-files -co --exclude-standard | grep -v '^experiments/\|^tmp/' | rsync -a --ignore-missing-args --files-from=- . $$tmp/ \
	  && cd $$tmp && env -u RESONANCES_BENCHMARK_DATA $(ALLOCATOR_ENV) $(CURDIR)/.venv/bin/python -m pytest -q -p no:cacheprovider \
	  -m "not slow and not benchmark and not long" tests; status=$$?; rm -rf $$tmp; exit $$status

# Integrations of several minutes and more (pytest marker `long`): the 10-Myr z1 run (~13 min)
test-long: install
	$(ALLOCATOR_ENV) KEEP_TEST_CACHE=$(KEEP) uv run pytest -v -m "long" tests

# the gates against the author's labels and every test that reads the local-only benchmark data
test-benchmark: install
	$(ALLOCATOR_ENV) KEEP_TEST_CACHE=$(KEEP) uv run pytest -v -s -m "benchmark" tests

.PHONY: docs run-docs docs-figures docs-examples publish-docs skill

# The documentation site (mkdocs-material; docs/cache, docs/examples and docs/tools are not published)
docs:
	NO_MKDOCS_2_WARNING=1 uv run mkdocs build --strict

run-docs:
	NO_MKDOCS_2_WARNING=1 uv run mkdocs serve

# Regenerate every figure the documentation uses from the saved benchmark series (no integration;
# needs benchmark/data, RESONANCES_BENCHMARK_DATA overrides it). ONLY=name,name redraws a few.
docs-figures:
	MPLBACKEND=Agg nice -n 10 uv run python docs/tools/build_figures.py $(if $(ONLY),--only $(ONLY),)

# Regenerate the references of the agent skill (resonances/skills/resonances/references) from docs/
skill:
	uv run python docs/tools/build_skill.py

# Run every code example of the documentation (short integrations, needs the network for NASA Horizons)
docs-examples:
	nice -n 10 uv run python docs/examples/run_all.py

publish-docs:
	NO_MKDOCS_2_WARNING=1 uv run mkdocs gh-deploy --strict

publish-test:
	uv build
	uv publish --index-url https://test.pypi.org/legacy/

publish:
	uv build
	uv publish

coverage: install
	uv run coverage run -m pytest -v tests/resonances
	uv run coverage report -m

clean:
	rm -f cache/allnum.cat
	rm -f cache/solar.bin
	rm -f cache/*.csv
	rm -f cache/*.png

cache-clear:
	rm -f cache/*.csv
	rm -f cache/*.png
