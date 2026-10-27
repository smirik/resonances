"""Write the references of the agent skill (resonances/skills/resonances/references/) from the documentation.

    uv run python docs/tools/build_skill.py          # rewrite the references (make skill)
    uv run python docs/tools/build_skill.py --check  # exit 1 when they are out of date with docs/

docs/ stays the single source: the installed wheel has no docs/, so the skill ships a generated copy of the pages
below, at the same relative paths (links between copied pages keep working). What mkdocs alone understands is
resolved: a snippet line `--8<-- "file.py:section"` becomes the code of that section of docs/examples/file.py,
`{ loading=lazy }` after an image is dropped, and a link to a page that is not copied, or to an image, becomes a
link to the published site. Admonitions (`!!! note`) and the LaTeX of the formulas stay as text. Edit the docs,
never the generated files; tests/resonances/skills/test_skill.py fails when they differ.
"""

from __future__ import annotations

import argparse
import posixpath
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'
SNIPPETS = DOCS / 'examples'
OUT = ROOT / 'resonances' / 'skills' / 'resonances' / 'references'
SITE = 'https://smirik.github.io/resonances/'

# The pages an agent needs to set up runs and read their results (the about, citation and benchmark pages are not)
PAGES = (
    'install.md',
    'quick-start.md',
    'core.md',
    'examples.md',
    'physics/overview.md',
    'physics/pendulum.md',
    'physics/signal.md',
    'physics/geometry.md',
    'physics/response.md',
    'physics/status-tree.md',
    'physics/confidence.md',
    'physics/secular-zlk.md',
    'physics/gallery.md',
    'matrix.md',
    'secular.md',
    'lidov-kozai.md',
    'clones.md',
    'dry-run.md',
    'plots.md',
    'output.md',
    'serialization.md',
    'coherence.md',
    'free-elements.md',
    'config.md',
    'libration-analysis.md',
    'classify.md',
)

HEADER = '<!-- Generated from docs/{page} by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->\n'

SNIPPET_LINE = re.compile(r'^(?P<indent>[ \t]*)--8<--\s+"(?P<file>[^":]+)(?::(?P<section>[\w-]+))?"\s*$')
SNIPPET_MARK = re.compile(r'--8<--\s*\[(start|end):[\w-]+\]')
IMAGE_ATTRS = re.compile(r'(!\[[^\]]*\]\([^)]*\))\{[^}]*\}')
# A markdown link or image target: ](target) — not inside code spans (the pages keep links out of code)
LINK = re.compile(r'\]\((?P<target>[^)\s]+)\)')
FENCE = re.compile(r'^\s*(```|~~~)')


def snippet(file: str, section: str | None, snippets_dir: Path = SNIPPETS) -> list[str]:
    """The lines of `section` of an example script (the whole file without a section), without the section marks."""
    lines = (snippets_dir / file).read_text(encoding='utf-8').splitlines()
    if section:
        start = f'--8<-- [start:{section}]'
        end = f'--8<-- [end:{section}]'
        begin = next((i for i, line in enumerate(lines) if line.strip().endswith(start)), None)
        stop = next((i for i, line in enumerate(lines) if line.strip().endswith(end)), None)
        if begin is None or stop is None or stop < begin:
            raise ValueError(f'snippet section {section!r} not found in {file}')
        lines = lines[begin + 1 : stop]
    return [line for line in lines if not SNIPPET_MARK.search(line)]


def page_url(target_page: str) -> str:
    """The published URL of a docs page (mkdocs directory URLs): physics/overview.md -> <site>physics/overview/."""
    stem = target_page[: -len('.md')]
    if stem == 'index':
        return SITE
    if stem.endswith('/index'):
        return SITE + stem[: -len('index')]
    return SITE + stem + '/'


def rewrite_link(target: str, page: str, pages: tuple[str, ...]) -> str:
    """A relative link of `page`: kept when it points to a copied page, else made absolute on the published site."""
    if re.match(r'^[a-z][a-z0-9+.-]*:', target) or target.startswith('#'):
        return target  # an URL, a mail link or an anchor in the same page
    path, _, anchor = target.partition('#')
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(page), path))
    if resolved.endswith('.md'):
        if resolved in pages:
            return target
        url = page_url(resolved)
    else:
        url = SITE + resolved  # an image or another file of the site
    return url + (f'#{anchor}' if anchor else '')


def convert(page: str, text: str, pages: tuple[str, ...] = PAGES, snippets_dir: Path = SNIPPETS) -> str:
    """The skill's copy of one docs page."""
    out = [HEADER.format(page=page)]
    in_fence = False
    for line in text.splitlines():
        match = SNIPPET_LINE.match(line)
        if match:
            indent = match.group('indent')
            out.extend((indent + code) if code else '' for code in snippet(match.group('file'), match.group('section'), snippets_dir))
            continue
        if FENCE.match(line):
            in_fence = not in_fence
        if not in_fence:
            line = IMAGE_ATTRS.sub(r'\1', line)
            line = LINK.sub(lambda m: '](' + rewrite_link(m.group('target'), page, pages) + ')', line)
        out.append(line)
    return '\n'.join(out).rstrip('\n') + '\n'


def build(out_dir: Path = OUT, docs_dir: Path = DOCS, pages: tuple[str, ...] = PAGES, snippets_dir: Path = SNIPPETS) -> list[Path]:
    """Write every page into `out_dir` (emptied first) and return the written files."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    written = []
    for page in pages:
        target = out_dir / page
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(convert(page, (docs_dir / page).read_text(encoding='utf-8'), pages, snippets_dir), encoding='utf-8')
        written.append(target)
    return written


def differences(out_dir: Path = OUT) -> list[str]:
    """The relative paths whose generated content differs from `out_dir` (missing, extra or changed files)."""
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / 'references'
        build(fresh)
        expected = {p.relative_to(fresh).as_posix(): p.read_bytes() for p in fresh.rglob('*') if p.is_file()}
    actual = {p.relative_to(out_dir).as_posix(): p.read_bytes() for p in out_dir.rglob('*') if p.is_file()} if out_dir.exists() else {}
    return sorted(name for name in expected.keys() | actual.keys() if expected.get(name) != actual.get(name))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true', help='exit 1 when the references are out of date')
    args = parser.parse_args(argv)
    if args.check:
        stale = differences()
        if stale:
            print('Out of date with docs/ (run `make skill`): ' + ', '.join(stale))
            return 1
        print(f'The skill references are up to date ({len(PAGES)} pages).')
        return 0
    written = build()
    print(f'Wrote {len(written)} pages into {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
