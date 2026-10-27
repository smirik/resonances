"""The agent skill shipped with the package: the generated references stay in sync with docs/, the skill is a valid
Agent Skill, and `resonances skill install` copies it where agents look for skills."""

import importlib.util
import re
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from resonances.cli.main import cli
from resonances.skills import SKILL_NAME, SkillExistsError, install_skill, skill_source

ROOT = Path(__file__).resolve().parents[3]
LINK = re.compile(r'\]\(([^)\s]+)\)')


def load_builder():
    spec = importlib.util.spec_from_file_location('build_skill', ROOT / 'docs' / 'tools' / 'build_skill.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frontmatter(path):
    text = path.read_text(encoding='utf-8')
    assert text.startswith('---\n'), 'SKILL.md must start with YAML frontmatter'
    return yaml.safe_load(text.split('---\n', 2)[1])


def test_references_are_in_sync_with_docs():
    """docs/ is the single source: `make skill` must have been run after the last change of a copied page."""
    stale = load_builder().differences(skill_source() / 'references')
    assert not stale, f'resonances/skills/resonances/references is out of date with docs/ (run `make skill`): {stale}'


def test_every_listed_page_is_copied_and_free_of_mkdocs_syntax():
    builder = load_builder()
    references = skill_source() / 'references'
    for page in builder.PAGES:
        text = (references / page).read_text(encoding='utf-8')
        assert text.startswith(f'<!-- Generated from docs/{page} ')
        assert '--8<--' not in text, page
        assert 'loading=lazy' not in text, page


def test_relative_links_resolve_inside_the_skill():
    """A relative link of SKILL.md or of a reference must point to a file of the installed skill (it has no docs/)."""
    source = skill_source()
    files = [source / 'SKILL.md', *sorted((source / 'references').rglob('*.md'))]
    broken = []
    for path in files:
        for target in LINK.findall(path.read_text(encoding='utf-8')):
            if re.match(r'^[a-z][a-z0-9+.-]*:', target) or target.startswith('#'):
                continue
            if not (path.parent / target.split('#')[0]).resolve().is_file():
                broken.append(f'{path.relative_to(source)} -> {target}')
    assert not broken, broken
    # the reference files named in SKILL.md's table exist
    skill_text = (source / 'SKILL.md').read_text(encoding='utf-8')
    named = set(re.findall(r'\b((?:physics/)?[a-z-]+\.md)\b', skill_text.split('## References', 1)[1]))
    assert named, 'SKILL.md lists its references'
    assert all((source / 'references' / name).is_file() for name in named), sorted(named)


def test_conversion_of_one_page(tmp_path):
    """The rules on a small docs tree whose expected output is written out by hand."""
    builder = load_builder()
    docs = tmp_path / 'docs'
    (docs / 'examples').mkdir(parents=True)
    (docs / 'physics').mkdir()
    (docs / 'examples' / 'demo.py').write_text(
        '"""doc"""\n\n# --8<-- [start:run]\nimport resonances\n\nsim = 1\n# --8<-- [end:run]\nother = 2\n', encoding='utf-8'
    )
    (docs / 'physics' / 'page.md').write_text(
        '# Page\n\n'
        'See [the run](../core.md#bodies), [benchmark](../benchmark.md), [same](#top), [web](https://x.org/a).\n\n'
        '[![fig](../img/a.png){ loading=lazy }](../img/a.png)\n\n'
        '```python\n--8<-- "demo.py:run"\n```\n\n'
        '!!! note "Kept"\n    text with $\\sigma$\n',
        encoding='utf-8',
    )
    (docs / 'core.md').write_text('# Core\n\n[Index](index.md)\n', encoding='utf-8')
    out = tmp_path / 'out'
    builder.build(out, docs_dir=docs, pages=('core.md', 'physics/page.md'), snippets_dir=docs / 'examples')

    site = 'https://smirik.github.io/resonances/'
    assert (out / 'physics' / 'page.md').read_text(encoding='utf-8') == (
        '<!-- Generated from docs/physics/page.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->\n\n'
        '# Page\n\n'
        f'See [the run](../core.md#bodies), [benchmark]({site}benchmark/), [same](#top), [web](https://x.org/a).\n\n'
        f'[![fig]({site}img/a.png)]({site}img/a.png)\n\n'
        '```python\nimport resonances\n\nsim = 1\n```\n\n'
        '!!! note "Kept"\n    text with $\\sigma$\n'
    )
    assert (out / 'core.md').read_text(encoding='utf-8').endswith(f'[Index]({site})\n')


def test_skill_frontmatter():
    meta = frontmatter(skill_source() / 'SKILL.md')
    assert meta['name'] == SKILL_NAME == skill_source().name
    assert re.fullmatch(r'[a-z0-9-]{1,64}', meta['name'])
    description = meta['description']
    assert 100 < len(description) <= 1024 and '<' not in description and '>' not in description
    assert set(meta) <= {'name', 'description', 'license', 'allowed-tools', 'metadata', 'compatibility'}


def test_cli_installs_into_a_project(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ['skill', 'install', '--path', str(tmp_path)])
    assert result.exit_code == 0, result.output
    target = tmp_path / '.claude' / 'skills' / 'resonances'
    assert str(target) in result.output
    assert frontmatter(target / 'SKILL.md')['name'] == 'resonances'
    for page in load_builder().PAGES:
        assert (target / 'references' / page).is_file(), page
    shipped = {p.relative_to(skill_source()) for p in skill_source().rglob('*') if p.is_file()}
    assert {p.relative_to(target) for p in target.rglob('*') if p.is_file()} == shipped

    # a second install refuses to overwrite and says how to
    (target / 'stale.md').write_text('from an older version', encoding='utf-8')
    result = runner.invoke(cli, ['skill', 'install', '--path', str(tmp_path)])
    assert result.exit_code == 1
    assert 'already installed' in result.output and '--force' in result.output
    assert (target / 'stale.md').exists()

    # --force replaces the folder: nothing of the old copy stays
    result = runner.invoke(cli, ['skill', 'install', '--path', str(tmp_path), '--force'])
    assert result.exit_code == 0, result.output
    assert not (target / 'stale.md').exists()
    assert (target / 'SKILL.md').read_bytes() == (skill_source() / 'SKILL.md').read_bytes()


def test_cli_user_install_respects_home(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ['skill', 'install', '--user'])
    assert result.exit_code == 0, result.output
    assert (home / '.claude' / 'skills' / 'resonances' / 'SKILL.md').is_file()
    assert not (tmp_path / '.claude').exists()  # not into the working directory

    result = CliRunner().invoke(cli, ['skill', 'install', '--user', '--path', str(tmp_path)])
    assert result.exit_code != 0  # --user and --path exclude each other


def test_install_defaults_to_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = install_skill()
    assert target == tmp_path / '.claude' / 'skills' / 'resonances'
    with pytest.raises(SkillExistsError):
        install_skill()
