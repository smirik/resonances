"""The agent skill shipped with the package (Agent Skills format: SKILL.md + references/).

`resonances skill install` copies `resonances/skills/resonances/` into a project's `.claude/skills/` (or the user's
`~/.claude/skills/` with `--user`). The references are a generated copy of the documentation
(`docs/tools/build_skill.py`, `make skill`); SKILL.md holds the workflow and the judgement.
"""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path
from typing import Optional

SKILL_NAME = 'resonances'


class SkillExistsError(FileExistsError):
    """The target folder already holds a skill of this name (pass force=True to replace it)."""


def skill_source() -> Path:
    """The folder of the shipped skill (SKILL.md and references/)."""
    return Path(str(resources.files('resonances') / 'skills' / SKILL_NAME))


def skills_dir(project: Optional[Path] = None, user: bool = False) -> Path:
    """Where agents look for skills: `<project>/.claude/skills` (default: the working directory) or `~/.claude/skills`."""
    base = Path.home() if user else Path(project) if project is not None else Path.cwd()
    return base / '.claude' / 'skills'


def install_skill(project: Optional[Path] = None, user: bool = False, force: bool = False) -> Path:
    """Copy the skill into the skills folder and return its new folder.

    An existing folder of the same name is replaced only with `force` (it is removed first, so no file of an older
    version stays behind); otherwise SkillExistsError.
    """
    target = skills_dir(project, user) / SKILL_NAME
    if target.exists() or target.is_symlink():
        if not force:
            raise SkillExistsError(str(target))
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(skill_source(), target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    return target
