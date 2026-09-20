#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Keep every Python file in this repo a modern, up-to-date uv script.

What it does, for every ``*.py`` file in the repository (excluding virtualenvs,
caches and other generated directories):

1. **Makes it a uv script.** Any file whose shebang still says ``python`` /
   ``python3`` is rewritten to ``#!/usr/bin/env -S uv run --script`` and gets an
   inline script-metadata block (PEP 723) if it does not already have one.
2. **Pins Python 3.14.** The metadata always declares ``requires-python = ">=3.14"``.
3. **Uses the latest dependency versions.** Dependencies are taken from the
   existing metadata (plus anything imported by the file that is not in the
   standard library or a sibling module) and their version constraints are bumped
   to the latest release published on PyPI.

The script is itself a uv script and only uses the standard library, so it can
bootstrap with ``uv run update.py`` without any prior setup.

Usage:
    uv run update.py                 # rewrite files in place
    uv run update.py --dry-run       # show what would change, write nothing
    uv run update.py --check         # exit 1 if any file is out of date
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

UV_SHEBANG = "#!/usr/bin/env -S uv run --script\n"
REQUIRES_PYTHON = ">=3.14"

# Directories that never contain hand-written scripts we want to touch.
EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "node_modules",
    "build",
    "dist",
}

# Import name -> PyPI distribution name, for the cases where they differ.
IMPORT_TO_PYPI = {
    "attr": "attrs",
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "git": "GitPython",
    "jwt": "PyJWT",
    "PIL": "pillow",
    "pkg_resources": "setuptools",
    "serial": "pyserial",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "OpenSSL": "pyOpenSSL",
}

SHEBANG_RE = re.compile(r"^#!.*\n")
SCRIPT_BLOCK_RE = re.compile(r"^# /// script\n(?:#.*\n)*?# ///\n", re.MULTILINE)
IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][\w]*)", re.MULTILINE)
DEP_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*)$")

PYPI_LATEST_URL = "https://pypi.org/pypi/{name}/json"


def canonical(name: str) -> str:
    """PEP 503 normalisation, used to de-duplicate distributions."""
    return re.sub(r"[-_.]+", "-", name).lower()


def is_excluded(path: Path, root: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts)


def find_python_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.py")
        if p.is_file() and not is_excluded(p, root)
    )


def local_module_names(paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for p in paths:
        names.add(p.stem)
        # A package ``foo/__init__.py`` is importable as ``foo``.
        if p.stem == "__init__":
            names.add(p.parent.name)
    return names


def parse_deps_from_block(block: str) -> list[str]:
    """Read ``dependencies`` out of a raw inline-metadata block."""
    lines = []
    for line in block.splitlines():
        content = line[1:].lstrip(" ") if line.startswith("#") else line
        # The ``# /// script`` / ``# ///`` delimiters are not TOML.
        if content.startswith("///"):
            continue
        lines.append(content)
    try:
        data = tomllib.loads("\n".join(lines))
    except tomllib.TOMLDecodeError:
        return []
    deps = data.get("dependencies", [])
    return [d for d in deps if isinstance(d, str)]


def extract_metadata_block(text: str) -> str | None:
    match = SCRIPT_BLOCK_RE.search(text)
    return match.group(0) if match else None


def imported_distributions(text: str, local: set[str]) -> list[str]:
    """Map top-level third-party imports to PyPI distribution names."""
    found: dict[str, str] = {}
    for module in IMPORT_RE.findall(text):
        if module in sys.stdlib_module_names or module in local:
            continue
        if module.startswith("_"):
            continue
        dist = IMPORT_TO_PYPI.get(module, module)
        found[canonical(dist)] = dist
    return list(found.values())


def dep_name(spec: str) -> str:
    match = DEP_RE.match(spec.strip())
    return match.group(1) if match else spec.strip()


def latest_version(name: str, cache: dict[str, str | None]) -> str | None:
    key = canonical(name)
    if key in cache:
        return cache[key]
    url = PYPI_LATEST_URL.format(name=name)
    request = urllib.request.Request(url, headers={"User-Agent": "osrm-approx-update/1.0"})
    version: str | None
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
        version = payload["info"]["version"]
    except (urllib.error.URLError, KeyError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        print(f"  ! could not resolve latest {name!r}: {exc}", file=sys.stderr)
        version = None
    cache[key] = version
    return version


def resolve_dependencies(
    text: str,
    local: set[str],
    cache: dict[str, str | None],
) -> list[str]:
    """Return the full dependency list with latest-version lower bounds."""
    block = extract_metadata_block(text)
    declared = parse_deps_from_block(block) if block else []

    # Keep the PyPI spelling from existing metadata, add newly discovered imports.
    names: dict[str, str] = {}
    for spec in declared:
        name = dep_name(spec)
        if name:
            names[canonical(name)] = name
    for name in imported_distributions(text, local):
        names.setdefault(canonical(name), name)

    specs: list[str] = []
    for key in sorted(names):
        name = names[key]
        version = latest_version(name, cache)
        specs.append(f"{name}>={version}" if version else name)
    return specs


def build_block(deps: list[str]) -> str:
    if deps:
        dep_lines = "\n".join(f"#   {json.dumps(dep)}," for dep in deps)
        deps_toml = f"# dependencies = [\n{dep_lines}\n# ]"
    else:
        deps_toml = "# dependencies = []"
    return (
        "# /// script\n"
        f'# requires-python = "{REQUIRES_PYTHON}"\n'
        f"{deps_toml}\n"
        "# ///\n"
    )


def render(text: str, deps: list[str]) -> str:
    match = SHEBANG_RE.match(text)
    header = match.group(0) if match else ""
    rest = text[match.end():] if match else text

    # Drop any existing metadata block, then re-insert a fresh one after the shebang.
    rest = SCRIPT_BLOCK_RE.sub("", rest, count=1).lstrip("\n")

    if "uv run --script" not in header:
        header = UV_SHEBANG

    return f"{header}{build_block(deps)}{rest}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root (default: this file's directory)")
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    parser.add_argument("--check", action="store_true", help="exit non-zero if any file needs updating")
    args = parser.parse_args()

    root = args.root.resolve()
    files = find_python_files(root)
    local = local_module_names(files)
    cache: dict[str, str | None] = {}

    changed: list[Path] = []
    for path in files:
        original = path.read_text(encoding="utf-8")
        deps = resolve_dependencies(original, local, cache)
        updated = render(original, deps)

        if updated == original:
            continue

        changed.append(path)
        status = "update" if not args.dry_run else "would update"
        print(f"[{status}] {path.relative_to(root)}")
        if not args.dry_run:
            path.write_text(updated, encoding="utf-8")

    if not changed:
        print(f"All {len(files)} Python file(s) already up to date.")
        return 0

    verb = "would be updated" if args.dry_run else "updated"
    print(f"\n{len(changed)} of {len(files)} Python file(s) {verb}.")
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
