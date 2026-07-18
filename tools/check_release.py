"""Run source-only release hygiene checks without importing the package."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from urllib.parse import unquote

TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}
REQUIRED_FILES = {
    ".gitignore",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "NOTICE.md",
    "README.md",
    "SECURITY.md",
    "pyproject.toml",
}
REQUIRED_PATHS = {".github/workflows/ci.yml"}


def _decorator_name(node: ast.expr) -> str:
    """Return the rightmost name of a decorator expression."""
    while isinstance(node, (ast.Call, ast.Attribute)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else ""


def check_docstrings(path: Path) -> list[str]:
    """Report missing module, class, and function docstrings in one Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    issues: list[str] = []
    if ast.get_docstring(tree) is None:
        issues.append(f"{path}: missing module docstring")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(_decorator_name(item) == "overload" for item in node.decorator_list):
            continue
        docstring = ast.get_docstring(node)
        if docstring is None:
            issues.append(f"{path}:{node.lineno}: {node.name} has no docstring")
            continue
        first_line = docstring.splitlines()[0]
        normalized_name = node.name.strip("_").replace("_", " ")
        generated = {
            "Initialize the instance.",
            f"Compute {normalized_name}.",
            f"Create {normalized_name}.",
            f"Return {normalized_name}.",
        }
        if first_line in generated:
            issues.append(f"{path}:{node.lineno}: {node.name} has a mechanical docstring")
    return issues


def check_markdown_links(path: Path, root: Path) -> list[str]:
    """Report local Markdown links whose target does not exist."""
    issues: list[str] = []
    text = path.read_text(encoding="utf-8")
    for match in re.finditer(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
        raw_target = match.group(1).strip().strip("<>")
        target = raw_target.split("#", 1)[0].strip()
        if not target or re.match(r"^(?:https?://|mailto:)", target, re.IGNORECASE):
            continue
        candidate = (path.parent / unquote(target)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            issues.append(f"{path}: local link escapes the project: {raw_target}")
            continue
        if not candidate.exists():
            issues.append(f"{path}: broken local link: {raw_target}")
    return issues


def iter_project_files(root: Path):
    """Yield project files while excluding generated directories."""
    for path in root.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        yield path


def check_forbidden_text(path: Path, checker: Path) -> list[str]:
    """Report development markers and machine-specific paths in release text."""
    if path == checker or path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "unfinished marker": r"\b(?:TO" + r"DO|FIXME|TBD)\b|占位|待完成|未完成提示",
        "retired package name": r"py" + r"platemo",
        "unrelated documentation name": r"\be" + r"vox\b",
        "machine-specific home path": r"(?:/root/|/home/[^/]+/)",
        "date-derived example seed": r"20260716",
    }
    issues: list[str] = []
    for label, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            line = text.count("\n", 0, match.start()) + 1
            issues.append(f"{path}:{line}: {label}")
    return issues


def main() -> int:
    """Validate a project tree and return a nonzero status for every finding."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    checker = Path(__file__).resolve()
    issues: list[str] = []

    missing = sorted(REQUIRED_FILES.difference(path.name for path in root.iterdir()))
    issues.extend(f"{root}: missing required file {name}" for name in missing)
    for relative_path in sorted(REQUIRED_PATHS):
        if not (root / relative_path).is_file():
            issues.append(f"{root}: missing required path {relative_path}")
    for path in iter_project_files(root):
        if path.suffix.lower() == ".py":
            issues.extend(check_docstrings(path))
        if path.suffix.lower() == ".md":
            issues.extend(check_markdown_links(path, root))
        issues.extend(check_forbidden_text(path, checker))

    generated = [
        path
        for path in root.rglob("*")
        if path.name == "__pycache__" or path.suffix.lower() in {".pyc", ".pyo"}
    ]
    issues.extend(f"{path}: generated Python artifact" for path in generated)

    for issue in issues:
        print(issue)
    print(f"release_hygiene={'ok' if not issues else 'failed'} findings={len(issues)}")
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
