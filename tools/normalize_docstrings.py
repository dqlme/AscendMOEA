"""Insert concise English docstrings without reformatting Python source."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

SPECIAL_FUNCTIONS = {
    "__init__": "Initialize the instance.",
    "__post_init__": "Validate and normalize initialized fields.",
    "__getattr__": "Delegate an unresolved attribute lookup.",
    "__getitem__": "Return the requested item.",
    "__setitem__": "Store the requested item.",
    "__len__": "Return the number of contained items.",
    "__iter__": "Iterate over the contained items.",
    "_solve": "Execute the vectorized evolutionary loop.",
    "setting": "Configure problem dimensions, bounds, encoding, and static data.",
    "cal_obj": "Evaluate objective values for a batch of decision vectors.",
    "cal_con": "Evaluate inequality constraints for a batch of decision vectors.",
    "get_optimum": "Generate objective-space reference points for metrics.",
    "get_pf": "Generate a drawable Pareto-front representation.",
    "evaluate": "Evaluate a complete batch of candidate solutions.",
    "evaluation": "Repair and evaluate a complete decision batch.",
    "initialization": "Create and evaluate an initial population.",
}


def words(name: str) -> str:
    """Convert a Python identifier to a readable lowercase phrase."""
    value = name.strip("_") or "operation"
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return value.replace("_", " ").lower()


def function_doc(name: str, module_path: Path) -> str:
    """Build a concise action-oriented function docstring."""
    if name in SPECIAL_FUNCTIONS:
        return SPECIAL_FUNCTIONS[name]
    phrase = words(name)
    stem = module_path.stem.lower()
    if name.startswith("load_"):
        return f"Load {words(name[5:])}."
    if name.startswith("get_"):
        return f"Return {words(name[4:])}."
    if name.startswith("set_"):
        return f"Set {words(name[4:])}."
    if name.startswith("is_") or name.startswith("has_"):
        return f"Return whether {phrase[3:]} holds."
    if name.startswith("repair_"):
        return f"Repair {words(name[7:])}."
    if name.startswith("select_"):
        return f"Select {words(name[7:])}."
    if name.startswith("update_"):
        return f"Update {words(name[7:])}."
    if stem in {"cec2019", "idmp", "idmp_e", "mmmop", "poly", "app"}:
        return f"Evaluate the {name} benchmark formula."
    return f"Compute {phrase}."


def class_doc(name: str, module_path: Path) -> str:
    """Build a concise class docstring from its package role."""
    parts = {part.lower() for part in module_path.parts}
    if "algorithms" in parts:
        return f"Implement the {name} multi-objective evolutionary algorithm."
    if "problems" in parts:
        return f"Implement the {name} vectorized optimization problem."
    return f"Represent {words(name)}."


def module_doc(path: Path, source_root: Path) -> str:
    """Build a concise module docstring from the relative source path."""
    relative = path.relative_to(source_root)
    if relative.name == "__init__.py":
        label = relative.parent.name or "AscendMOEA"
        return f'"""Public exports for the {label} package."""'
    if "algorithms" in relative.parts:
        return f'"""Implementation helpers for the {relative.stem} algorithm."""'
    if "problems" in relative.parts:
        return f'"""Vectorized implementation of the {relative.stem} problem."""'
    return f'"""Implementation utilities for {relative.stem}."""'


def insertion_index_for_module(lines: list[str]) -> int:
    """Return a legal module-docstring insertion index."""
    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1
    if index < len(lines) and "coding" in lines[index][:80]:
        index += 1
    return index


def normalize_file(path: Path, source_root: Path, check: bool) -> int:
    """Insert every missing module, class, and function docstring in a file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    missing_nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = {
                decorator.id
                for decorator in getattr(node, "decorator_list", ())
                if isinstance(decorator, ast.Name)
            }
            if "overload" in decorators:
                continue
            if not ast.get_docstring(node):
                missing_nodes.append(node)
    missing_module = ast.get_docstring(tree) is None
    count = len(missing_nodes) + int(missing_module)
    if check or count == 0:
        return count

    newline = "\r\n" if "\r\n" in source else "\n"
    lines = source.splitlines(keepends=True)
    for node in sorted(missing_nodes, key=lambda item: item.body[0].lineno, reverse=True):
        first = node.body[0]
        indent = " " * first.col_offset
        if isinstance(node, ast.ClassDef):
            text = class_doc(node.name, path.relative_to(source_root))
        else:
            text = function_doc(node.name, path.relative_to(source_root))
        lines.insert(first.lineno - 1, f'{indent}"""{text}"""{newline}')

    if missing_module:
        index = insertion_index_for_module(lines)
        lines.insert(index, module_doc(path, source_root) + newline + newline)

    path.write_text("".join(lines), encoding="utf-8", newline="")
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return count


def main() -> int:
    """Normalize a source tree or report its remaining docstring gaps."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.source_root.resolve()
    total = 0
    files = 0
    for path in sorted(root.rglob("*.py")):
        files += 1
        total += normalize_file(path, root, args.check)
    print(f"files={files} missing={total} mode={'check' if args.check else 'write'}")
    return int(args.check and total != 0)


if __name__ == "__main__":
    raise SystemExit(main())
