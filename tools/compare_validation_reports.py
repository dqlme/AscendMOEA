"""Compare CPU and accelerator release-validation reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_report(path: Path) -> dict[str, object]:
    """Load one JSON report and require a successful validation status."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "ok":
        raise ValueError(f"{path} does not contain a successful validation report")
    return report


def result_index(report: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    """Index validation rows by category and component name."""
    rows = report.get("results")
    if not isinstance(rows, list):
        raise ValueError("validation report has no result list")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("validation result rows must be objects")
        key = (str(row.get("category")), str(row.get("name")))
        if key in indexed:
            raise ValueError(f"duplicate validation result: {key}")
        if row.get("status") != "ok":
            raise ValueError(f"unsuccessful validation result: {key}")
        indexed[key] = row
    return indexed


def require_equal(left, right, label: str) -> None:
    """Raise a descriptive error when two structural values differ."""
    if left != right:
        raise ValueError(f"{label} differs: {left!r} != {right!r}")


def compare_reports(
    left: dict[str, object],
    right: dict[str, object],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, object]:
    """Compare report structure and deterministic metric values."""
    for field in ("schema_version", "package", "version", "counts"):
        require_equal(left.get(field), right.get(field), field)

    left_rows = result_index(left)
    right_rows = result_index(right)
    require_equal(set(left_rows), set(right_rows), "validated component set")

    structural_fields = {
        "resources": ("mat_files",),
        "operators": ("operator_count", "reference_count"),
        "metrics": ("metric_count", "input"),
        "problem": ("M", "D", "FE"),
        "algorithm": (
            "problem",
            "max_fe",
            "budget_reached",
            "population",
            "algorithm_options",
        ),
        "extension": ("FE", "population"),
    }
    for key in sorted(left_rows):
        category, _ = key
        left_details = left_rows[key].get("details", {})
        right_details = right_rows[key].get("details", {})
        if not isinstance(left_details, dict) or not isinstance(right_details, dict):
            raise ValueError(f"{key} details must be objects")
        for field in structural_fields.get(category, ()):
            require_equal(
                left_details.get(field),
                right_details.get(field),
                f"{key} detail {field}",
            )

    metric_key = ("metrics", "public_metric_surface")
    left_values = left_rows[metric_key]["details"]["values"]
    right_values = right_rows[metric_key]["details"]["values"]
    require_equal(set(left_values), set(right_values), "metric name set")
    differences = {}
    for name in sorted(left_values):
        left_value = float(left_values[name])
        right_value = float(right_values[name])
        if not math.isclose(
            left_value,
            right_value,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            raise ValueError(
                f"metric {name} differs beyond tolerance: {left_value} != {right_value}"
            )
        absolute_error = abs(left_value - right_value)
        relative_error = absolute_error / max(abs(left_value), abs(right_value), 1e-300)
        differences[name] = {
            "absolute_error": absolute_error,
            "relative_error": relative_error,
        }

    return {
        "status": "ok",
        "left_device": left.get("device"),
        "right_device": right.get("device"),
        "components": len(left_rows),
        "metrics": len(left_values),
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "differences": differences,
    }


def main() -> int:
    """Parse command-line options and print a JSON comparison summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("tolerances must be nonnegative")

    summary = compare_reports(
        load_report(args.left),
        load_report(args.right),
        relative_tolerance=args.rtol,
        absolute_tolerance=args.atol,
    )
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
