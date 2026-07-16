# Contributing

Contributions should preserve vectorized batch semantics, deterministic tests,
and device-neutral tensor placement. Keep unrelated refactoring out of focused
changes and document every new public symbol.

## Development setup

```bash
python -m pip install -e ".[dev,all]"
python -m ruff check src tests examples tools
python -m pytest
python tools/check_release.py
```

Run accelerator-specific tests in an environment whose PyTorch, backend, driver,
and toolkit versions are mutually compatible. CPU-only checks must not import or
initialize an accelerator backend.

## Changes to algorithms and problems

- Accept and return complete tensor batches; avoid per-individual Python loops.
- Allocate tensors on the input or problem device with the matching dtype.
- Preserve minimization semantics and the `constraint <= 0` feasibility rule.
- Add a deterministic regression test for corrected behavior.
- Record constructor parameters and public behavior in the matching API document.
- Validate normal and degenerate inputs, including duplicate solutions and
  constant dimensions.

## Pull requests

A pull request should state the behavioral change, numerical tolerance where
applicable, supported devices, and commands used for verification. Generated
artifacts, local environments, benchmark output, and machine-specific paths must
not be committed.
