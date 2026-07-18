# Contributing

Contributions should preserve vectorized batch semantics and device-neutral
tensor placement. Keep unrelated refactoring out of focused changes and document
every new public symbol.

## Development setup

```bash
python -m pip install -e ".[dev,all]"
```

Accelerator-specific work requires mutually compatible PyTorch, backend, driver,
and toolkit versions. CPU-only workflows should not import or initialize an
accelerator backend.

## Changes to algorithms and problems

- Accept and return complete tensor batches; avoid per-individual Python loops.
- Allocate tensors on the input or problem device with the matching dtype.
- Preserve minimization semantics and the `constraint <= 0` feasibility rule.
- Record constructor parameters and public behavior in the matching API document.
- Describe behavior for duplicate solutions, constant dimensions, and empty
  feasible sets when these cases affect the public interface.

## Pull requests

A pull request should state the public behavior change, supported devices, and
documentation impact. Generated artifacts, local environments, benchmark output,
and machine-specific paths must not be committed.
