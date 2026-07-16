# Changelog

This file records the contents and validation status of each public release.

## 0.1.0 - 2026-07-16

### Added

- Public `Algorithm`, `Problem`, `Population`, `Workflow`, `Monitor`, and registry APIs.
- Twenty built-in multi-objective evolutionary algorithms.
- One hundred and fifty-five benchmark and real-world problems.
- Public tensorized selection, variation, diversity, sampling, and population operators.
- CPU and Huawei Ascend NPU device selection, deterministic seeding, and synchronized timing.
- Batch-oriented custom algorithm and custom problem extension examples.
- Chinese user, developer, performance, numerical-validation, troubleshooting, and release documentation.
- Complete release validation for every built-in algorithm and problem.

### Validation

- Ruff checks pass for source, tests, examples, and release tools.
- Nine unit tests pass on the Ascend server CPU environment.
- CPU smoke validation passes for 20 algorithms, 155 problems, public operators, and custom
  extensions from both editable and wheel installations.
- NPU smoke validation passes for 20 algorithms, 155 problems, public operators, and custom
  extensions from an isolated wheel installation.
