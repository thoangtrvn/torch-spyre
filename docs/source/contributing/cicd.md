# CI/CD

This page describes the continuous integration and continuous delivery
(CI/CD) infrastructure for Torch-Spyre. The goal is to ensure that every
code change is validated early, that releases are stable, and that
Spyre's integration health is publicly visible to the PyTorch community.

## Pipelines

### torch-spyre GitHub Actions

Every PR to `torch-spyre/torch-spyre` triggers the following workflows:

| Workflow | Description |
|----------|-------------|
| **Lint** | `ruff` for formatting, import sorting, and spell checking |
| **Unit tests (no device)** | All tests that do not require Spyre hardware |
| **Packaging** | Build source distribution and wheel; validate artifacts |
| **Device tests** | Triggered separately on Spyre hardware cluster |

When a GitHub release is published, an additional workflow pushes the
release artifacts to PyPI.

### Internal Spyre Stack (Jenkins)

An internal Jenkins pipeline runs on every PR to the proprietary Spyre
software stack:

1. Build the Spyre runtime container image
2. Run all internal Spyre component unit tests
3. Install `torch-spyre` and dev/test dependencies on the runtime image
4. Run `torch-spyre` unit and integration tests
5. Run the enabled subset of PyTorch CI tests

If all steps pass, the image is pushed to the container registry for
downstream consumption.

## PyTorch CI on Spyre

Torch-Spyre targets having Spyre's test results visible on
[PyTorch HUD](https://hud.pytorch.org). The rollout follows a phased
approach:

| Stage | Hardware | Test coverage | Frequency |
|-------|----------|---------------|-----------|
| 0 | 24 AIU cards | Internal + torch-spyre tests | On PR |
| 1 | 128 AIU cards | 5% in-scope PyTorch tests | Nightly |
| 2 | 512 AIU cards | 25% in-scope PyTorch tests | Nightly |
| 3 | 1 024 AIU cards | 50% in-scope PyTorch tests | 3–4×/day |
| 4 | 2 048 AIU cards | 80% in-scope PyTorch tests | On PR (non-blocking) |

Out-of-scope tests include all training tests and GPU-specific (CUDA)
tests. The target at Stage 4 is that CI runs on every PR without
blocking the ability to merge.

## Quality Criteria

| Metric | Target |
|--------|--------|
| Public accessibility | CI results visible to PyTorch open-source community |
| Flaky tests | < 5% of tests marked flaky |
| CI/CD stability | Outages recover within 3 days |
| Per-commit cycle time | Within a reasonable range for good developer experience |

## Running Tests Locally

Run the no-device unit test suite:

```bash
pip install -r requirements/dev.txt
pytest tests/ -m "not device"
```

Run the linter:

```bash
pip install pre-commit
pre-commit run --all-files
```
