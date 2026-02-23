# Contribution Guidelines

Thank you for your interest in contributing to Torch-Spyre! There are many
ways to contribute — bug reports, documentation improvements, new operation
support, and more.

For the full contribution guidelines, see the
[CONTRIBUTING.md](https://github.com/torch-spyre/torch-spyre/blob/main/CONTRIBUTING.md)
file in the repository root.

## Before You Start

* **Open an issue first** for large PRs so the team can align on the
  approach before you invest significant effort.
* **Sign off your commits** with `git commit -s` (Developer Certificate
  of Origin).

## Code Quality Standards

* Follow the **Google Python Style Guide** and **Google C++ Style Guide**.
* **Run pre-commit** before submitting to ensure linting passes:

  ```bash
  pip install pre-commit
  pre-commit run --all-files
  ```

  See the [pre-commit docs](https://pre-commit.com/#usage) if this is new
  to you.

* **Write tests** — both unit tests and integration tests — to keep the
  project correct and robust.

* **Document user-facing changes.** If your PR modifies how Torch-Spyre
  behaves from a user's perspective, add or update the relevant page under
  `docs/source/`. See the section structure in this documentation site for
  guidance on where to place new content.

* **Dev environment setup** — install development dependencies via:

  ```bash
  pip install -r requirements/dev.txt
  ```

## How to Extend the Compiler

The most common contribution is adding support for a new PyTorch operation.
See the [Spyre Inductor Operation Cookbook](../compiler/adding_operations.md)
for step-by-step patterns.

## Reporting Issues

Please open issues at
[github.com/torch-spyre/torch-spyre/issues](https://github.com/torch-spyre/torch-spyre/issues).
