# Testing Models with torch.compile

## Overview

This docucment explains how to run a subset of test cases for validating models with `torch.compile`.
You can:

* Run all tests
* Selectively skip tests
* Report detailed information about skipped and failed cases.

## How to run tests

### Run tests by default

By default, all pytest-style files are executed, except those deselected using pytest's markers defined in `pytest.ini`
via `addopts`.

```
pytest tests/_inductor/models
```

### Deselect behavior

Markers are defined in `pytest.ini`

Default deselection:

```
not paddedtensor and not largedimtensor and not fpoperation and not constant
```

You can override this behavior by specifying your own marker expression.

**Examples**:

```
pytest -m "not padded tensor and not largedimtensor" tests/_inductor/models
pytest -m "fpoperation" tests/_inductor/models
```
