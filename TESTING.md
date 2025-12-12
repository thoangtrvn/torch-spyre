# test operations used in models with torch.compile

## Overview
This repository provides a test framework for validating models with `torch.compile`.  
You can run all tests, selectively skip tests, or report detailed information about skipped and failed cases.

## How to run tests

### Run tests by default

executes all pytest-style files except ones deselected using pytest's markers by `addops` in `pytest.ini`

```
pytest tests/_inductor/models
```

### deselect behavior

default: `not paddedtensor and not largedimtensor and not fpoperation and not constant`

you can specify as you expected.

Examples

```
pytest -m "not padded tensor and not largedimtensor" tests/_inductor/models
pytest -m "fpoperation" tests/_inductor/models

Markers are defined in `pytest.ini`
