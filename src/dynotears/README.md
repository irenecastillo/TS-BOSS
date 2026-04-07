# dynotears — local copy from CausalNex

This folder contains files copied and lightly adapted from the
[CausalNex](https://github.com/mckinsey/causalnex) library, which is
**not installable on Python 3.12** (no compatible wheel exists).
The files are included here so the project can run without that dependency.

## Origin

| File | Original path in CausalNex repo |
|---|---|
| `dynotears.py` | `causalnex/structure/dynotears.py` |
| `structuremodel.py` | `causalnex/structure/__init__.py` (StructureModel class) |
| `transformers.py` | `causalnex/structure/transformers.py` |

Original repository: https://github.com/mckinsey/causalnex  
License: Apache License 2.0  
Copyright 2019-2020 QuantumBlack Visual Analytics Limited

## Changes made

- `dynotears.py`: the two `from causalnex.structure ...` import lines have been
  replaced with relative imports pointing to the local copies in this folder.
  All function names, signatures, and return types are unchanged.
- `structuremodel.py`: **unmodified** copy from the original repository.
- `transformers.py`: one line changed — `t.index.is_integer()` replaced with
  `pd.api.types.is_integer_dtype(t.index)` to silence a `FutureWarning` in
  pandas >= 2.0. All other code is unchanged.

## License

All three files remain under the original Apache 2.0 license.
See the license header at the top of each file and
[`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md) at the project root.
