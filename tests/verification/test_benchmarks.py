"""Run every verification benchmark at its ``quick_level`` and gate on its reference.

Auto-discovers ``examples/verification/ex_*.py``, so a new benchmark is covered the
moment it lands -- there is no list here to forget to update.

Each case here is one or more full solves, so this module is much heavier per test
than the unit suite; it still runs in the same process as everything else, because
``tests/conftest.py`` frees each test's FE ops at teardown.
"""

import importlib.util
import pathlib

import pytest

EXAMPLES = pathlib.Path(__file__).parents[2] / "examples" / "verification"


def _case_files():
    return sorted(EXAMPLES.glob("ex_*.py"))


def _load(path):
    spec = importlib.util.spec_from_file_location(f"_verif_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("path", _case_files(), ids=lambda p: p.stem)
def test_benchmark_matches_reference(path):
    mod = _load(path)
    case = getattr(mod, "CASE", None)
    assert case is not None, f"{path.name} must define a module-level CASE"

    if case.open_finding:
        pytest.xfail(f"{case.name}: known open finding -- {case.open_finding}")

    missing = case.missing_requirements()
    if missing:
        pytest.skip(f"{case.name}: needs optional {', '.join(missing)} "
                    f"(pip install 'hermit[opt]')")

    ok, value, err = case.check(case.quick_level)
    assert ok, (
        f"{case.name}: {case.quantity} = {value:.6e} at level {case.quick_level}, "
        f"reference {case.reference:.6e} [{case.citation}] -- relative error "
        f"{err:.3e} exceeds tolerance {case.tolerance:.3g}.\n"
        f"Do not widen the tolerance to make this pass; see "
        f"examples/verification/README.md, 'Reference-value integrity'."
    )


@pytest.mark.parametrize("path", _case_files(), ids=lambda p: p.stem)
def test_benchmark_is_declared_properly(path):
    """Cheap metadata gate -- no solve. Catches a case that ships without a real
    citation, or whose ``quick_level`` is not one of its swept levels."""
    case = getattr(_load(path), "CASE", None)
    assert case is not None, f"{path.name} must define a module-level CASE"
    assert case.citation.strip(), f"{case.name}: citation must not be empty"
    assert len(case.citation) > 20, f"{case.name}: citation looks like a placeholder"
    assert case.tolerance > 0, f"{case.name}: tolerance must be positive"
    assert case.quick_level in case.levels, (
        f"{case.name}: quick_level {case.quick_level} is not in levels {tuple(case.levels)}"
    )
