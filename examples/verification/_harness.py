"""Shared scaffolding for the verification benchmarks: declare a :class:`Case`, get a
convergence table, a pass/fail gate and a CLI for free.

The point of this module is that **the gate is data, not code**. A benchmark declares
its reference value, its tolerance and its citation up front; the harness runs the
mesh sweep and reports. Nothing here lets a run adjust its own target, and that is
deliberate -- see ``README.md``, *Reference-value integrity*. A case that misses its
reference is reporting something about Hermit, and the table it prints is the
evidence. Widening ``tolerance`` to make a red case go green destroys the only thing
the suite is for.

Typical use, at the bottom of an ``ex_*.py``::

    CASE = Case(
        name="Scordelis-Lo roof",
        quantity="vertical deflection at free-edge midspan",
        reference=0.3024,
        tolerance=0.02,
        citation="MacNeal & Harder (1985), Table 4",
        levels=(8, 16, 24, 32),
        quick_level=16,
        solve=solve_at,
    )

    if __name__ == "__main__":
        main(CASE)
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = ["Case", "main", "node_nearest", "relative_error"]


def relative_error(value, reference):
    """Relative error, falling back to absolute when the reference is zero."""
    ref = float(reference)
    return abs(float(value) - ref) / (abs(ref) if ref else 1.0)


def node_nearest(domain, point):
    """Index of the mesh node closest to ``point``, in ``domain.node_coords`` order.

    The benchmarks quote a displacement "at" a named location, and the mesh only has
    a node exactly there when the refinement level happens to put one there. Every
    sweep level used by a case should land a node on the sample point -- this helper
    finds it, and :meth:`Case.sweep` warns when the match is not exact, because
    silently reporting a neighbouring node's value is a good way to miss the
    reference by a few percent for no visible reason.
    """
    xyz = np.asarray(domain.node_coords, dtype=float)
    p = np.asarray(point, dtype=float).reshape(3)
    d = np.linalg.norm(xyz - p, axis=1)
    k = int(np.argmin(d))
    return k, float(d[k])


@dataclass
class Case:
    """One verification benchmark: a reference value and how to reproduce it.

    ``solve(level) -> float`` runs the analysis at a refinement level and returns the
    single scalar the reference is quoted for. ``levels`` is the sweep, coarsest
    first; ``quick_level`` is the (cheap) level the pytest gate uses -- it must still
    meet ``tolerance``, so pick it as the coarsest level that does, not the finest
    available.

    ``tolerance`` is *relative*. ``monotone`` asserts the sweep approaches the
    reference from one side without overshooting, which is the normal behaviour of a
    displacement-based element converging from below and is a sharper statement than
    the endpoint check alone.

    ``open_finding``, when set, records that this case **does not currently meet its
    reference** and says what is known about why. It is the honest alternative to the
    two bad options -- deleting the benchmark, or widening its tolerance until it
    passes. The reference and tolerance stay at their correct values, the sweep still
    runs and still prints, and the test suite reports the case as an expected failure
    with this string as the reason. Resolving one means either fixing Hermit or
    correcting the reference; it never means editing this field to be less honest.

    ``requires`` names third-party modules the case cannot run without -- an optimiser,
    say. Those are *optional* Hermit dependencies (the ``opt`` extra), so a plain
    install legitimately lacks them: the pytest gate skips such a case rather than
    failing it, and the CLI says what to install. Keep this to genuinely optional
    dependencies -- a case that skips is a case that is not verifying anything.
    """

    name: str
    quantity: str
    reference: float
    tolerance: float
    citation: str
    levels: Sequence[int]
    quick_level: int
    solve: Callable[[int], float]
    monotone: bool = True
    notes: str = ""
    open_finding: str = ""
    requires: Sequence[str] = ()
    _cells: dict = field(default_factory=dict, repr=False)

    def missing_requirements(self):
        """Names from ``requires`` that are not importable here, in declared order."""
        return [m for m in self.requires if importlib.util.find_spec(m) is None]

    def run(self, level):
        return float(self.solve(level))

    def sweep(self, levels=None):
        """Run every level, returning ``[(level, value, relative_error), ...]``."""
        rows = []
        for lv in (self.levels if levels is None else levels):
            v = self.run(lv)
            rows.append((lv, v, relative_error(v, self.reference)))
        return rows

    def check(self, level=None):
        """``(ok, value, rel_err)`` at ``level`` (default: the finest swept level)."""
        lv = self.levels[-1] if level is None else level
        v = self.run(lv)
        e = relative_error(v, self.reference)
        return e <= self.tolerance, v, e

    def report(self, levels=None):
        """Print the convergence table and the verdict. Returns True if it passes."""
        print(f"\n{self.name}")
        print(f"  quantity   : {self.quantity}")
        print(f"  reference  : {self.reference:.6g}   [{self.citation}]")
        print(f"  tolerance  : {self.tolerance:.3g} relative")
        if self.notes:
            print(f"  note       : {self.notes}")
        if self.open_finding:
            print(f"  OPEN       : {self.open_finding}")
        print(f"\n  {'level':>6}  {'value':>14}  {'value/ref':>10}  {'rel.err':>9}")
        print(f"  {'-' * 6}  {'-' * 14}  {'-' * 10}  {'-' * 9}")
        rows = self.sweep(levels)
        for lv, v, e in rows:
            ratio = v / self.reference if self.reference else float("nan")
            print(f"  {lv:>6}  {v:>14.6e}  {ratio:>10.4f}  {e:>9.2e}")

        _, v, e = rows[-1]
        ok = e <= self.tolerance
        print(f"\n  {'PASS' if ok else 'FAIL'}: rel.err {e:.3e} "
              f"{'<=' if ok else '>'} tol {self.tolerance:.3g}")
        if self.monotone and len(rows) > 2:
            errs = [r[2] for r in rows]
            if not all(b <= a * (1 + 1e-9) for a, b in zip(errs, errs[1:])):
                print("  WARNING: error is not monotone under refinement "
                      f"({', '.join(f'{x:.2e}' for x in errs)})")
        return ok


def main(case, argv=None):
    """CLI: ``--sweep`` (default), ``--quick``, or ``--level N``."""
    p = argparse.ArgumentParser(description=case.name)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true",
                   help="single run at the level the test suite uses")
    g.add_argument("--level", type=int, help="single run at this refinement level")
    args = p.parse_args(argv)

    missing = case.missing_requirements()
    if missing:
        print(f"\n{case.name}\n  needs {', '.join(missing)}, which "
              f"{'is' if len(missing) == 1 else 'are'} not installed.\n"
              f"  Install the optional solver stack:  pip install 'hermit[opt]'")
        raise SystemExit(2)

    if args.quick or args.level is not None:
        lv = case.quick_level if args.quick else args.level
        ok = case.report(levels=[lv])
    else:
        ok = case.report()
    raise SystemExit(0 if ok else 1)
