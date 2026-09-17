# Verification suite

Canonical benchmarks with **known answers** — textbook plate and beam solutions, the
standard shell obstacle course, closed-form laminate results, conservation
identities, and analytic derivatives. Each one runs a mesh refinement sweep and reports its value against a
published or independently computed reference.

These are examples first and a regression gate second: every file is a readable,
self-contained script that a user can run to see Hermit reproduce a result they
already trust.

```bash
conda activate hermit
python examples/verification/ex_scordelis_lo.py            # convergence sweep
python examples/verification/ex_scordelis_lo.py --quick    # the single level CI runs
```

## Writing a new one

Copy [`ex_scordelis_lo.py`](ex_scordelis_lo.py) (curved geometry, distributed load)
or [`ex_pinched_cylinder.py`](ex_pinched_cylinder.py) (symmetry planes, point load).
The shape is always the same:

1. A module docstring whose **first line is the title** — the docs build turns it
   into a page and *requires* it. Follow it with the physical setup, the reference
   value, and what makes the case worth having.
2. A `solve_at(level) -> float` returning the single scalar the reference is quoted
   for.
3. A `CASE = Case(...)` declaring `reference`, `tolerance`, `citation`, `levels` and
   `quick_level`.
4. `if __name__ == "__main__": main(CASE)`.

Mesh builders live in [`_geometry.py`](_geometry.py) — add one there rather than
inlining a mesh in an example. Meshes are built **in memory**: the env has no meshio,
and a benchmark's mesh is a function of its refinement level anyway.

`quick_level` is the level CI runs, so it should be the cheapest one that is still
honest about the case. Prefer the coarsest level meeting `tolerance`; where no swept
level meets it, keep the most diagnostic affordable level rather than moving the
target (see `ex_ss_plate_uniform`). Keep a full sweep under ~30 s.

## Reference-value integrity

The failure mode this suite must not have is a benchmark that passes while verifying
nothing. Four rules, in priority order:

1. **Compute the reference rather than quoting it, wherever the closed form allows.**
   Sum the Navier series in NumPy; assemble the laminate `A`/`B`/`D` by hand from the
   ply definitions. A computed reference cannot be misremembered, and it documents
   itself. Quote a constant only when the reference genuinely is a published number
   from a converged numerical study (the obstacle-course cases).
2. **Every quoted constant carries a real citation** — author, work, and the table or
   equation it came from — in the `Case.citation` field and in
   [`REFERENCES.md`](REFERENCES.md).
3. **The reference and the tolerance are fixed before the first run and are not
   adjustable to fit the result.** If a case misses, report the convergence table and
   say so. A benchmark that misses its reference is a *finding about Hermit* — the
   API redesign turned up nine real defects this way, and several of them looked
   exactly like "this case is a bit off". Loosening the gate throws that signal away.
4. **Assert convergence, not just the endpoint.** A wrong reference constant usually
   shows up as a sweep converging beautifully to a different number, which is
   diagnostic. The harness warns when the error is not monotone under refinement.

## When a benchmark does not meet its reference

Set `open_finding=` on the `Case` with what is known and what is merely suspected.
The reference and tolerance stay at their correct values, the sweep still runs and
prints, and the test suite reports the case as an expected failure with that string as
the reason. Resolving one means fixing Hermit or correcting the reference — never
editing the field to be less honest.

This is not a formality. Of the five cases that first failed here, one was a wrong
analytic target, one a wrong load magnitude, one a measurement point at the wrong
place, one a genuine element defect (the warped-quadrilateral curvature, since fixed),
and one an optimiser failure (`ex_optimal_thickness_taper`, since resolved --- it
passes). Four of the five would have been erased by a slightly wider tolerance.

`ex_curved_beam` is the only case still carrying an `open_finding`, and it is a
documented element limitation on a deliberately extreme 1-element-wide strip rather
than a defect --- see its `open_finding` string for the measurements that establish
that.

## Warped quadrilateral cells

`ex_warped_quad_consistency.py` records the resolved warped-cell defect and gates its
replacement. The former curvature gave a 40% wrong twisted-beam answer that worsened
under refinement. Hermit's constant-normal, shallow-shell curvature now makes the
quadrilateral and triangular sequences converge together. Triangles are therefore no
longer required merely because a surface is doubly curved; see the formulation limits
in the user guide before applying this flat-facet theory to moderately thick,
strongly curved, membrane-loaded shells.

## Layout

| file | what |
|---|---|
| `_geometry.py` | analytic in-memory mesh builders (plate, cylinder, sphere, disk, annulus, twisted strip, curved beam) |
| `_harness.py` | `Case`, the convergence table, the pass/fail gate, the CLI |
| `ex_*.py` | one benchmark each |
| `REFERENCES.md` | every reference value with its source |
