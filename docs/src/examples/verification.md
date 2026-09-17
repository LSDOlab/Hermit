# Verification benchmarks

Canonical cases with known answers: textbook plate and beam solutions, the standard
shell obstacle course, closed-form laminate results, conservation identities, and
analytic derivatives. Each script runs a mesh-refinement sweep and reports its converged value
against a published or independently computed reference.

These are examples first and a regression gate second. Every file in
`examples/verification/` is a readable, self-contained script you can run to watch
Hermit reproduce a result you already trust:

```sh
conda activate hermit
python examples/verification/ex_scordelis_lo.py            # full convergence sweep
python examples/verification/ex_scordelis_lo.py --quick    # the single level CI runs
python examples/verification/ex_scordelis_lo.py --level 32
```

Each run prints the reference, its citation, the tolerance, and a level-by-level
table of value, value/reference and relative error, ending in `PASS` or `FAIL`. The
harness warns separately when the error is *not* monotone under refinement, because a
sweep that converges beautifully to the wrong number is diagnostic of a bad reference
rather than a bad element.

## Where the reference values come from

The failure mode this suite must not have is a benchmark that passes while verifying
nothing. Four rules govern every reference, in priority order.

**1. Compute the reference rather than quoting it, wherever a closed form allows.**
Most cases sum their own Navier series in NumPy, or assemble the laminate $A$/$B$/$D$
by hand from the ply definitions, inside the example file. A computed reference cannot
be misremembered. A constant is quoted only when the reference genuinely *is* a
published number from a converged numerical study — in practice, the obstacle-course
cases.

**2. Every quoted constant carries a real citation** — author, work, and the table or
equation it came from — in the case's `citation` field, and therefore in the generated
[reference table](../_temp/examples/verification/REFERENCES).

**3. The reference and the tolerance are fixed before the first run**, and are not
adjustable to fit the result. A case that misses its reference is reporting a
*finding about Hermit*; the API redesign turned up nine real defects exactly this way,
and several of them looked at first like "this case is a bit off". Loosening the gate
throws that signal away.

**4. Quoted constants get an independent review pass** that re-derives or re-sources
the number without seeing the original justification. Disagreements are resolved
before the example lands, never by adjusting the tolerance.

`REFERENCES.md` is *generated* from the `Case` objects by
`python examples/verification/_references.py`, and CI fails if it is stale, so the
table cannot drift away from the declarations it describes.

## When a case does not meet its reference

The case sets `open_finding=` with what is known and what is merely suspected. The
reference and tolerance stay at their correct values, the sweep still runs and prints,
and the test suite reports the case as an expected failure with that string as the
reason. Resolving one means fixing Hermit or correcting the reference — never editing
the field to be less honest.

This is not a formality. Five cases failed on first run: one had a wrong analytic
target, one a wrong load magnitude, one a measurement point at the wrong place, one a
genuine element defect (the warped-quadrilateral curvature, since fixed), and one an
optimiser failure (`ex_optimal_thickness_taper`, since resolved). Four of the five
would have been erased by a slightly wider tolerance.

One case is currently open: `ex_curved_beam` plateaus about 3.9% below the
MacNeal–Harder value on a deliberately extreme one-element-wide strip. Widening the
strip to four elements closes the gap to 1.9%, and the in-plane load case converges to
1.3%, so this is a documented element limitation for out-of-plane twisting at extreme
aspect ratio, not a general membrane offset.

## The cases

### Analytic plate and beam solutions

The reference is a closed form the example evaluates itself.

| case | quantity | reference |
|---|---|---|
| `ex_cantilever_tip_load` | tip deflection | Timoshenko beam formula, computed |
| `ex_cantilever_tip_moment` | tip deflection and rotation | Euler–Bernoulli constant curvature, computed |
| `ex_cantilever_uniform_pressure` | tip deflection | Timoshenko beam formula, computed |
| `ex_ss_plate_uniform` | centre deflection | Kirchhoff–Love Navier double series, 801 terms/direction |
| `ex_ss_plate_sinusoidal` | centre deflection | Kirchhoff–Love single Navier mode |
| `ex_clamped_plate_uniform` | centre deflection | Timoshenko & Woinowsky-Krieger (1959), Table 35, p. 202 |
| `ex_clamped_circular_plate` | centre deflection | $w = q a^4 / (64 D)$, computed |
| `ex_thin_limit_shear` | FE / analytic ratio vs. $a/h$ | Reissner–Mindlin single mode, computed |

`ex_thin_limit_shear` sweeps slenderness rather than mesh size: a ratio that falls
away as the plate thins is shear locking.

### The shell obstacle course

Published constants from converged numerical studies, all from MacNeal & Harder,
*A proposed standard set of problems to test finite element accuracy*, Finite Elements
in Analysis and Design **1**(1), 1985.

| case | quantity | reference |
|---|---|---|
| `ex_scordelis_lo` / `ex_scordelis_lo_quad` | free-edge midspan deflection | 0.3024 |
| `ex_pinched_cylinder` | radial deflection under the load | 1.8248e-5 |
| `ex_pinched_hemisphere` / `ex_pinched_hemisphere_quad` | radial deflection under the load | 0.0924 |
| `ex_twisted_beam` | tip deflection in $z$ | 0.005424 |
| `ex_curved_beam` | tip deflection in $z$ | 0.5022 (open finding, above) |
| `ex_cooks_membrane` | mid-right-edge deflection | 23.9 — Cook (1974) |

Scordelis–Lo and the pinched hemisphere each appear twice, on triangles and on
quadrilaterals. That pairing is deliberate: it is the standing check that the two cell
types converge to the same answer.

### Warped quadrilaterals

| case | what it gates |
|---|---|
| `ex_warped_quad_consistency` | warped-quad tip deflection / triangle tip deflection → 1 |
| `ex_hypar_warped_quad` | hypar free-edge deflection against a converged triangle solution |

These record the resolved warped-cell defect and gate its replacement. The former
curvature differentiated the cell normal along with the rotation, which stored bending
energy under rigid-body motion; it gave a 40% high twisted-beam answer that got *worse*
under refinement, and the consistency ratio converged to 1.407. Hermit's
constant-normal, shallow-shell curvature makes the quadrilateral and triangular
sequences converge together. Triangles are therefore no longer required merely because
a surface is doubly curved — but see the formulation limits in {doc}`../background`
before applying flat-facet theory to a moderately thick, strongly curved,
membrane-loaded shell.

### Laminates and failure

| case | quantity | reference |
|---|---|---|
| `ex_clt_abd` | $A$/$B$/$D$ assembly discrepancy | independent NumPy evaluation of the CLT thickness integrals |
| `ex_laminate_matches_isotropic` | laminate vs. `hm.isotropic` difference | same-run isotropic solution |
| `ex_crossply_bending` | midpoint deflection | one-term Navier solution from the independently assembled $D$ |
| `ex_tsai_wu_uniaxial` | Tsai–Wu field and KS discrepancy | criterion evaluated by hand from the ply strengths |

`ex_laminate_matches_isotropic` is the cross-path check: two spellings of one physical
shell — an isotropic material and a laminate of isotropic plies — must agree to
round-off. It is gated at a relative 1e-8.

### Conservation and identities

| case | quantity | reference |
|---|---|---|
| `ex_mass_and_cg` | mass and CG, uniform and graded thickness | closed-form area integrals |
| `ex_elastic_energy_identity` | energy / compliance identity residual | Hermit's own output conventions |

These have a reference of zero and a tolerance near machine precision: their residual
is round-off in the direct solve, not a discretisation error.

### Derivatives

| case | quantity | reference |
|---|---|---|
| `ex_analytic_compliance_gradient` | $\mathrm{d}C/\mathrm{d}t \,/\, (-3C/t)$ | Euler–Bernoulli pure-bending scaling |
| `ex_gradient_finite_difference` | CSDL vs. central differences, thickness | second-order central-difference formula |
| `ex_shape_derivative_fd` | CSDL vs. central differences, mesh coordinates | second-order central-difference formula |
| `ex_shape_derivative_direction_failure` | CSDL vs. central differences, Tsai-Wu index under a `fiber_direction` | second-order central-difference formula |
| `ex_optimal_thickness_taper` | optimiser-recovered taper exponent | $t \propto \sqrt{M}$, hence $(L-x)$; cross-checked against a 1-D SLSQP solve |

`ex_gradient_finite_difference` and `ex_shape_derivative_fd` run a step study, so the
reported number separates FD truncation error from cancellation; their tolerances are
FD-limited, not Hermit-limited. `ex_shape_derivative_direction_failure` differs in one
important way: its finite differences perturb the **mesh coordinates** and rebuild the
mesh, rather than perturbing a `node_disp` variable. A `node_disp` perturbation leaves
`domain.local_frames()` untouched, so an orientation frozen at the reference geometry
stays equally frozen in both legs and the check passes even when the gradient is
wrong. That is exactly how the defect this case now gates escaped earlier review. `ex_optimal_thickness_taper` needs the optional optimiser stack
(`pip install 'hermit[opt]'`) and is skipped, not failed, without it.

## Adding a benchmark

Copy `ex_scordelis_lo.py` (curved geometry, distributed load) or
`ex_pinched_cylinder.py` (symmetry planes, point load). The shape is always the same:

1. A module docstring whose **first line is the title** — the docs build turns it into
   a page and requires it. Follow it with the physical setup, the reference value, and
   what makes the case worth having.
2. A `solve_at(level) -> float` returning the single scalar the reference is quoted for.
3. A `CASE = Case(...)` declaring `reference`, `tolerance`, `citation`, `levels` and
   `quick_level`.
4. `if __name__ == "__main__": main(CASE)`.

`quick_level` is the level CI runs, so it should be the cheapest one that is still
honest about the case. Prefer the coarsest level meeting `tolerance`; where no swept
level meets it, keep the most diagnostic affordable level rather than moving the
target (`ex_ss_plate_uniform` does this, and eight of the cases run their finest
swept level). Keep a full sweep under about 30 seconds.

Mesh builders live in `_geometry.py`; add one there rather than inlining a mesh.
Meshes are built **in memory** (the environment has no meshio, and a benchmark's mesh
is a function of its refinement level anyway).

## Every case, in full

```{toctree}
:maxdepth: 1

../_temp/examples/verification/README
../_temp/examples/verification/REFERENCES
```

```{toctree}
:maxdepth: 1
:glob:

../_temp/examples/verification/ex_*
```
