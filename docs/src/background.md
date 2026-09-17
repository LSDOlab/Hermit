# Background

## The Reissner-Mindlin shell

Hermit solves the linear-static Reissner-Mindlin (first-order shear deformation) shell
{cite:p}`reissner1945,mindlin1951`. The kinematics carry a mid-surface displacement
$\mathbf{u}$ and an independent rotation $\boldsymbol{\theta}$ of the shell director,
so transverse shear is retained (unlike Kirchhoff-Love).

Working in the local element frame $(\mathbf{e}_0, \mathbf{e}_1, \mathbf{e}_2)$ built
from the cell normal and Jacobian, the membrane strain $\boldsymbol{\varepsilon}$,
bending curvature $\boldsymbol{\kappa}$ and transverse shear $\boldsymbol{\gamma}$ are
linear in $(\mathbf{u}, \boldsymbol{\theta})$. The stored energy is

$$
\Pi = \tfrac12 \int_\Omega
  \begin{Bmatrix}\boldsymbol{\varepsilon}\\ \boldsymbol{\kappa}\end{Bmatrix}^{\!\top}
  \begin{bmatrix}\mathbf{A} & \mathbf{B}\\ \mathbf{B} & \mathbf{D}\end{bmatrix}
  \begin{Bmatrix}\boldsymbol{\varepsilon}\\ \boldsymbol{\kappa}\end{Bmatrix}
  \, \mathrm{d}\Omega
  + \tfrac12 \int_\Omega \boldsymbol{\gamma}^{\!\top} \mathbf{A}_s \boldsymbol{\gamma}
  \, \mathrm{d}\Omega ,
$$

with the ABD constitutive matrices: $\mathbf{A}$ membrane, $\mathbf{D}$ bending,
$\mathbf{B}$ membrane-bending coupling, $\mathbf{A}_s$ transverse shear. A small
drilling-stabilization term regularises the in-plane rotation.

### Constitutive input

- **Isotropic** (`hm.isotropic`): $\mathbf{A}, \mathbf{B}, \mathbf{D},
  \mathbf{A}_s$ are assembled in plain CSDL from $E$, $\nu$ and the thickness, with a
  $0.833$ shear-correction factor. Each of $E$, $\nu$, thickness, density may be a single
  value or a `Field` on its own FE space; a scalar is broadcast to the mesh.
- **Composite** (`hm.laminate`): classical lamination theory --- the ABD of a ply
  stack, differentiable with respect to ply angles and thicknesses.
- **Pre-computed** (`hm.composite`): ABD fields supplied directly.

The residual always takes ABD, so there is no isotropic/composite branching in the
UFL form.

### Fibre orientation

`hm.laminate` / `hm.composite` produce the ABD in the laminate's own 0-deg
frame. On a real shell that frame is not the element local frame (which follows the
mesh parametrisation). Pass `fiber_direction` (a global vector, or a per-cell field) or
`fiber_angle` (per-cell radians from the element `e0`); Hermit rotates the ABD into
each element's local frame before the solve. A `fiber_angle` `csdl.Variable` is a
differentiable design field. Orientation may use any supported field space.

## Discretization

A mixed space: displacement in CG2 (quadratic Lagrange), rotation in CG1, on quad or
triangle meshes (`ShellDomain(element="CG2CG1")`, the default). Alternatives:
`"CG1CG1"` (linear displacement), and `"CG2CR1"` (Crouzeix--Raviart rotation, a
non-conforming element that resists shear locking on thin shells --- **triangle
meshes only**).

### Constant-normal bending curvature

The bending curvature is a shallow-shell, flat-facet measure: within each cell the
normal is held constant in the bending kinematics. With
$t_{ij}=\mathbf e_i\mathbin{\cdot}\partial_{\mathbf e_j}\boldsymbol\theta$,

$$
\boldsymbol\kappa =
\begin{bmatrix}
-t_{10} & (t_{00}-t_{11})/2\\
(t_{00}-t_{11})/2 & t_{01}
\end{bmatrix},
$$

so curvature comes from $\nabla\boldsymbol\theta$ alone. Under an infinitesimal rigid
motion $\boldsymbol\theta$ is constant and $\boldsymbol\kappa$ vanishes structurally,
in warped quadrilateral cells as well as planar triangles. Stress recovery uses the
same kinematics, $\boldsymbol\varepsilon(\xi)=\boldsymbol\varepsilon_\mathrm{mid}
-\xi\boldsymbol\kappa$, so displacement, strain energy, von Mises stress and the
stress aggregates agree about rigid-body motion.

Constant-normal flat-facet approximations belong to the established shallow-shell
element lineage, of which Belytschko--Tsay is a representative
{cite:p}`belytschko1984`.

#### Limitations

The measure drops a term coupling initial curvature to membrane stretch, so it is
exact only in the shallow-shell limit. The error grows with the
thickness-to-radius ratio and with laminate unsymmetry, and is largest when the shell
is also membrane-loaded. Bending-dominated response is insensitive to the dropped
term: the hypar cantilever and the twisted beam are unaffected by it, and both meet
their references in the verification suite.

In practice: doubly curved geometry does not require triangles to obtain rigid-body
objectivity, and warped quadrilaterals are supported by the same theory. For a
moderately thick shell with strong initial curvature and significant membrane
loading --- especially an unsymmetric laminate --- run a thickness and mesh study, and
do not read the result as a full covariant shell solution. The Reissner--Mindlin
formulation of Schöllhammer and Fries is the relevant direction for that regime
{cite:p}`schoellhammer2019`.

## Loads

Every load builder returns a composable `Loads`; `a + b` merges them, and each term
reaches the residual on its own FE space with no interpolation between terms.

### Distributed over the surface

- `hm.pressure(domain, p)` --- a scalar load along the shell normal, positive along
  $+\mathbf{n}$. The residual uses the live `ufl.CellNormal`, so it is
  shape-differentiable. It requires a consistently wound mesh and raises otherwise:
  on a badly wound import, a uniform pressure would silently become a
  sign-alternating load.
- `hm.traction(domain, t)` --- force per unit area in global components. This never
  touches the normal, so it is the correct choice for a body force such as self
  weight on a curved roof, where the two are very different loads.
- `hm.moment(domain, m)` --- moment per unit area, conjugate to the director rotation.

### On an edge

`hm.edge_traction`, `hm.edge_moment` and `hm.edge_pressure` apply the same three
loads **per unit length** on the exterior facets a `where=` predicate selects --- the
same predicate contract the boundary conditions use:

```python
tip = hm.edge_traction(domain, [0.0, 0.0, -1.0], where=hm.near("x", LENGTH))
lip = hm.edge_moment(domain, [0.0, M, 0.0], where=hm.near("x", LENGTH))
```

The coefficient stays a CSDL variable, so an edge load is differentiable like any
other; the tagged facet measure is structural form data and is fixed at construction.

### Concentrated and direct

`hm.point_load(domain, at=..., force=..., moment=...)` is a *consistent* point load:
the containing cell is located, the state-space basis is evaluated there, and the
result is scattered into the right-hand side --- the weak form of a Dirac delta, so
`at` need not be a mesh vertex. It is differentiable in `force` and `moment`, but not
in `at` or the mesh coordinates. `hm.load_vector(domain, vec)` supplies a generalized
right-hand side directly, in `domain.W` dof order.

## Boundary conditions

Two paths select a region with a coordinate predicate:

- **Penalty** (default): a penalty term over the selected facets constrains the
  masked generalized DOFs. It eliminates no degrees of freedom, and supports several
  independently masked regions in one form.
- **Strong Dirichlet** (`hm.clamp(..., method="strong")`): standard `dirichletbc`
  constraints.

`hm.clamp` fixes all six DOFs, `hm.pin` a named subset of
`("ux", "uy", "uz", "rx", "ry", "rz")`, `hm.symmetry` applies a plane mask, and
`hm.gauge` adds strong point pins to remove residual null-space modes. `hm.near` and
`hm.on_plane` build the predicates; any
`Callable[[ndarray (3, N)], ndarray[bool] (N,)]` works.

### Prescribed (non-zero) values

Every builder takes `value=`, which defaults to zero. It accepts a scalar, a `(6,)`
vector over `(ux, uy, uz, rx, ry, rz)`, or a callable that receives `(N, 3)`
coordinates and returns `(N, 6)`:

```python
# a 1 mm enforced settlement, and a linearly varying enforced rotation
settle = hm.pin(domain, where=hm.near("x", L), dofs=("uz",), value=[0, 0, -1e-3, 0, 0, 0])
twist  = hm.pin(domain, where=hm.near("x", L), dofs=("rx",),
                value=lambda x: np.column_stack([np.zeros((len(x), 3)),
                                                 0.01 * x[:, 1], np.zeros((len(x), 2))]))
```

A prescribed value may **not** be a `csdl.Variable`: it is interpolated once into the
state space rather than carried as a differentiable solve input, so an enforced
displacement cannot be a design variable, while loads and material properties can.
Hermit raises a `TypeError` rather than silently ignoring the dependence. Widening
this later is backward-compatible.

### Merging

`a + b` merges two `BoundaryConditions` on the same domain, and **later terms win on
overlap**: a facet claimed by two regions is governed by the later region's mask
alone, not the union of both masks. That makes `hm.clamp(...) + hm.symmetry(...)`
behave predictably where the two regions touch.

## CSDL coupling and adjoints

`solve` is a CSDL `CustomImplicitOperation`. Because the shell tangent
$\partial R/\partial w$ is symmetric, a single MUMPS factorization serves both the
forward linear solve and the adjoint. The state-based output functions are a
`CustomExplicitOperation` that assembles each scalar form and, in reverse mode,
$\partial(\text{form})/\partial(\text{arg})$.

Reverse-mode sensitivities are available for:

| with respect to | how |
|---|---|
| thickness, density, ABD, $E$, $\nu$ | `ufl.derivative(form, coefficient)` |
| nodal loads / pressure / moments | linear maps in CSDL |
| direct load vector | trivial ($\partial R/\partial b = -I$) |
| **mesh coordinates** | `ufl.derivative(form, SpatialCoordinate)`, scattered to node order |

See {doc}`shape_derivatives` for the mesh-coordinate path.

## Bibliography

```{bibliography} references.bib
```
