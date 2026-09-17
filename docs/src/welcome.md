# Hermit

**Reissner-Mindlin shell analysis in FEniCSx, connected to CSDL for gradient-based design.**

Hermit is a linear-static shell finite-element code built on
[FEniCSx](https://fenicsproject.org/) (DOLFINx 0.9 or 0.11). It wraps the FE solve and its
output functionals as [CSDL](https://github.com/LSDOlab/CSDL_alpha) custom operations,
so shell compliance, mass, centre of gravity, stress and nodal fields are all
differentiable with respect to thickness, composite ply layup, applied loads, and
the mesh coordinates. The mechanics are based on `RMShell` model in
[`femo_alpha`](https://github.com/LSDOlab/femo_alpha).

```python
import hermit as hm

domain   = hm.ShellDomain(mesh)
material = hm.isotropic(domain, E=E, nu=nu, thickness=thickness, density=density)
state    = hm.solve(domain, material, hm.pressure(domain, p),
                    hm.clamp(domain, where=hm.near("x", 0.0)))
compliance, mass = hm.compliance(state), hm.mass(state)
# compliance and mass are csdl.Variables
```

Start with {doc}`src/getting_started`, then the {doc}`src/tutorials`.

```{toctree}
:maxdepth: 1
:hidden:

src/getting_started
src/background
src/architecture
src/shape_derivatives
src/tutorials
src/examples
src/api
```
