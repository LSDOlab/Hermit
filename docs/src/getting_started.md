# Getting started

## Installation

Hermit, [FEniCSx](https://fenicsproject.org/) (DOLFINx 0.9 or 0.11),
[CSDL Alpha](https://github.com/LSDOlab/CSDL_alpha), and `caddee_materials` are
available from the `HgXe` and `conda-forge` channels. CSDL Alpha is currently
promoted through the `HgXe/label/test` label:

```sh
conda create -n hermit -c HgXe/label/test -c HgXe -c conda-forge hermit
conda activate hermit
```

Hermit's forms are real-valued, so its package selects the **real-scalar** PETSc
build. The `HgXe` channel supplies the CSDL lab packages and `conda-forge` supplies
FEniCSx and its compiled dependencies.

For editable development, create the base environment explicitly and install the lab
packages from their repositories:

```sh
conda create -n hermit-dev -c conda-forge python=3.12 fenics-dolfinx=0.11 \
    mpich 'petsc=*=real*' petsc4py numpy scipy sympy h5py rustworkx networkx pydot
conda activate hermit-dev
pip install git+https://github.com/LSDOlab/caddee_materials.git
pip install --force-reinstall --no-deps \
    git+https://github.com/LSDOlab/CSDL_alpha.git@main
pip install -e .
```

Optional extras: `.[examples]` for the plotting examples and `.[docs]` for the
documentation toolchain.

The optimizer stack --- needed by `ex_thickness_opt.py` and
`ex_optimal_thickness_taper.py` --- is LSDOlab's `modopt`, which shares a name with an
unrelated package on PyPI. Install it from git:

```sh
pip install git+https://github.com/LSDOlab/modopt.git pyslsqp
```

Hermit runs **serially**: `ShellDomain` requires a mesh whose vertex numbering is a
permutation of the file order, which an MPI-partitioned mesh is not.

## A minimal example

```python
import csdl_alpha as csdl
import hermit as hm

mesh = hm.read_mesh("plate.xdmf")
domain = hm.ShellDomain(mesh, element="CG2CG1")
bcs = hm.clamp(domain, where=hm.near("x", 0.0))

recorder = csdl.Recorder(inline=True); recorder.start()
material = hm.isotropic(domain, E=4.32e8, nu=0.0, thickness=0.2, density=1.0)
state = hm.solve(domain, material, hm.pressure(domain, 2.0), bcs)
print(hm.compliance(state).value, hm.mass(state).value)
recorder.stop()
```

Arrays passed to simple builders are converted to fields by length; use
`hm.from_nodal`, `hm.from_cells`, or `hm.from_function` when the intended input space
matters. See {doc}`tutorials` and {doc}`examples` for complete scripts.

## Running the tests

```sh
python -m pytest
```

The suite checks the forward quantities and thickness adjoints Hermit shares with
`femo_alpha` against a captured RMShell reference, and runs each of the
{doc}`examples/verification` benchmarks at a cheap refinement level.

Any individual benchmark also runs on its own, and prints its convergence table:

```sh
python examples/verification/ex_scordelis_lo.py
```
