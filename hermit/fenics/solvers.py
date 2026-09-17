"""Linear / Newton solves of the shell residual (DOLFINx 0.9 / 0.11, MUMPS).

Direct solve is a port of ``custom_solve_direct`` from femo dev_coupling: assemble the
tangent and residual, apply BC lifting, solve one Newton step (exact for a linear
problem). ``direct_residual_inputs`` lets the caller add extra RHS terms (the direct
generalized load vector) without going through UFL assembly.
"""

import numpy as np
import ufl
from dolfinx.fem import form
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from petsc4py import PETSc

from .assembly import ksp_mumps


def _apply_direct_residual_inputs(vec: PETSc.Vec, direct_inputs):
    """vec += sum(sign * value) for each {'value','sign'} in direct_inputs."""
    if not direct_inputs:
        return
    with vec.localForm() as loc:
        arr = loc.getArray()
        for item in direct_inputs:
            arr[:] += item["sign"] * np.asarray(item["value"]).reshape(-1)


def linear_solve(residual, w, bcs=(), direct_residual_inputs=None):
    """Solve F(w) = 0 for a linear F: one Newton step from the current w."""
    bcs = list(bcs)
    du = ufl.TrialFunction(w.function_space)
    J = ufl.derivative(residual, w, du)

    A = assemble_matrix(form(J), bcs=bcs)
    A.assemble()

    r = assemble_vector(form(residual))
    _apply_direct_residual_inputs(r, direct_residual_inputs)
    apply_lifting(r, [form(J)], [bcs], x0=[w.x.petsc_vec])
    r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(r, bcs, w.x.petsc_vec)
    r.scale(-1.0)

    dw = w.x.petsc_vec.copy()
    dw.set(0.0)
    ksp_mumps(A).solve(r, dw)

    w.x.petsc_vec.axpy(1.0, dw)
    w.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    w.x.scatter_forward()
