"""Thin PETSc/UFL assembly helpers for DOLFINx (0.9 / 0.11).

Slim port of the pieces of ``femo_alpha/fea/utils_dolfinx.py`` that Hermit's custom
operations actually need. No CSDL here.
"""

import numpy as np
import ufl
from dolfinx.fem import form, assemble_scalar as _assemble_scalar, Function
from dolfinx.fem.petsc import assemble_matrix as _assemble_matrix, assemble_vector as _assemble_vector
from petsc4py import PETSc


def get_array(f: Function) -> np.ndarray:
    """Local array of a Function (owned + ghost)."""
    return f.x.array


def set_array(f: Function, values) -> None:
    """Set a Function from a flat array and sync ghosts."""
    f.x.array[:] = np.asarray(values, dtype=f.x.array.dtype).reshape(-1)
    f.x.scatter_forward()


def assemble_scalar(f) -> float:
    """Assemble a scalar UFL form to a float."""
    return float(_assemble_scalar(form(f)))


def assemble_vector(f) -> np.ndarray:
    """Assemble a 1-form to a numpy array."""
    v = _assemble_vector(form(f))
    v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return v.getArray().copy()


def assemble_matrix(a, bcs=()) -> PETSc.Mat:
    """Assemble a 2-form to an assembled PETSc matrix."""
    A = _assemble_matrix(form(a), bcs=list(bcs))
    A.assemble()
    return A


def derivative(f, u, du=None):
    """d f / d u as a UFL form (or w.r.t. a coefficient / SpatialCoordinate)."""
    return ufl.derivative(f, u) if du is None else ufl.derivative(f, u, du)


def transpose(A: PETSc.Mat) -> PETSc.Mat:
    return A.transpose(PETSc.Mat())


def matvec(A: PETSc.Mat, x: Function) -> np.ndarray:
    """A @ x  (x a Function, result a numpy array of A's row layout)."""
    y = A.createVecLeft()
    A.mult(x.x.petsc_vec, y)
    y.assemble()
    return y.getArray().copy()


def rmatvec(A: PETSc.Mat, x: Function) -> np.ndarray:
    """A.T @ x."""
    y = A.createVecRight()
    A.multTranspose(x.x.petsc_vec, y)
    y.assemble()
    return y.getArray().copy()


def ksp_mumps(A: PETSc.Mat) -> PETSc.KSP:
    """A preonly + LU(MUMPS) KSP for the (already assembled) operator ``A``."""
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    ksp.setUp()
    return ksp


def project(expr, target: Function, lump_mass: bool = False) -> None:
    """L2-project a UFL expression onto ``target``'s space (used for field outputs)."""
    V = target.function_space
    v, Pv = ufl.TestFunction(V), ufl.TrialFunction(V)
    L = ufl.inner(expr, v) * ufl.dx
    if lump_mass:
        a = ufl.inner(1.0, v) * ufl.dx
        A = _assemble_vector(form(a))
        b = _assemble_vector(form(L))
        target.x.petsc_vec.pointwiseDivide(b, A)
    else:
        A = assemble_matrix(ufl.inner(Pv, v) * ufl.dx)
        b = _assemble_vector(form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        ksp_mumps(A).solve(b, target.x.petsc_vec)
    target.x.scatter_forward()


def owning_cell(dofmap_list, n_scalar_dofs):
    """``scalar dof -> the cell that writes it last`` in ascending cell order.

    ``Function.interpolate`` scatters a point evaluation per cell, so a shared
    (continuous) target dof is written once per adjacent cell and the last writer
    wins. The interpolation *Jacobian* must therefore carry only the owning cell's
    row -- taking the union of every adjacent cell's contribution (which is what a
    plain COO scatter does, since each cell contributes on its own source columns)
    over-counts it. Deduplicating (row, col) pairs is not enough: the columns
    mostly differ, so the duplicates never collide.

    Same construction as ``hermit._field._Tabulation.dof_cell``, and a no-op on DG
    (every dof has exactly one cell).
    """
    owner = np.empty(n_scalar_dofs, dtype=np.int64)
    owner[dofmap_list.ravel()] = np.repeat(np.arange(dofmap_list.shape[0]),
                                           dofmap_list.shape[1])
    return owner


def owned_entry_mask(dofmap_list, n_scalar_dofs):
    """``(n_cells, n_local_nodes)`` bool -- True where the cell owns that dof."""
    owner = owning_cell(dofmap_list, n_scalar_dofs)
    return owner[dofmap_list] == np.arange(dofmap_list.shape[0])[:, None]
