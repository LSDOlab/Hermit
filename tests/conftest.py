import pathlib

import numpy as np
import pytest

HERE = pathlib.Path(__file__).parent
MESHES = HERE / "meshes"
DATA = HERE / "data"


def pytest_runtest_teardown(item, nextitem):
    """Free each test's FE-backed custom ops before the next test runs.

    ``csdl_alpha`` (tracked from ``main``) never releases a ``Recorder``: it appends
    itself to ``manager.constructed_recorders`` and never comes off, and the graph
    nodes back-reference it, so every op ever built stays reachable for the life of
    the interpreter. Hermit's ops each pin one or more MPI communicators (a MUMPS
    factorization, a ``ShellPDE``'s function spaces), so a single-process run of the
    whole suite used to hit the MPICH 2048-context limit and ``MPI_Abort`` around
    test ~135 -- which is why CI and ``docs/DEVELOPMENT.md`` had to shard the suite
    into separate ``pytest`` processes.

    ``hermit.fenics._cleanup.release_fe_resources`` destroys those communicator-
    holding resources directly (the retention is multi-anchored, so dropping Python
    references alone does not). Skip it when the test never imported ``hermit`` --
    there is nothing to release, and no reason to pay the import.
    """
    import sys

    if "hermit" in sys.modules:
        from hermit.fenics._cleanup import release_fe_resources

        release_fe_resources()


@pytest.fixture(scope="session")
def plate_mesh():
    import dolfinx
    from mpi4py import MPI

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, str(MESHES / "plate_2x10_quad_4x20.xdmf"), "r") as x:
        return x.read_mesh(name="Grid")


@pytest.fixture(scope="session")
def tri_mesh():
    import dolfinx
    from mpi4py import MPI

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, str(MESHES / "plate_2x10_tri_4x20.xdmf"), "r") as x:
        return x.read_mesh(name="Grid")


@pytest.fixture(scope="session")
def cantilever_ref():
    return dict(np.load(DATA / "rmshell_cantilever.npz"))


def assert_matches_legacy(got, want, *, rtol=1e-7, name="disp_solid"):
    """Compare a *solved* quantity against the stored reference.

    Deliberately not ``np.array_equal``. When the reference path was recomputed live in
    the same process, both sides shared every rounding decision and bit-equality was a
    fair, very sharp gate. The reference is now a stored array
    (``tests/data/legacy_api_reference.npz``), so bit-equality would instead assert that
    a MUMPS solve is reproducible across whatever BLAS / PETSc / DOLFINx build the run
    happens to have -- which is not a real property. CI failed on exactly the
    ``array_equal`` gates and on nothing else.

    Scale-relative rather than element-wise: ``disp_solid`` interleaves displacements
    and rotations spanning orders of magnitude, so an element-wise rtol punishes the
    near-zero entries for no reason.

    The gate keeps essentially all of its power. What it exists to catch is a boundary
    condition silently dropped or misapplied, and that moves ``max|w|`` from 8.7e-3 to
    1.7e8 -- about ten orders of magnitude past this tolerance. Measured cross-run drift
    on this machine is ~1e-13 relative (BLAS threading, see
    ``test_point_load_at_vertex_matches_legacy_nodal_forces``).

    ``rtol`` is 1e-7, not the 1e-9 it was while the bending measure was unchanged.
    The constant-normal curvature (issue #7) is *analytically identical* to the old
    measure on these fixtures -- every one is a flat plate, where the cell normal is
    constant and ``grad(E2)`` vanishes -- but it evaluates a different expression
    tree, and the beta=1e15 penalty system amplifies that into a ~7e-9 relative
    difference. Round-off from a deliberate formulation change, not drift. 1e-7 still
    sits about ten orders below the dropped-BC signature the gate exists to catch.
    """
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"{name}: shape {got.shape} != stored {want.shape}"
    scale = float(np.abs(want).max())
    err = float(np.abs(got - want).max())
    assert err <= rtol * scale, (
        f"{name}: max|diff| = {err:.6e}, tolerance = {rtol:.0e} * scale {scale:.6e} "
        f"= {rtol * scale:.6e}")


@pytest.fixture(scope="session")
def legacy_ref():
    return dict(np.load(DATA / "legacy_api_reference.npz"))


@pytest.fixture
def recorder():
    import csdl_alpha as csdl

    rec = csdl.Recorder(inline=True)
    rec.start()
    yield rec
    rec.stop()


def clamped_at_x0(x):
    return np.less(x[0], 1e-12)
