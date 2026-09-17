"""Shape derivative of failure index for direction-oriented laminates.

This benchmark verifies that the shape derivative of the Tsai-Wu failure index
is complete when the laminate orientation is specified via a global `fiber_direction`.

**TRAP AVOIDED**: The finite difference check perturbs the MESH COORDINATES directly
and rebuilding the mesh, rather than perturbing `node_disp`. A `node_disp` perturbation
fails to expose the missing geometry dependence in the numpy-computed `fiber_direction`
because `domain.local_frames()` is built from `domain.mesh` (which `node_disp` never
modifies). This trap was the reason the bug survived.
"""

import pathlib
import sys
import numpy as np
import csdl_alpha as csdl
import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import rect_plate
from _harness import Case, main

LENGTH = 10.0
WIDTH = 2.0
PRESSURE = 1.0e3

def _rebuild_mesh(nx, ny, shape_override=None):
    mesh = rect_plate(LENGTH, WIDTH, nx=nx, ny=ny, cell="quad")
    if shape_override is not None:
        mesh.geometry.x[:] = shape_override
    else:
        # Cambered plate: z = 0.5 * (x / LENGTH)**2
        mesh.geometry.x[:, 2] = 0.5 * (mesh.geometry.x[:, 0] / LENGTH)**2
    return mesh

def _evaluate(mesh, compute_gradient=False, orientation_kind="direction"):
    rec = csdl.Recorder(inline=True)
    rec.start()
    
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    
    import caddee_materials
    ply = caddee_materials.TransverseMaterial(name="carbon", density=1.6e3, EA=135e9, ET=10e9, vA=0.3, vT=0.3, GA=5e9)
    ply.set_strength(F1t=1.5e9, F1c=1.2e9, F2t=50e6, F2c=250e6, F12=70e6, F23=50e6)
    layup = hm.Layup(ply, np.radians([0.0, 90.0, 0.0]), np.array([0.02 / 3] * 3))
    
    if orientation_kind == "direction":
        orientation = hm.fiber_direction(domain, [np.cos(0.4), np.sin(0.4), 0.0])
    else:
        # For 'angle' branch verification
        orientation = hm.fiber_angle(domain, 0.4)
        
    material = hm.laminate(domain, layup=layup, density=1.6e3, orientation=orientation)
    
    # Boundary and loads
    clamped = hm.clamp(domain, where=lambda x: np.isclose(x[0], 0.0))
    load = hm.pressure(domain, PRESSURE)
    
    # Note: we need mesh_nodes as a differentiable input to get the adjoint geometry derivative
    nd = csdl.Variable(value=mesh.geometry.x.copy(), name="mesh_nodes")
    state = hm.solve(domain, material, load, clamped, geometry=hm.geometry(domain, nodes=nd))
    
    fi = hm.failure_index(state, rho=100.0)
    
    val = float(np.asarray(fi.value).ravel()[0])
    grad = None
    if compute_gradient:
        sim = csdl.experimental.PySimulator(rec)
        grad = np.asarray(sim.compute_totals([fi], [nd])[fi, nd])
    rec.stop()
    return val, grad

def solve_at(level):
    nx = 4 * level
    ny = level
    
    # Base configuration
    mesh = _rebuild_mesh(nx, ny)
    base_coords = mesh.geometry.x.copy()
    
    # Adjoint gradient
    fi_base, grad_base = _evaluate(mesh, compute_gradient=True, orientation_kind="direction")
    grad_base = grad_base.reshape(-1, 3)
    
    # Central finite differences by rebuilding the mesh
    np.random.seed(42)
    perturbation = np.random.randn(*base_coords.shape)
    # Hold the clamped edge fixed
    clamped_mask = np.isclose(base_coords[:, 0], 0.0)
    perturbation[clamped_mask] = 0.0
    # Normalize perturbation to unit length
    perturbation /= np.linalg.norm(perturbation)
    
    directional_derivative_adjoint = np.sum(grad_base * perturbation)
    
    step = 1e-6
    mesh_plus = _rebuild_mesh(nx, ny, shape_override=base_coords + step * perturbation)
    fi_plus, _ = _evaluate(mesh_plus, compute_gradient=False, orientation_kind="direction")
    
    mesh_minus = _rebuild_mesh(nx, ny, shape_override=base_coords - step * perturbation)
    fi_minus, _ = _evaluate(mesh_minus, compute_gradient=False, orientation_kind="direction")
    
    fd = (fi_plus - fi_minus) / (2 * step)
    
    rel_error = abs(directional_derivative_adjoint - fd) / max(abs(fd), 1e-14)
    print(f"Level {level}: FI = {fi_base:.6e}, adjoint dir_deriv = {directional_derivative_adjoint:.6e}, FD = {fd:.6e}, rel_error = {rel_error:.3e}")
    
    return 1.0 - rel_error

CASE = Case(
    name="Shape derivative of failure index with fiber direction",
    quantity="1 - relative error between adjoint and FD",
    reference=1.0,
    tolerance=5e-3,
    citation="Tsai-Wu failure index with `fiber_direction` geometry sensitivity",
    levels=(2, 3),
    quick_level=2,
    solve=solve_at,
    monotone=False,
    notes="FD perturbs mesh coordinates directly and rebuilds the mesh to catch severed geometry dependency in orientations",
)

if __name__ == "__main__":
    main(CASE)
