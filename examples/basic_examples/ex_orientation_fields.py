"""Fibre orientation and strain-field frames (with plots)

A carbon/epoxy cantilever plate on a **triangle** mesh -- so the per-element in-plane
frame alternates cell to cell -- with a *curvilinear* fibre path: the fibres fan from
0 deg at the clamped root to ``THETA_TIP`` at the free tip, set through a per-cell
``fiber_angle`` field. The script then

* reads the bending ``curvature`` back as a ``Field`` and re-expresses the same
  component (``kappa_xx``) with ``to_frame`` -- the raw element frame (a per-cell
  artefact -> patchy), the local fibre frame, and the global Cartesian frame
  (``to_global()`` -> the smooth physical field),
* contrasts that DG (per-cell, discontinuous) field with a smooth CG-projected nodal
  field (``hm.strain_fields(..., space=("Lagrange", 1))`` -> a global-frame ``Field``),
* sweeps a *uniform* fibre angle and plots compliance and the Tsai-Wu failure index.

Writes ``ex_orientation_fields.png`` next to this file (needs ``matplotlib``).

    conda activate hermit
    python examples/basic_examples/ex_orientation_fields.py
"""

import pathlib

import numpy as np
import csdl_alpha as csdl

import hermit as hm
from hermit import Layup
from caddee_materials import TransverseMaterial

MESH = pathlib.Path(__file__).parents[2] / "tests" / "meshes" / "plate_2x10_tri_4x20.xdmf"
OUT_PNG = pathlib.Path(__file__).with_name("ex_orientation_fields.png")

LENGTH, WIDTH = 10.0, 2.0
THETA_TIP = 60.0            # curvilinear fibre angle at the tip (deg)
TOTAL_H, DENSITY = 0.02, 1.6e3
PRESSURE_Z = 1.0e3


def clamped_at_x0(x):
    return np.less(x[0], 1e-12)


def main():
    mesh = hm.read_mesh(MESH)

    rec = csdl.Recorder(inline=True)
    rec.start()

    domain = hm.ShellDomain(mesh, element="CG2CG1")
    ud = TransverseMaterial(name="ud", EA=138e9, ET=10e9, vA=0.34, vT=0.67, GA=7e9, density=DENSITY)
    ud.set_strength(F1t=1500e6, F1c=1200e6, F2t=50e6, F2c=200e6, F12=70e6, F23=40e6)
    layup = Layup(ud, csdl.Variable(value=np.zeros(3)), np.full(3, TOTAL_H / 3), num_plies=3)
    loads = hm.pressure(domain, PRESSURE_Z)
    bcs = hm.clamp(domain, where=clamped_at_x0)

    # -- per-cell fibre frame (file cell order) ----------------------------
    frames = domain.local_frames()                                # (n_cells, 3, 3), file order
    mids = domain.cell_centroids
    theta = np.radians(THETA_TIP) * mids[:, 0] / LENGTH            # 0 at root -> THETA_TIP at tip
    fibre_dir = (np.cos(theta)[:, None] * frames[:, 0]
                 + np.sin(theta)[:, None] * frames[:, 1])          # (n_cells, 3), file order

    material = hm.laminate(domain, layup=layup, density=DENSITY,
                           orientation=hm.fiber_angle(domain, csdl.Variable(value=theta)))

    # -- solve once, post-process twice (DG default, then smooth CG) --------
    state = hm.solve(domain, material, loads, bcs)
    compliance, failure_index = hm.compliance(state), hm.failure_index(state)
    _, kap_dg, _ = hm.strain_fields(state)                         # ("DG", 2), element frame
    _, kap_cg, _ = hm.strain_fields(state, space=("Lagrange", 1)) # smooth nodal, global frame

    print(f"  triangle mesh: {domain.n_cells} cells; per-cell fibre 0 deg (root) -> {THETA_TIP:.0f} deg (tip)")
    print(f"  compliance                        : {float(np.ravel(compliance.value)[0]):.4e}")
    print(f"  Tsai-Wu failure index             : {float(np.ravel(failure_index.value)[0]):.4e}")
    print(f"  curvature DG field (element frame) : {kap_dg.values.shape}  kind={kap_dg.kind}")
    print(f"  curvature CG field (global frame)  : "
          f"{np.asarray(kap_cg.coeffs.value).reshape(-1, 6).shape}  kind={kap_cg.kind}")

    # kappa_xx in three frames (DG, per-cell, user cell order)
    kxx_elem = kap_dg.values[:, 0]
    kxx_fibre = kap_dg.to_frame(fibre_dir).values[:, 0]            # resolved along the local fibre
    kxx_glob = kap_dg.to_global().values[:, 0]                    # smooth physical field
    kxx_cg = np.asarray(kap_cg.coeffs.value).reshape(-1, 6)[:, 0]  # global frame, CG1 nodes

    # -- uniform fibre-angle sweep ----------------------------------------
    sweep_deg = np.arange(0, 91, 15)
    comp, fi = [], []
    for deg in sweep_deg:
        d = [np.cos(np.radians(deg)), np.sin(np.radians(deg)), 0.0]
        m = hm.laminate(domain, layup=layup, density=DENSITY,
                        orientation=hm.fiber_direction(domain, d))
        s = hm.solve(domain, m, loads, bcs)
        comp.append(float(np.ravel(hm.compliance(s).value)[0]))
        fi.append(float(np.ravel(hm.failure_index(s).value)[0]))

    rec.stop()

    # -- figure ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        from matplotlib.tri import Triangulation
    except ImportError:
        print("\n  matplotlib not available -- skipping the figure")
        return

    verts = mesh.geometry.x[:, :2]
    try:
        cell_nodes = np.asarray(mesh.geometry.dofmaps[0])
    except (AttributeError, IndexError):
        cell_nodes = np.asarray(mesh.geometry.dofmap)
    tri_cells = Triangulation(verts[:, 0], verts[:, 1], cell_nodes[domain.reverse_cell_idx])
    cg_xy = domain.dof_coords(kap_cg.space)[:, :2]
    tri_nodes = Triangulation(cg_xy[:, 0], cg_xy[:, 1])
    allk = np.concatenate([kxx_elem, kxx_fibre, kxx_glob, kxx_cg])
    norm = TwoSlopeNorm(vcenter=0.0, vmin=min(allk.min(), -1e-9), vmax=max(allk.max(), 1e-9))

    fig, axes = plt.subplots(2, 3, figsize=(15, 6), constrained_layout=True)
    kw = dict(cmap="RdBu_r", norm=norm)
    kpanels = [
        (axes[0, 1], kxx_elem, False, r"$\kappa_{xx}$ -- raw element frame (DG2, per-cell)"),
        (axes[0, 2], kxx_fibre, False, r"$\kappa$ along local fibre -- to_frame(fibre)"),
        (axes[1, 0], kxx_glob, False, r"$\kappa_{xx}$ -- global frame -- to_global()"),
        (axes[1, 1], kxx_cg, True, r"$\kappa_{xx}$ -- global frame, smooth (CG1 project)"),
    ]
    for ax, vals, gouraud, title in kpanels:
        m = (ax.tripcolor(tri_nodes, vals, shading="gouraud", **kw) if gouraud
             else ax.tripcolor(tri_cells, facecolors=vals, **kw))
        ax.set_xlim(0, LENGTH); ax.set_ylim(0, WIDTH)
        ax.set_title(title, fontsize=10)
        fig.colorbar(m, ax=ax, fraction=0.03, pad=0.02)

    ax = axes[0, 0]
    ax.triplot(tri_cells, color="0.88", lw=0.3)
    gx, gy = np.meshgrid(np.linspace(0.5, LENGTH - 0.5, 20), np.linspace(0.3, WIDTH - 0.3, 4))
    gth = np.radians(THETA_TIP) * gx / LENGTH
    q = ax.quiver(gx, gy, np.cos(gth), np.sin(gth), np.degrees(gth),
                  cmap="viridis", pivot="mid", scale=22, width=0.006)
    ax.set_xlim(0, LENGTH); ax.set_ylim(0, WIDTH)
    ax.set_title(f"curvilinear fibre_angle  (0 deg root -> {THETA_TIP:.0f} deg tip)", fontsize=10)
    fig.colorbar(q, ax=ax, fraction=0.03, pad=0.02, label="deg")

    ax = axes[1, 2]
    ax.plot(sweep_deg, comp, "o-", color="C0")
    ax.set_xlabel("uniform fibre angle from beam axis (deg)")
    ax.set_ylabel("compliance", color="C0"); ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    ax2.plot(sweep_deg, fi, "s--", color="C3")
    ax2.set_ylabel("Tsai-Wu failure index", color="C3"); ax2.tick_params(axis="y", labelcolor="C3")
    ax.set_title("uniform fibre-angle sweep", fontsize=10)

    fig.savefig(OUT_PNG, dpi=130)
    rel = OUT_PNG.relative_to(pathlib.Path.cwd()) if OUT_PNG.is_relative_to(pathlib.Path.cwd()) else OUT_PNG
    print(f"\n  wrote {rel}")


if __name__ == "__main__":
    main()
