"""Composite Tsai-Wu failure post-processing: ``hm.failure_index`` /
``hm.failure_field`` off a solved ``ShellState``."""

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit
import hermit.bcs as hbc
import hermit.loads as hld
import hermit.material as hmat
import hermit.outputs as out
from hermit.domain import ShellDomain
from hermit._solve import solve
from conftest import clamped_at_x0


def _ud():
    from caddee_materials import TransverseMaterial

    m = TransverseMaterial(name="ud", EA=138e9, ET=10e9, vA=0.34, vT=0.4, GA=7e9, density=1.6e3)
    m.set_strength(F1t=1500e6, F1c=1200e6, F2t=50e6, F2c=200e6, F12=70e6, F23=40e6)
    return m


def _layup(angles_deg, total_h, heights_scale=None):
    from hermit._laminate import Layup

    n = len(angles_deg)
    angles = csdl.Variable(value=np.radians(np.asarray(angles_deg, float)), name="ply_angles")
    base = np.full(n, total_h / n)
    if heights_scale is None:
        heights = base
    else:
        heights = heights_scale * csdl.Variable(value=base)
    return Layup(_ud(), angles, heights, num_plies=n), angles


def _run(plate_mesh, layup, nn, pz, rho=1.6e3, node_disp=None):
    dom = ShellDomain(plate_mesh, element="CG2CG1")
    mat = hmat.laminate(dom, layup=layup, density=rho * np.ones(nn),
                        constitutive_space=("Lagrange", 1))
    geom = None if node_disp is None else hermit.geometry(dom, node_disp=node_disp)
    return solve(dom, mat, hld.pressure(dom, pz), hbc.clamp(dom, where=clamped_at_x0),
                 geometry=geom)


# --- 1. the vectorised index matches LamAD's per-point version --------------

def test_vectorized_index_matches_lamad():
    from hermit._laminate.failure import tsai_wu_failure_index
    from hermit._laminate.failure import _tsai_wu_index, ply_properties

    rng = np.random.default_rng(0)
    eps = rng.normal(scale=1e-3, size=(7, 3))
    shear = rng.normal(scale=1e-3, size=(7, 2))
    angle_val = np.radians(37.0)
    mat = _ud()

    rec = csdl.Recorder(inline=True); rec.start()
    p = ply_properties(mat)
    ang = csdl.Variable(value=angle_val)
    got = _tsai_wu_index(csdl.Variable(value=eps), csdl.Variable(value=shear), ang, p).value

    # spelled out rather than passing ``p`` straight through: this is the list of keys
    # the vendored per-point path is expected to consume, so a key that ply_properties
    # stops returning (as it once did for the interlaminar G13/S13 pair -- the tau_13
    # term was silently dead) shows up here as a KeyError, not as a quiet zero.
    lam_props = dict(E1=p["E1"], E2=p["E2"], v12=p["v12"], G12=p["G12"],
                     G13=p["G13"], G23=p["G23"],
                     Xt=p["Xt"], Xc=p["Xc"], Yt=p["Yt"], Yc=p["Yc"],
                     S12=p["S12"], S13=p["S13"], S23=p["S23"])
    want = np.array([
        np.ravel(tsai_wu_failure_index(
            csdl.Variable(value=eps[i]), csdl.Variable(value=shear[i]),
            csdl.Variable(value=angle_val), lam_props).value)[0]
        for i in range(7)
    ])
    rec.stop()
    assert np.allclose(got.ravel(), want, rtol=1e-10, atol=1e-12)


# --- 2. forward behaviour --------------------------------------------------

@pytest.mark.parametrize("sampling", ["average", "midpoint"])
def test_failure_index_forward(plate_mesh, cantilever_ref, sampling, recorder):
    nn = int(cantilever_ref["n_nodes"])
    layup, _ = _layup([0, 90, 0], 0.2)
    state = _run(plate_mesh, layup, nn, pz=2.0)

    field = np.asarray(out.failure_field(state, sampling=sampling).values)
    fi = float(np.ravel(out.failure_index(state, sampling=sampling).value)[0])
    nel = plate_mesh.topology.index_map(2).size_local
    assert field.shape == (nel, 6)                     # 3 plies x 2 faces
    assert fi >= field.max() - 1e-9                    # KS is an upper bound
    assert fi <= field.max() + np.log(field.size) / 100.0 + 1e-9


def test_average_and_midpoint_differ_on_warped_mesh(plate_mesh, cantilever_ref, recorder):
    """On a curved quad mesh the two samplings are genuinely different discretizations."""
    nn = int(cantilever_ref["n_nodes"])
    nd = np.zeros((nn, 3))
    nd[:, 2] = 0.4 * (plate_mesh.geometry.x[:, 0] / 10.0) ** 2
    layup, _ = _layup([20, -20, 70], 0.02)
    state = _run(plate_mesh, layup, nn, pz=5e3, node_disp=nd)
    fa_ = float(np.ravel(out.failure_index(state, sampling="average").value)[0])
    fm = float(np.ravel(out.failure_index(state, sampling="midpoint").value)[0])
    print(f"\naverage FI={fa_:.4e}  midpoint FI={fm:.4e}")
    assert abs(fa_ - fm) / fa_ > 1e-3


def test_thin_laminate_fails_thick_is_safe(plate_mesh, cantilever_ref, recorder):
    nn = int(cantilever_ref["n_nodes"])
    thin, _ = _layup([0, 90, 0], 6e-3)
    thick, _ = _layup([0, 90, 0], 6e-2)
    s_thin = _run(plate_mesh, thin, nn, pz=2.0e3)
    s_thick = _run(plate_mesh, thick, nn, pz=2.0e3)
    fi_thin = float(np.ravel(out.failure_index(s_thin).value)[0])
    fi_thick = float(np.ravel(out.failure_index(s_thick).value)[0])
    print(f"\nFI thin={fi_thin:.3e}  thick={fi_thick:.3e}")
    assert fi_thin > 1.0
    assert fi_thick < fi_thin


# --- 3. derivatives ------------------------------------------------------

def test_failure_index_adjoint_wrt_ply_angles(plate_mesh, cantilever_ref):
    nn = int(cantilever_ref["n_nodes"])
    ang0 = [12.0, 74.0, 33.0]

    def run(delta, grad):
        rec = csdl.Recorder(inline=True); rec.start()
        layup, angles = _layup(ang0, 0.02)
        if delta is not None:
            angles.value[:] = angles.value + delta
        o = out.failure_index(_run(plate_mesh, layup, nn, pz=5.0e3))
        val = float(np.ravel(o.value)[0])
        g = None
        if grad:
            g = np.asarray(csdl.experimental.PySimulator(rec).compute_totals([o], [angles])[o, angles]).ravel()
        rec.stop()
        return val, g

    _, g = run(None, True)
    assert np.all(np.isfinite(g)) and np.linalg.norm(g) > 0
    V = np.array([1.0, -0.6, 0.4]); V /= np.linalg.norm(V)
    # 1e-3, not 1e-4. A central difference amplifies each solve's round-off by
    # 1/(2*step), so too small a step is noise-dominated, not accurate. Measured here
    # (rel error vs the adjoint): 3.7e-4 at 1e-4, 7.4e-5 at 3e-4, 4.2e-5 at 1e-3,
    # 3.3e-5 at 3e-3 -- i.e. it bottoms out near 3e-5 once truncation takes over.
    # At 1e-4 the gate had only a 1.3x margin and CI's FEniCSx 0.11 build tipped it
    # over (6.2e-4); at 1e-3 the margin is ~12x. The adjoint itself is unchanged to
    # seven digits either way, so this is the FD estimate being sharpened, not the
    # gate being loosened -- the tolerance stays at 5e-4.
    step = 1e-3
    vp, _ = run(step * V, False)
    vm, _ = run(-step * V, False)
    dd_fd = (vp - vm) / (2 * step)
    dd_an = float(g @ V)
    rel = abs(dd_an - dd_fd) / abs(dd_fd)
    print(f"\nd(FI)/d(angles).V  analytic={dd_an:.6e}  fd={dd_fd:.6e}  rel={rel:.2e}")
    assert rel < 5e-4


def test_failure_index_adjoint_wrt_thickness(plate_mesh, cantilever_ref):
    nn = int(cantilever_ref["n_nodes"])

    def run(scale, grad):
        rec = csdl.Recorder(inline=True); rec.start()
        s = csdl.Variable(value=float(scale), name="s")
        layup, _ = _layup([15, -15, 80], 0.02, heights_scale=s)
        o = out.failure_index(_run(plate_mesh, layup, nn, pz=5.0e3))
        val = float(np.ravel(o.value)[0])
        g = None
        if grad:
            g = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([o], [s])[o, s])[0])
        rec.stop()
        return val, g

    _, ana = run(1.0, True)
    d = 1e-4
    vp, _ = run(1.0 + d, False)
    vm, _ = run(1.0 - d, False)
    cd = (vp - vm) / (2 * d)
    rel = abs(ana - cd) / abs(cd)
    print(f"\nd(FI)/d(thickness scale)  analytic={ana:.6e}  cd={cd:.6e}  rel={rel:.2e}")
    assert cd < 0                       # thicker laminate -> lower failure index
    assert rel < 2e-3


# --- 4. guards ---------------------------------------------------------

def test_failure_needs_composite(plate_mesh, cantilever_ref, recorder):
    nn = int(cantilever_ref["n_nodes"])
    E, nu, rd, h = (float(cantilever_ref[k]) for k in ("E_val", "nu_val", "rho_val", "h_val"))
    dom = ShellDomain(plate_mesh)
    mat = hmat.isotropic(dom, E=E*np.ones(nn), nu=nu*np.ones(nn),
                         thickness=h*np.ones(nn), density=rd*np.ones(nn),
                         constitutive_space=("Lagrange", 1))
    state = solve(dom, mat, hld.pressure(dom, 2.0), hbc.clamp(dom, where=clamped_at_x0))
    with pytest.raises(ValueError, match="composite"):
        out.failure_index(state)


@pytest.mark.parametrize("sampling", ["average", "midpoint"])
def test_failure_index_shape_derivative(plate_mesh, cantilever_ref, sampling):
    """d(failure_index)/d(node_disp) vs central difference, smooth warp + smooth direction."""
    nn = int(cantilever_ref["n_nodes"])
    gx = plate_mesh.geometry.x.copy()
    L, W = 10.0, 2.0
    bend = np.zeros((nn, 3))
    bend[:, 2] = 0.3 * (gx[:, 0] / L) ** 2
    bend[:, 0] = 0.02 * gx[:, 0] / L
    Vdir = np.zeros((nn, 3))
    Vdir[:, 2] = np.sin(np.pi * gx[:, 0] / L) * (0.5 + gx[:, 1] / W)
    Vdir[:, 0] = 0.3 * gx[:, 0] / L
    Vdir /= np.linalg.norm(Vdir)

    def run(nd_val, want_grad):
        assert np.allclose(plate_mesh.geometry.x, gx), "an op left mesh.geometry.x moved"
        rec = csdl.Recorder(inline=True); rec.start()
        layup, _ = _layup([15, -15, 80], 0.02)
        nd = csdl.Variable(value=nd_val.copy(), name="nd")
        o = out.failure_index(_run(plate_mesh, layup, nn, pz=5e3, node_disp=nd),
                              sampling=sampling)
        val = float(np.ravel(o.value)[0])
        g = None
        if want_grad:
            g = np.asarray(csdl.experimental.PySimulator(rec).compute_totals(
                [o], [nd])[o, nd]).reshape(nn, 3)
        rec.stop()
        return val, g

    _, g = run(bend, True)
    dd_an = float((g * Vdir).sum())
    step = 1e-3
    vp, _ = run(bend + step * Vdir, False)
    vm, _ = run(bend - step * Vdir, False)
    dd_fd = (vp - vm) / (2 * step)
    rel = abs(dd_an - dd_fd) / abs(dd_fd)
    print(f"\n[{sampling}] d(FI)/d(node_disp).V  analytic={dd_an:+.6e}  fd={dd_fd:+.6e}  rel={rel:.2e}")
    assert rel < 5e-4


# --- 5. interlaminar shear tau_13 ------------------------------------------
#
# The index must carry both tau_23 and tau_13. If `g13` is not computed and
# `ply_properties` withholds the G13/S13 keys the vendored per-point path asks for,
# the tau_13 branch goes dead, which under-predicts failure for out-of-plane load
# aligned with the fibres -- the non-conservative direction.

GAMMA = 1.0e-3          # engineering shear strain used by the unit gates below


def _index(eps, shear, angle_deg, props):
    """``_tsai_wu_index`` for a single point: laminate-frame ``[ex,ey,gxy]`` and
    ``[gxz,gyz]`` on one ply at ``angle_deg``. Needs a live inline recorder."""
    from hermit._laminate.failure import _tsai_wu_index

    return float(np.ravel(_tsai_wu_index(
        csdl.Variable(value=np.atleast_2d(np.asarray(eps, float))),
        csdl.Variable(value=np.atleast_2d(np.asarray(shear, float))),
        csdl.Variable(value=np.radians(float(angle_deg))), props).value)[0])


def test_ply_properties_exposes_the_13_pair():
    """``S13``/``G13`` are returned, and read from the right places.

    ``set_strength(F1t, F1c, F2t, F2c, F12, F23)`` stores the shear row as
    ``[F12, F12, F23]``: the middle entry *is* S13, equal to S12 because for a
    transversely isotropic ply the 1-3 plane contains the fibre exactly as the 1-2
    plane does. G13 is GA for the same reason (and ``clt.calc_q`` already splits the
    FSDT block that way). Pinning both against the raw strength array is what guards
    the ``s[2, 1]`` read if caddee_materials ever reshuffles the layout.
    """
    from hermit._laminate.failure import ply_properties

    mat = _ud()
    s = np.asarray(mat.strength, dtype=float)
    p = ply_properties(mat)

    assert p["S13"] == s[2, 1] == 70e6           # the entry that used to be skipped
    assert p["S13"] == p["S12"]                  # transverse isotropy: 1-3 == 1-2
    assert p["S23"] == s[2, 2] == 40e6           # ... and 2-3 is the weak plane
    assert p["G13"] == p["G12"] == 7e9           # GA shears any plane holding the fibre
    assert p["G23"] == pytest.approx(10e9 / (2 * (1 + 0.4)), rel=1e-12)


def test_tau13_enters_the_index_only_along_the_fibre(recorder):
    """A 0-degree ply carrying ``gxz`` is pure tau_13; carrying ``gyz`` it is pure tau_23.

    With the in-plane strain zero the whole index *is* the transverse-shear term, so
    the numbers are exact rather than incidental: (G13*gamma/S13)^2 = 0.01 here.
    """
    from hermit._laminate.failure import ply_properties

    p = ply_properties(_ud())
    no_13 = {k: v for k, v in p.items() if k not in ("S13", "G13")}   # the old behaviour
    zero = np.zeros(3)

    along = _index(zero, [GAMMA, 0.0], 0.0, p)
    across = _index(zero, [0.0, GAMMA], 0.0, p)
    print(f"\n0-deg ply, gamma={GAMMA}: tau_13 index={along:.6e}  tau_23 index={across:.6e}")

    # tau_13 loading: was exactly 0 (the term did not exist), is now the full term
    assert _index(zero, [GAMMA, 0.0], 0.0, no_13) == 0.0
    assert along == pytest.approx((p["G13"] * GAMMA) ** 2 / p["S13"] ** 2, rel=1e-12)
    assert along > 0.0

    # tau_23 loading on the same ply: g13 == 0, so the new term contributes nothing and
    # the index is bit-identical to the old one
    assert across == _index(zero, [0.0, GAMMA], 0.0, no_13)
    assert across == pytest.approx((p["G23"] * GAMMA) ** 2 / p["S23"] ** 2, rel=1e-12)

    # and the projection rotates with the ply: at 90 degrees the roles swap exactly
    assert _index(zero, [0.0, GAMMA], 90.0, p) == pytest.approx(along, rel=1e-12)
    assert _index(zero, [GAMMA, 0.0], 90.0, p) == pytest.approx(across, rel=1e-12)


def test_tau13_and_tau12_agree_by_transverse_isotropy(recorder):
    """For a transversely isotropic ply, rotating 90 degrees *about the fibre* is a
    material symmetry: it carries the 1-2 plane onto the 1-3 plane. So pure in-plane
    shear ``g12 = gamma`` and pure interlaminar shear ``g13 = gamma`` must give the
    same index -- which holds iff ``G13 == G12`` *and* ``S13 == S12`` together.

    The 2-3 plane is *not* equivalent (it is the isotropic plane, with the weaker
    matrix-dominated G23/S23), so tau_23 must land somewhere else -- otherwise a
    mis-wire of S13 to S23 would slip through the gate above.
    """
    from hermit._laminate.failure import ply_properties

    p = ply_properties(_ud())
    i12 = _index([0.0, 0.0, GAMMA], [0.0, 0.0], 0.0, p)     # in-plane shear only
    i13 = _index(np.zeros(3), [GAMMA, 0.0], 0.0, p)         # interlaminar, along fibre
    i23 = _index(np.zeros(3), [0.0, GAMMA], 0.0, p)         # interlaminar, across fibre
    print(f"\ngamma={GAMMA}: index tau_12={i12:.9e}  tau_13={i13:.9e}  tau_23={i23:.9e}")

    assert i13 == pytest.approx(i12, rel=1e-12)
    assert abs(i23 / i13 - 1.0) > 0.1        # measured ratio 0.797: a different plane


def test_index_degrades_without_interlaminar_data(recorder):
    """A property dict missing the pair (or carrying G13 = 0) must fall back to the
    in-plane criterion, exactly as the tau_23 term already did -- not raise."""
    from hermit._laminate.failure import ply_properties

    p = ply_properties(_ud())
    eps, shear = [1e-4, -2e-4, 3e-4], [GAMMA, 0.5 * GAMMA]
    full = _index(eps, shear, 25.0, p)
    for stripped in ({k: v for k, v in p.items() if k not in ("S13", "G13")},
                     dict(p, G13=0.0), dict(p, S13=0.0)):
        val = _index(eps, shear, 25.0, stripped)
        assert val < full                     # the term is genuinely gone ...
        assert np.isfinite(val)               # ... and nothing raised


def test_tau13_raises_the_solved_plate_index(plate_mesh, cantilever_ref, recorder, monkeypatch):
    """End to end: the term moves the real model, and moves it the way tau_13 must.

    The delta is >= 0 everywhere (a squared term can only add), and it is *identical
    on the two faces of a ply* because tau_13 does not vary through the ply thickness
    -- an in-plane or bending contribution could not do that.
    """
    import hermit._laminate.failure as F

    nn = int(cantilever_ref["n_nodes"])
    layup, _ = _layup([0, 90, 0], 6e-3)
    state = _run(plate_mesh, layup, nn, pz=2.0e3)
    new = np.asarray(out.failure_field(state).values)
    ks_new = float(np.ravel(out.failure_index(state).value)[0])

    full = F.ply_properties
    monkeypatch.setattr(F, "ply_properties",
                        lambda m: {k: v for k, v in full(m).items() if k not in ("S13", "G13")})
    old = np.asarray(out.failure_field(state).values)
    ks_old = float(np.ravel(out.failure_index(state).value)[0])

    d = new - old
    print(f"\n[0/90/0] KS failure index: without tau_13={ks_old:.6e}  with={ks_new:.6e} "
          f"(+{100 * (ks_new / ks_old - 1):.3f}%)  max cell delta={d.max():.6e}")
    assert d.min() >= 0.0 and d.max() > 0.0
    assert ks_new >= ks_old
    for k in range(3):                       # per ply: bottom face delta == top face delta
        assert np.allclose(d[:, 2 * k], d[:, 2 * k + 1], rtol=1e-12, atol=1e-14)
