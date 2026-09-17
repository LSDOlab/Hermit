"""``hermit._geometry`` (``Geometry`` / ``hm.geometry``) against
``hermit.csdl_helpers.resolve_mesh_nodes`` /
``hermit.fenics.maps.OrderingMaps.reference_mesh_nodes``.
"""

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit
from hermit._geometry import Geometry, _GEOM_SPACE, _assert_p1_coordinate_element, geometry
from hermit.fenics.maps import OrderingMaps
from hermit.fenics.shell_pde import ShellPDE


@pytest.fixture
def domain(plate_mesh):
    return hermit.ShellDomain(plate_mesh)


# -- the numerical gate: file-order reference nodes match the legacy machinery -----

def test_reference_nodes_match_ordering_maps_exactly(plate_mesh, domain, recorder):
    pde = ShellPDE(plate_mesh)
    maps = OrderingMaps(plate_mesh, pde)
    g = geometry(domain)
    assert np.array_equal(g.nodes.value, maps.reference_mesh_nodes)
    # and to ShellDomain's own file-order node_coords
    assert np.array_equal(g.nodes.value, domain.node_coords)


def test_reference_default_not_differentiable(domain, recorder):
    g = geometry(domain)
    assert g.is_differentiable is False
    assert np.array_equal(g.nodes.value, domain.node_coords)


# -- node_disp / nodes / both-given -------------------------------------------

def test_node_disp_adds_to_reference_and_sets_differentiable(domain, recorder):
    rng = np.random.RandomState(0)
    disp = csdl.Variable(value=0.01 * rng.normal(size=(domain.n_nodes, 3)))
    g = geometry(domain, node_disp=disp)
    assert g.is_differentiable is True
    assert np.allclose(g.nodes.value, domain.node_coords + disp.value)


def test_node_disp_accepts_plain_array(domain, recorder):
    disp = 0.01 * np.ones((domain.n_nodes, 3))
    g = geometry(domain, node_disp=disp)
    assert g.is_differentiable is True
    assert np.allclose(g.nodes.value, domain.node_coords + disp)


def test_nodes_absolute_path(domain, recorder):
    custom = domain.node_coords + 1.0
    g = geometry(domain, nodes=custom)
    assert g.is_differentiable is True
    assert np.allclose(g.nodes.value, custom)


def test_both_given_raises(domain, recorder):
    zeros = np.zeros((domain.n_nodes, 3))
    with pytest.raises(ValueError, match="not both"):
        geometry(domain, node_disp=zeros, nodes=zeros)


# -- .field: FE-order Field on the mesh coordinate element ----------------------

def test_field_is_on_geometry_space_and_matches_local_order(plate_mesh, domain, recorder):
    g = geometry(domain)
    assert g.field.space[:2] == _GEOM_SPACE[:2]
    assert g.field.value_shape == (3,)
    fe = np.asarray(g.field.coeffs.value).reshape(-1, 3)
    # file -> FE (== local, for CG1) permutation must reproduce the mesh's own local
    # vertex coordinates exactly.
    assert np.array_equal(fe, plate_mesh.geometry.x)


def test_field_reflects_node_disp(domain, recorder):
    disp = 0.02 * np.ones((domain.n_nodes, 3))
    g = geometry(domain, node_disp=disp)
    fe = np.asarray(g.field.coeffs.value).reshape(-1, 3)
    expected = (domain.node_coords + disp)[domain.node_input_idx]
    assert np.allclose(fe, expected)


# -- coordinate-element assertion ------------------------------------------

class _FakeCmap:
    def __init__(self, degree):
        self.degree = degree


class _FakeGeometry:
    def __init__(self, degree):
        self.cmaps = [_FakeCmap(degree)]


class _FakeMesh:
    def __init__(self, degree):
        self.geometry = _FakeGeometry(degree)


def test_p1_coordinate_element_assertion_passes_for_p1():
    _assert_p1_coordinate_element(_FakeMesh(1))  # no raise


def test_p1_coordinate_element_assertion_raises_for_higher_order():
    with pytest.raises(ValueError, match="P1"):
        _assert_p1_coordinate_element(_FakeMesh(2))


def test_geometry_construction_calls_the_p1_assertion(domain, recorder, monkeypatch):
    """Geometry.__init__ actually invokes the coordinate-element check (not just a
    dead helper) -- mesh.geometry.cmaps is a read-only dolfinx C++ property, so this
    patches the check itself rather than faking a higher-order mesh."""
    import sys

    # The module is ``hermit._geometry``: the underscore keeps it out of the way of
    # the exported ``hm.geometry`` builder (importing a submodule rebinds the parent's
    # attribute to the module, silently replacing the function). Same reason as
    # ``hermit/_field.py`` and ``hermit/_solve.py``.
    geom_mod = sys.modules["hermit._geometry"]

    def _boom(mesh):
        raise ValueError("P1 check ran")

    monkeypatch.setattr(geom_mod, "_assert_p1_coordinate_element", _boom)
    with pytest.raises(ValueError, match="P1 check ran"):
        geometry(domain)
