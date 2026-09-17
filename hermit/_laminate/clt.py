"""Classical lamination theory -- per-ply ABD assembly.

Vendored verbatim (modulo imports) from ``lamad.src.CLT``. ``compute_clt(layup)``
returns ``A, B, D, A_star`` for a ``Layup``, differentiable w.r.t. the ply angles
and heights.
"""

import numpy as np
import csdl_alpha as csdl
from caddee_materials import TransverseMaterial, IsotropicMaterial

from .layup import Layup


def compute_clt(layup: Layup):
    ply_stack = layup.angles
    h = layup.heights
    no_plies = layup.num_plies
    mat_list = read_materials(layup.materials)
    q, q_out = calc_q(mat_list)

    # transformed stiffness matrices, per ply
    q_bar = csdl.Variable((3, 3, no_plies), value=0)
    q_bar_out = csdl.Variable((2, 2, no_plies), value=0)
    for i in csdl.frange(no_plies):
        t_eps = calc_t_eps(ply_stack[i])
        t_sig_inv = calc_t_sig_inv(ply_stack[i])
        aa = t_sig_inv @ q[i]
        q_bar = q_bar.set(csdl.slice[:, :, i], aa @ t_eps)

        c = csdl.cos(ply_stack[i])
        s = csdl.sin(ply_stack[i])
        q_bar_out = q_bar_out.set(csdl.slice[0, 0, i], q_out[i, 0, 0] * c**2 + q_out[i, 1, 1] * s**2)
        q_bar_out = q_bar_out.set(csdl.slice[0, 1, i], (q_out[i, 1, 1] - q_out[i, 0, 0]) * s * c)
        q_bar_out = q_bar_out.set(csdl.slice[1, 0, i], (q_out[i, 1, 1] - q_out[i, 0, 0]) * s * c)
        q_bar_out = q_bar_out.set(csdl.slice[1, 1, i], q_out[i, 0, 0] * s**2 + q_out[i, 1, 1] * c**2)

    A, B, D, A_star = calc_ABD(q_bar, q_bar_out, h, no_plies)
    return A, B, D, A_star


def read_materials(materials):
    mat_list = []
    for mat in materials:
        if isinstance(mat, TransverseMaterial):
            E1, E2, v12, v23, G12 = mat.get_constants()
            mat_list.append([E1, E2, v12, G12, v23])
        elif isinstance(mat, IsotropicMaterial):
            E1, v12, G = mat.get_constants()
            mat_list.append([E1, E1, v12, G, v12])
        else:
            raise ValueError(f"Material type {type(mat).__name__} not supported")
    return mat_list


def calc_q(mat_prop):
    q = np.zeros((len(mat_prop), 3, 3))
    q_out = np.zeros((len(mat_prop), 2, 2))
    for i, mat in enumerate(mat_prop):
        E1, E2, v12, G12, v23 = mat
        G23 = E2 / (2 * (1 + v23))
        D = 1 - E2 / E1 * v12**2
        q[i] = np.array([[E1 / D, v12 * E2 / D, 0],
                         [v12 * E2 / D, E2 / D, 0],
                         [0, 0, G12]])            # CLT plane stress
        q_out[i] = np.array([[G23, 0], [0, G12]])  # out-of-plane (FSDT)
    return csdl.Variable(value=q), csdl.Variable(value=q_out)


def calc_t_sig_inv(theta):
    C = csdl.cos(theta).reshape((1, 1))
    S = csdl.sin(theta).reshape((1, 1))
    # CSDL main collapses ``(1, 1) ** 2`` to a one-dimensional singleton.
    # Matrix products retain the two-dimensional block shape required by
    # ``blockmat`` (and give the same values and derivatives).
    CC, SS = C * C, S * S
    return csdl.blockmat([[CC, SS, -2 * S * C],
                          [SS, CC, 2 * S * C],
                          [S * C, -S * C, CC - SS]])


def calc_t_eps(theta):
    C = csdl.cos(theta).reshape((1, 1))
    S = csdl.sin(theta).reshape((1, 1))
    CC, SS = C * C, S * S
    return csdl.blockmat([[CC, SS, S * C],
                          [SS, CC, -S * C],
                          [-2 * S * C, 2 * S * C, CC - SS]])


def calc_ABD(q_bar, q_bar_out, h, no_plies):
    A = csdl.Variable((3, 3), value=0)
    B = csdl.Variable((3, 3), value=0)
    D = csdl.Variable((3, 3), value=0)
    A_star = csdl.Variable((2, 2), value=0)

    z_bottom = -csdl.sum(h) / 2
    for i in csdl.frange(no_plies):
        z_top = z_bottom + h[i]
        a = h[i]
        b = (z_top**2 - z_bottom**2) / 2
        d = (z_top**3 - z_bottom**3) / 3
        A = A + q_bar[:, :, i] * a
        B = B + q_bar[:, :, i] * b
        D = D + q_bar[:, :, i] * d
        A_star = A_star + q_bar_out[:, :, i] * a
        z_bottom = z_top

    return A, B, D, A_star
