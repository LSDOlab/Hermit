"""Through-thickness stress recovery for the R-M shell (isotropic).

Port of ``ShellStressRM`` from femo dev_coupling. Used only for stress postprocessing;
needs the isotropic ``E``, ``nu`` (so composite stress recovery is a separate future
concern). All gradients are plain ``ufl.grad``.
"""

from ufl import (
    as_matrix,
    as_tensor,
    as_vector,
    cross,
    dot,
    grad,
    indices,
    sqrt,
    split,
)

from .kinematics import (
    constant_normal_bending_curvature,
    global_to_local_inplane,
    local_basis_inplane,
    membrane_strain,
    voigt2D,
)


class ShellStressRM:
    def __init__(self, mesh, w, h_th, E, nu, *, strains=None):
        """Build isotropic through-thickness stress recovery.

        ``strains=(eps_mid, kappa, gamma)`` lets the shell form supply its own
        kinematics. The fallback preserves the existing constructor surface while
        using the same constant-normal definitions.
        """
        self.mesh = mesh
        self.u_mid, self.theta = split(w)

        E0, E1, self.E2 = local_basis_inplane(mesh)
        self.E01 = global_to_local_inplane(E0, E1)
        self.E012 = as_matrix(
            [[E0[i] for i in range(3)], [E1[i] for i in range(3)], [self.E2[i] for i in range(3)]]
        )

        if strains is None:
            self.eps_mid = membrane_strain(self.u_mid, self.E01)
            self.kappa = constant_normal_bending_curvature(self.theta, self.E01)
            self.gamma = None
        else:
            self.eps_mid, self.kappa, self.gamma = strains

        self.G = E / 2 / (1 + nu)
        self.Dmat = (E / (1.0 - nu * nu)) * as_matrix(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
        )

    def u(self, xi2):
        return self.u_mid - xi2 * cross(self.E2, self.theta)

    def eps(self, xi2):
        return voigt2D(self.eps_mid - xi2 * self.kappa)

    def gamma_2(self, xi2):
        """Transverse shear, constant through the thickness for RM kinematics.

        When the shell form supplies its strains, use its ``gamma`` rather than
        recomputing: the local form here differentiates ``u(xi2)``, which carries a
        ``xi2 * grad(E2)`` term that is non-zero on a warped cell under rigid-body
        motion -- the same defect as issue #7, in the shear slot. The supplied
        ``gamma`` is built from ``grad(u_mid)`` and cancels exactly.
        """
        if self.gamma is not None:
            return self.gamma
        dudxi2_global = -cross(self.E2, self.theta)
        i, j = indices(2)
        dudxi2_local = as_tensor(dudxi2_global[j] * self.E01[i, j], (i,))
        gradu2_local = as_tensor(dot(self.E2, grad(self.u(xi2)))[j] * self.E01[i, j], (i,))
        return dudxi2_local + gradu2_local

    def cauchy(self, xi2):
        return self.Dmat * self.eps(xi2), self.G * self.gamma_2(xi2)

    def inplane_stress(self, xi2):
        sig_hat, _ = self.cauchy(xi2)
        sig3d = as_matrix(
            [[sig_hat[0], sig_hat[2], 0], [sig_hat[2], sig_hat[1], 0], [0, 0, 0]]
        )
        i, j, k, ll = indices(4)
        return as_tensor(self.E012[i, k] * sig3d[k, ll] * self.E012[j, ll], (i, j))

    def von_mises(self, xi2):
        sig_hat, _ = self.cauchy(xi2)
        return sqrt(
            sig_hat[0] ** 2 - sig_hat[0] * sig_hat[1] + sig_hat[1] ** 2 + 3 * sig_hat[2] ** 2
        )
