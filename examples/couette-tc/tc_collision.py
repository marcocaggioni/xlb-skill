#!/usr/bin/env python3
"""Three-component (TC) collision operator for XLB 0.3.2 (JAX backend).

XLB ships BGK, KBC and Smagorinsky collision operators, but no rheology
dropdown: a non-Newtonian fluid means making the relaxation frequency a
function of the local shear rate. This operator does that for the TC law

    tau = sigma_y + sigma_y*sqrt(gamma_dot/gamma_dot_c) + eta_bg*gamma_dot

following the pattern of XLB's own SmagorinskyLESBGK operator
(xlb/operator/collision/smagorinsky_les_bgk.py): the local strain rate
is read off the non-equilibrium stress tensor, no velocity gradients
are needed.

Strain rate from the populations (validated numerically, Nov 17, on the
Newtonian Couette steady field: median gamma_pi/gamma_analytic = 1.003,
mean 1.012 over interior fluid nodes):
    Pi_neq[a,b] = sum_i c_ia c_ib (f_i - f_i^eq)     (cc = (xx, xy, yy))
    S = -Pi_neq / (2 rho c_s^2 tau),  c_s^2 = 1/3
    gamma_dot_lb = sqrt(2 S:S) = 3*|Pi_neq|_F / (sqrt(2) rho tau)
where |.|_F is the Frobenius norm and tau = 1/omega is the LOCAL
relaxation time. The denominator uses tau, not tau - 1/2 (checked:
tau - 1/2 gives a factor 2.02 error on the same test).

tau itself depends on gamma_dot through the viscosity, so the two
relations are solved together, per node, by bisection on
    h(g) = g * (0.5 + C0*eta(g)) = K/dt,
    K = 3*|Pi_neq|_F / (sqrt(2) rho),   C0 = 3*dt / (rho_phys*dx^2),
    g = gamma_dot in 1/s,  eta(g) = min(eta_TC(g), eta_max).
h(g) is monotonic increasing (g*eta(g) = sigma_y + sigma_y*sqrt(g/g_c)
+ eta_bg*g in the uncapped region, eta_max*g in the capped region), so
the bisection (30 iterations, vectorised over the grid) has a unique
root. The scalar omega the stepper passes in is ignored: it is only
the reference omega the lattice was built at (omega_ref = 1.0).

Regularization: eta(gamma) diverges as gamma -> 0, so eta is capped
at eta_max = eta(gamma_reg), gamma_reg = 1e-3 1/s (case.yaml). In the
plug eta = eta_max and omega = 1/(0.5 + C0*eta_max) falls to ~0.02 at
gamma = 0.1 and ~0.00003 at gamma = 100 — far below the [0.8, 1.9]
band, but that is the plug being near-solid, not an instability: the
plug is near-stationary (populations near rest equilibrium) and the
runs are 3,000 steps. omega is clipped above at 1.9 (eta -> eta_bg
at very high local gamma) for stability; it is not clipped below.

JAX backend only (Warp/Neon would need a warp functional; this repo's
runs use JAX on CPU).
"""
from functools import partial

import jax.numpy as jnp
from jax import jit

from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.collision.collision import Collision


class TCCollision(Collision):
    """BGK collision with a local omega from the TC constitutive law."""

    def __init__(self, velocity_set=None, precision_policy=None,
                 compute_backend=None, dt=1.0, dx=1.0, rho_phys=1000.0,
                 sigma_y=70.0, gamma_c=100.0, eta_bg=0.05, eta_max=70221.4):
        super().__init__(velocity_set, precision_policy, compute_backend)
        self._dt = float(dt)
        self._dx = float(dx)
        self._rho_phys = float(rho_phys)
        self._sigma_y = float(sigma_y)
        self._gamma_c = float(gamma_c)
        self._eta_bg = float(eta_bg)
        self._eta_max = float(eta_max)
        # C0 = 3*dt / (rho_phys * dx^2): tau = 0.5 + C0 * eta(gamma)
        self._c0 = 3.0 * float(dt) / (float(rho_phys) * float(dx) ** 2)

    def _eta(self, g):
        """TC viscosity (Pa s), capped at eta_max. g in 1/s, >= 0."""
        dt = self.compute_dtype
        safe_g = jnp.where(g > 0.0, g, 1.0)
        uncapped = (dt(self._sigma_y) / safe_g
                    + dt(self._sigma_y) / jnp.sqrt(safe_g * dt(self._gamma_c))
                    + dt(self._eta_bg))
        uncapped = jnp.where(g > 0.0, uncapped, dt(self._eta_max))
        return jnp.minimum(uncapped, dt(self._eta_max))

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, f, feq, omega):
        # omega (scalar, reference) is intentionally unused: the local
        # relaxation frequency is computed from the local strain rate.
        dt = self.compute_dtype
        fneq = f - feq
        # Non-equilibrium stress, cc components (xx, xy, yy).
        pi = jnp.tensordot(self.velocity_set.cc, fneq, axes=(0, 0))
        a = jnp.sqrt(pi[0] ** 2 + pi[2] ** 2 + dt(2.0) * pi[1] ** 2)
        rho = jnp.sum(f, axis=0)
        rho = jnp.where(rho > 0.0, rho, dt(1.0))
        # K = gamma_dot_lb * tau ; target S = K/dt = g*(0.5 + C0*eta(g))
        k = dt(3.0) * a / (dt(1.4142135623730951) * rho)
        target = k / dt(self._dt)
        c0 = dt(self._c0)
        # Bisection on h(g) = 0.5*g + C0*g*eta(g); h(0) = 0, h(2*target)
        # >= target, so the root lies in [0, 2*target].
        lo = jnp.zeros_like(target)
        hi = dt(2.0) * target
        for _ in range(30):
            mid = dt(0.5) * (lo + hi)
            h = dt(0.5) * mid + c0 * mid * self._eta(mid)
            lo = jnp.where(h < target, mid, lo)
            hi = jnp.where(h < target, hi, mid)
        g = dt(0.5) * (lo + hi)
        tau = dt(0.5) + c0 * self._eta(g)
        omega_eff = dt(1.0) / tau
        # Stability clip at the top only (eta -> eta_bg at huge gamma).
        omega_eff = jnp.minimum(omega_eff, dt(1.9))
        return f - omega_eff[None, ...] * fneq
