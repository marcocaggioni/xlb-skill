#!/usr/bin/env python3
"""Reference analytic TC Couette solution (shared by all five solver repos).

Geometry: inner cylinder Ri rotating at Omega, outer Ro fixed, length L.
TC constitutive law: tau = sigma_y + sigma_y*sqrt(gamma_dot/gamma_dot_c)
                          + eta_bg*gamma_dot.
In Couette flow tau(r) = T / (2*pi*L*r^2).  The local shear rate follows
from inverting the constitutive law (quadratic in sqrt(gamma_dot));
where tau < sigma_y the fluid is unyielded (gamma_dot = 0), so the
integral for Omega only runs out to the yield radius
    R_y = min(Ro, sqrt(T / (2*pi*L*sigma_y))).
Omega = integral_{Ri}^{R_y} gamma_dot(tau(r)) dr / r.
Solve for T given Omega by bisection (Omega(T) is monotonic in T).

Sweep points use the Newtonian-run convention: the label gamma is the
Newtonian inner-wall shear rate, Omega = gamma*(Ro^2-Ri^2)/(2*Ro^2).
"""
import math

# --- TC parameters (Marco, Nov 17, voice) ---
SIGMA_Y = 70.0      # Pa
GAMMA_DOT_C = 100.0  # 1/s
ETA_BG = 0.05        # Pa s  (50 cP)

# --- Geometry (same as Newtonian benchmark) ---
RI = 0.020  # m
RO = 0.025  # m
L = 0.060   # m

B = SIGMA_Y / math.sqrt(GAMMA_DOT_C)  # coefficient of sqrt(gamma_dot)


def gamma_of_tau(tau):
    """Invert TC law: shear rate for a given shear stress (0 if unyielded)."""
    if tau <= SIGMA_Y:
        return 0.0
    # eta_bg*x^2 + B*x + (sigma_y - tau) = 0, x = sqrt(gamma_dot)
    disc = B * B - 4.0 * ETA_BG * (SIGMA_Y - tau)
    x = (-B + math.sqrt(disc)) / (2.0 * ETA_BG)
    return x * x


def omega_of_T(T, n=2000):
    """Angular velocity for a given torque, by quadrature over the gap."""
    if T <= 0:
        return 0.0
    r_yield = math.sqrt(T / (2.0 * math.pi * L * SIGMA_Y))
    r_hi = min(RO, r_yield)
    if r_hi <= RI:
        return 0.0
    # log-spaced quadrature (integrand ~ 1/r)
    total = 0.0
    ln_lo, ln_hi = math.log(RI), math.log(r_hi)
    dln = (ln_hi - ln_lo) / n
    for i in range(n):
        ln_r = ln_lo + (i + 0.5) * dln
        r = math.exp(ln_r)
        tau = T / (2.0 * math.pi * L * r * r)
        total += gamma_of_tau(tau) * dln  # gamma(r)/r * dr = gamma * d(ln r)
    return total


def torque_of_omega(omega, tol=1e-12):
    """Torque for a given angular velocity, by bisection on T."""
    # T must exceed the yield torque at the inner wall for any motion.
    T_lo = 2.0 * math.pi * L * SIGMA_Y * RI * RI  # tau(Ri) = sigma_y
    T_hi = T_lo * 2.0
    while omega_of_T(T_hi) < omega:
        T_hi *= 2.0
    for _ in range(200):
        T_mid = 0.5 * (T_lo + T_hi)
        if omega_of_T(T_mid) < omega:
            T_lo = T_mid
        else:
            T_hi = T_mid
        if (T_hi - T_lo) / T_mid < tol:
            break
    return 0.5 * (T_lo + T_hi)


def tau_inner(T):
    return T / (2.0 * math.pi * RI * RI * L)


if __name__ == "__main__":
    print("gamma_label  Omega(rad/s)  T_analytic(N m)  tau_i(Pa)  eta_app=tau_i/gamma")
    for gamma in [0.1, 0.3, 1, 3, 10, 30, 100]:
        omega = gamma * (RO * RO - RI * RI) / (2.0 * RO * RO)
        T = torque_of_omega(omega)
        tau_i = tau_inner(T)
        # yield radius check
        r_y = math.sqrt(T / (2 * math.pi * L * SIGMA_Y))
        print(f"{gamma:>6}  {omega:>10.4f}  {T:.6e}  {tau_i:>9.3f}  {tau_i/gamma:>10.4f}   "
              f"R_y={min(r_y, RO)*1000:.2f} mm")
    # sanity: Newtonian eta=1 at gamma=10 -> T = 4*pi*L*eta*Omega*Ri^2*Ro^2/(Ro^2-Ri^2)
    print("\nSanity: pure Newtonian check via same integral machinery is not "
          "applicable (different law); instead check TC at high rate tends to "
          "sigma_y + sigma_y*sqrt(gamma/gamma_c) + eta_bg*gamma at inner wall:")
    for gamma in [10, 100]:
        omega = gamma * (RO * RO - RI * RI) / (2.0 * RO * RO)
        T = torque_of_omega(omega)
        tau_tc_label = (SIGMA_Y + SIGMA_Y * math.sqrt(gamma / GAMMA_DOT_C)
                        + ETA_BG * gamma)
        print(f"  gamma={gamma}: tau_i={tau_inner(T):.3f} Pa vs TC law at label "
              f"rate {tau_tc_label:.3f} Pa (inner-wall rate exceeds label rate)")
