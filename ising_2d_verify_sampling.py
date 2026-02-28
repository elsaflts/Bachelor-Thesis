"""
Verifier for 2D Ising Model — Sampling-based inference (Wolff Cluster, Strategy 2)
Checks that inference results adhere to all probabilistic model requirements.
"""

def solver():
    import numpy as np
    from scipy.special import ellipk, ellipe
    from scipy import integrate

    # First, run the sampling solver to get results
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "sampling_solver",
        "/home/elsafelts/Bachelor thesis/Quantum Chemistry/ising_2d_solver_sampling.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.solver()

    failures = []
    J = 1.0
    T_c_exact = 2.0 * J / np.log(1.0 + np.sqrt(2.0))

    # ── Requirement 1: All requested quantities are present ──────────────
    required_keys = [
        "T_c_exact", "T_grid", "energy_per_spin_MC", "energy_per_spin_exact",
        "specific_heat_MC", "specific_heat_exact", "magnetization_MC",
        "magnetization_exact", "free_energy_MC", "free_energy_exact",
        "susceptibility_MC", "diagnostics"
    ]
    missing = [k for k in required_keys if k not in result]
    if missing:
        failures.append(f"missing_keys({','.join(missing)})")

    # ── Requirement 2: T_c identification ────────────────────────────────
    # The exact critical temperature must be correctly identified
    if abs(result.get("T_c_exact", 0) - T_c_exact) > 1e-4:
        failures.append("T_c_exact_incorrect")

    # The estimated T_c from MC should be within ~10% of exact for L>=16
    T_c_est = result.get("T_c_estimated_c", 0)
    if abs(T_c_est - T_c_exact) / T_c_exact > 0.15:
        failures.append("T_c_estimated_too_far")

    # ── Requirement 3: Energy per spin domain and accuracy ───────────────
    T_grid = np.array(result.get("T_grid", []))
    u_mc = np.array(result.get("energy_per_spin_MC", []))
    u_exact = np.array(result.get("energy_per_spin_exact", []))

    if len(u_mc) > 0:
        # Energy per spin must be in [-2J, 0] for the 2D Ising model
        if np.any(u_mc < -2.0 * J - 0.01) or np.any(u_mc > 0.01):
            failures.append("energy_out_of_domain")

        # Energy should be monotonically increasing with T (less negative)
        if len(u_mc) > 1:
            sort_idx = np.argsort(T_grid)
            u_sorted = u_mc[sort_idx]
            # Allow small violations due to MC noise
            violations = np.sum(np.diff(u_sorted) < -0.05)
            if violations > 1:
                failures.append("energy_not_monotonic")

        # RMSE between MC and exact should be reasonable for L>=16
        rmse_u = np.sqrt(np.mean((u_mc - u_exact)**2))
        if rmse_u > 0.1:
            failures.append(f"energy_RMSE_too_large({rmse_u:.4f})")
    else:
        failures.append("energy_data_empty")

    # ── Requirement 4: Specific heat must be non-negative ────────────────
    c_mc = np.array(result.get("specific_heat_MC", []))
    if len(c_mc) > 0:
        if np.any(c_mc < -0.01):
            failures.append("specific_heat_negative")
        # Specific heat should peak near T_c
        if len(c_mc) > 3 and len(T_grid) > 3:
            peak_T = T_grid[np.argmax(c_mc)]
            if abs(peak_T - T_c_exact) / T_c_exact > 0.2:
                failures.append("specific_heat_peak_mislocated")

    # ── Requirement 5: Magnetization domain [0, 1] ──────────────────────
    m_mc = np.array(result.get("magnetization_MC", []))
    if len(m_mc) > 0:
        if np.any(m_mc < -0.01) or np.any(m_mc > 1.01):
            failures.append("magnetization_out_of_domain")
        # Below T_c, magnetization should be significantly nonzero
        below_tc = T_grid < T_c_exact * 0.9
        if np.any(below_tc):
            m_below = m_mc[below_tc]
            if np.mean(m_below) < 0.3:
                failures.append("magnetization_too_low_below_Tc")

    # ── Requirement 6: Free energy is finite and reasonable ──────────────
    f_mc = np.array(result.get("free_energy_MC", []))
    f_exact = np.array(result.get("free_energy_exact", []))
    if len(f_mc) > 0 and len(f_exact) > 0:
        if np.any(~np.isfinite(f_mc)):
            failures.append("free_energy_not_finite")
        # Free energy should be negative for all T
        valid = np.isfinite(f_mc) & (f_mc != 0)
        if np.any(valid):
            if np.any(f_mc[valid] > 0.1):
                failures.append("free_energy_positive")

    # ── Requirement 7: Exact values are correctly computed ───────────────
    # Verify that the exact solution functions give correct known values
    def exact_energy(T):
        K = J / T
        kappa = 2.0 * np.sinh(2*K) / np.cosh(2*K)**2
        ks = np.clip(np.abs(kappa), 0, 1-1e-15)
        K1 = ellipk(ks**2)
        return -J/np.tanh(2*K) * (1 + (2/np.pi)*(2*np.tanh(2*K)**2 - 1)*K1)

    u_tc_exact = exact_energy(T_c_exact)
    if abs(u_tc_exact - (-np.sqrt(2))) > 1e-4:
        failures.append("exact_energy_at_Tc_wrong")

    # Check exact values in the result match our independent computation
    if len(u_exact) > 0 and len(T_grid) > 0:
        our_exact = np.array([exact_energy(t) for t in T_grid])
        if np.max(np.abs(u_exact - our_exact)) > 1e-3:
            failures.append("exact_reference_values_inconsistent")

    # ── Requirement 8: Diagnostics present ───────────────────────────────
    diag = result.get("diagnostics", {})
    if not isinstance(diag, dict):
        failures.append("diagnostics_not_dict")
    elif "RMSE_energy" not in diag:
        failures.append("diagnostics_missing_RMSE")

    # ── Final evaluation ─────────────────────────────────────────────────
    if not failures:
        evaluation = {"evaluation": "correct"}
    else:
        evaluation = {"evaluation": ", ".join(failures)}

    print("=" * 60)
    print("VERIFICATION RESULT (Sampling)")
    print("=" * 60)
    print(evaluation)
    return evaluation


if __name__ == "__main__":
    result = solver()
