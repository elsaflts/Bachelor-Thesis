"""
Verifier for 2D Ising Model — Variational Inference (Mean-Field + VAN)
Checks that inference results adhere to all probabilistic model requirements.
"""

def solver():
    import numpy as np

    # Run the VI solver to get results
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vi_solver",
        "/home/elsafelts/Bachelor thesis/Quantum Chemistry/ising_2d_solver_vi.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.solver()

    failures = []
    J = 1.0
    T_c_exact = 2.0 * J / np.log(1.0 + np.sqrt(2.0))

    # ── Requirement 1: Top-level structure ───────────────────────────────
    for key in ["T_c_exact", "mean_field", "exact"]:
        if key not in result:
            failures.append(f"missing_top_key({key})")

    if "error" in result:
        failures.append(f"solver_error({result['error']})")
        print({"evaluation": ", ".join(failures)})
        return {"evaluation": ", ".join(failures)}

    # ── Requirement 2: T_c correctly identified ──────────────────────────
    if abs(result.get("T_c_exact", 0) - T_c_exact) > 1e-4:
        failures.append("T_c_exact_incorrect")

    # ── Requirement 3: Mean-field results present and valid ──────────────
    mf = result.get("mean_field", {})
    if not isinstance(mf, dict):
        failures.append("mean_field_not_dict")
    else:
        for key in ["T_grid", "magnetization_MF", "energy_MF", "free_energy_MF"]:
            if key not in mf:
                failures.append(f"MF_missing({key})")

        T_mf = mf.get("T_c_MF", 0)
        # Mean-field T_c for z=4 square lattice should be 4J
        if abs(T_mf - 4.0 * J) > 0.1:
            failures.append("MF_Tc_incorrect")

        m_mf = np.array(mf.get("magnetization_MF", []))
        if len(m_mf) > 0:
            # Magnetization must be in [0, 1]
            if np.any(m_mf < -0.01) or np.any(m_mf > 1.01):
                failures.append("MF_magnetization_out_of_domain")
            # At low T (e.g. T=1), MF magnetization should be near 1
            T_grid_mf = np.array(mf.get("T_grid", []))
            if len(T_grid_mf) > 0:
                low_T_mask = T_grid_mf < 1.5
                if np.any(low_T_mask) and np.mean(m_mf[low_T_mask]) < 0.8:
                    failures.append("MF_magnetization_too_low_at_low_T")

        u_mf = np.array(mf.get("energy_MF", []))
        if len(u_mf) > 0:
            # MF energy per spin: u = -(z/2)*J*m^2, range [-2, 0]
            if np.any(u_mf < -2.01 * J) or np.any(u_mf > 0.01):
                failures.append("MF_energy_out_of_domain")

        f_mf = np.array(mf.get("free_energy_MF", []))
        if len(f_mf) > 0:
            if np.any(~np.isfinite(f_mf)):
                failures.append("MF_free_energy_not_finite")

    # ── Requirement 4: Exact reference values present and correct ────────
    exact = result.get("exact", {})
    if not isinstance(exact, dict):
        failures.append("exact_not_dict")
    else:
        for key in ["T", "f", "u", "c", "m"]:
            if key not in exact:
                failures.append(f"exact_missing({key})")

        u_ex = np.array(exact.get("u", []))
        if len(u_ex) > 0:
            # Exact energy must be in [-2, 0]
            if np.any(u_ex < -2.01) or np.any(u_ex > 0.01):
                failures.append("exact_energy_out_of_domain")

        m_ex = np.array(exact.get("m", []))
        if len(m_ex) > 0:
            if np.any(m_ex < -0.01) or np.any(m_ex > 1.01):
                failures.append("exact_magnetization_out_of_domain")

        c_ex = np.array(exact.get("c", []))
        if len(c_ex) > 0:
            if np.any(c_ex < -0.01):
                failures.append("exact_specific_heat_negative")

    # ── Requirement 5: VAN results (if present) are reasonable ───────────
    van = result.get("van", None)
    if van is not None and isinstance(van, dict):
        f_van = np.array(van.get("free_energy_VAN", []))
        u_van = np.array(van.get("energy_VAN", []))
        m_van = np.array(van.get("magnetization_VAN", []))

        if len(f_van) > 0 and np.any(~np.isfinite(f_van)):
            failures.append("VAN_free_energy_not_finite")

        if len(u_van) > 0:
            # VAN energy should be in plausible range (wider tolerance for small lattice)
            if np.any(u_van < -2.5) or np.any(u_van > 0.5):
                failures.append("VAN_energy_out_of_range")

        if len(m_van) > 0:
            if np.any(m_van < -0.01) or np.any(m_van > 1.01):
                failures.append("VAN_magnetization_out_of_domain")

        # VAN free energy should be an upper bound on exact (variational principle)
        # Allow some tolerance for importance sampling estimate
        T_van = np.array(van.get("T_grid", []))
        T_ex = np.array(exact.get("T", []))
        f_ex_vals = np.array(exact.get("f", []))
        if len(f_van) > 0 and len(T_van) > 0 and len(T_ex) > 0:
            for i, tv in enumerate(T_van):
                idx = np.argmin(np.abs(T_ex - tv))
                if abs(T_ex[idx] - tv) < 0.05:
                    # IS estimate can go below exact, but not by much
                    if f_van[i] < f_ex_vals[idx] - 0.5:
                        failures.append(f"VAN_free_energy_below_exact_at_T={tv:.2f}")

    # ── Requirement 6: Diagnostics present ───────────────────────────────
    diag = result.get("diagnostics", {})
    if not isinstance(diag, dict):
        failures.append("diagnostics_not_dict")

    # ── Requirement 7: MF is qualitatively correct variational approx ────
    # MF free energy should be >= exact free energy (variational upper bound)
    if len(f_mf) > 0 and "f" in exact:
        T_mf_grid = np.array(mf.get("T_grid", []))
        T_ex_grid = np.array(exact["T"])
        f_ex_grid = np.array(exact["f"])
        bound_violations = 0
        for i, t in enumerate(T_mf_grid):
            idx = np.argmin(np.abs(T_ex_grid - t))
            if abs(T_ex_grid[idx] - t) < 0.05:
                # MF free energy should be >= exact (it's an upper bound)
                if f_mf[i] < f_ex_grid[idx] - 0.01:
                    bound_violations += 1
        if bound_violations > len(T_mf_grid) // 2:
            failures.append("MF_variational_bound_violated")

    # ── Final evaluation ─────────────────────────────────────────────────
    if not failures:
        evaluation = {"evaluation": "correct"}
    else:
        evaluation = {"evaluation": ", ".join(failures)}

    print("=" * 60)
    print("VERIFICATION RESULT (Variational Inference)")
    print("=" * 60)
    print(evaluation)
    return evaluation


if __name__ == "__main__":
    result = solver()
