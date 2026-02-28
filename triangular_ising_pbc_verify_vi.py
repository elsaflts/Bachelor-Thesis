def verify(result: dict) -> dict:
    """Verify the VI solver output for the 2D triangular Ising model."""
    import numpy as np

    checks = []
    missing = []
    warnings = []

    beta_c_exact = 0.25 * np.log(3)
    T_c_exact = 1.0 / beta_c_exact
    z = 6
    J = 1.0

    # ── Check 1: Required top-level keys ──────────────────────────────
    required_keys = ["beta_c_exact", "T_c_exact", "mean_field"]
    for key in required_keys:
        if key not in result:
            missing.append(key)
    checks.append({
        "name": "required_keys_present",
        "pass": len(missing) == 0,
        "detail": f"Missing: {missing}" if missing else "All required keys present."
    })
    if missing:
        return {"pass": False, "checks": checks, "missing": missing,
                "warnings": warnings, "error": "Missing required keys."}

    # ── Check 2: T_c_exact correct ────────────────────────────────────
    tc = result.get("T_c_exact", 0)
    tc_ok = abs(tc - T_c_exact) < 1e-3
    checks.append({
        "name": "T_c_exact_correct",
        "pass": tc_ok,
        "detail": f"T_c = {tc:.6f}, expected {T_c_exact:.6f}."
    })

    # ── Check 3: Mean-field structure ─────────────────────────────────
    mf = result.get("mean_field", {})
    mf_keys = ["T_c_MF", "magnetization_at_bc", "energy_per_spin",
               "free_energy_per_spin", "entropy_per_spin"]
    mf_missing = [k for k in mf_keys if k not in mf]
    checks.append({
        "name": "mean_field_keys",
        "pass": len(mf_missing) == 0,
        "detail": f"Missing MF keys: {mf_missing}" if mf_missing
                  else "All mean-field keys present."
    })

    # ── Check 4: MF T_c = zJ = 6 ─────────────────────────────────────
    tc_mf = mf.get("T_c_MF", 0)
    tc_mf_ok = abs(tc_mf - z * J) < 0.1
    checks.append({
        "name": "MF_Tc_correct",
        "pass": tc_mf_ok,
        "detail": f"T_c^MF = {tc_mf:.4f}, expected {z*J:.1f}."
    })

    # ── Check 5: MF magnetization at β_c in [0, 1] ───────────────────
    m_mf = mf.get("magnetization_at_bc", -1)
    m_ok = 0.0 <= m_mf <= 1.0
    checks.append({
        "name": "MF_magnetization_domain",
        "pass": m_ok,
        "detail": f"m_MF = {m_mf:.6f}, expected in [0, 1]."
    })

    # MF at exact β_c (which is below MF β_c), m should be > 0
    # since β_c_exact > β_c_MF = 1/6, we are in the MF ordered phase
    m_ordered = m_mf > 0.5
    checks.append({
        "name": "MF_magnetization_ordered",
        "pass": m_ordered,
        "detail": f"m_MF = {m_mf:.6f} at exact β_c (in MF ordered phase, expect > 0.5)."
    })

    # ── Check 6: MF energy domain ────────────────────────────────────
    u_mf = mf.get("energy_per_spin", 0)
    # u_MF = -(z/2)*J*m^2, range [-3, 0]
    u_mf_ok = -3.01 < u_mf < 0.01
    checks.append({
        "name": "MF_energy_domain",
        "pass": u_mf_ok,
        "detail": f"u_MF = {u_mf:.6f}, expected in [-3, 0]."
    })

    # Verify u_MF = -(z/2)*m^2
    u_mf_expected = -(z / 2.0) * J * m_mf ** 2
    u_mf_consistent = abs(u_mf - u_mf_expected) < 0.01
    checks.append({
        "name": "MF_energy_consistent",
        "pass": u_mf_consistent,
        "detail": f"u_MF = {u_mf:.6f}, -(z/2)m² = {u_mf_expected:.6f}."
    })

    # ── Check 7: MF free energy finite ────────────────────────────────
    f_mf = mf.get("free_energy_per_spin", float('nan'))
    f_mf_ok = np.isfinite(f_mf)
    checks.append({
        "name": "MF_free_energy_finite",
        "pass": f_mf_ok,
        "detail": f"f_MF = {f_mf:.6f}." if f_mf_ok else f"f_MF = {f_mf} (not finite)."
    })

    # ── Check 8: MF entropy non-negative ──────────────────────────────
    s_mf = mf.get("entropy_per_spin", -1)
    s_mf_ok = -0.01 < s_mf < np.log(2) + 0.05
    checks.append({
        "name": "MF_entropy_domain",
        "pass": s_mf_ok,
        "detail": f"s_MF = {s_mf:.6f}, expected in [0, ln2 ≈ 0.693]."
    })

    # ── Check 9: MF thermodynamic consistency ─────────────────────────
    s_mf_check = beta_c_exact * (u_mf - f_mf)
    s_mf_consistent = abs(s_mf - s_mf_check) < 0.05
    checks.append({
        "name": "MF_thermodynamic_consistency",
        "pass": s_mf_consistent,
        "detail": f"s_MF = {s_mf:.6f}, β(u-f) = {s_mf_check:.6f}."
    })

    # ── Check 10: MF variational bound: f_MF ≥ f_exact ───────────────
    # We don't know f_exact precisely, but f_MF should be > -4
    # (a very conservative lower bound)
    f_mf_bound = f_mf > -5.0
    checks.append({
        "name": "MF_free_energy_plausible",
        "pass": f_mf_bound,
        "detail": f"f_MF = {f_mf:.6f}, should be > -5 (conservative bound)."
    })

    # ── Check 11: MF grid data present and consistent ─────────────────
    T_grid = mf.get("T_grid", [])
    m_grid = mf.get("magnetization_MF", [])
    u_grid = mf.get("energy_MF", [])
    f_grid = mf.get("free_energy_MF", [])
    grid_ok = (len(T_grid) > 0 and len(T_grid) == len(m_grid)
               == len(u_grid) == len(f_grid))
    checks.append({
        "name": "MF_grid_data_consistent",
        "pass": grid_ok,
        "detail": f"Grid lengths: T={len(T_grid)}, m={len(m_grid)}, "
                  f"u={len(u_grid)}, f={len(f_grid)}."
    })

    # ── Check 12: MF magnetization in [0, 1] for all T ───────────────
    if len(m_grid) > 0:
        m_arr = np.array(m_grid)
        m_domain_ok = np.all(m_arr >= -0.01) and np.all(m_arr <= 1.01)
        checks.append({
            "name": "MF_magnetization_grid_domain",
            "pass": m_domain_ok,
            "detail": f"m range: [{m_arr.min():.6f}, {m_arr.max():.6f}], "
                      "expected [0, 1]."
        })

    # ── Check 13: VAN results (if present) ────────────────────────────
    van = result.get("van", None)
    if van is not None and isinstance(van, dict):
        van_keys = ["sizes", "f_values", "u_values", "s_values",
                    "f_infty", "u_infty", "s_infty"]
        van_missing = [k for k in van_keys if k not in van]
        checks.append({
            "name": "VAN_keys_present",
            "pass": len(van_missing) == 0,
            "detail": f"Missing VAN keys: {van_missing}" if van_missing
                      else "All VAN keys present."
        })

        # VAN free energy values should be finite
        van_f = van.get("f_values", [])
        van_f_finite = all(np.isfinite(v) for v in van_f)
        checks.append({
            "name": "VAN_free_energy_finite",
            "pass": van_f_finite,
            "detail": "All VAN free energy values finite." if van_f_finite
                      else "Some VAN free energy values are NaN/inf."
        })

        # VAN energy should be in plausible range
        van_u = van.get("u_values", [])
        if van_u:
            van_u_ok = all(-4.0 < v < 1.0 for v in van_u)
            checks.append({
                "name": "VAN_energy_plausible",
                "pass": van_u_ok,
                "detail": f"VAN u range: [{min(van_u):.4f}, {max(van_u):.4f}]."
            })

        # VAN extrapolated entropy should be non-negative
        s_van = van.get("s_infty", -1)
        s_van_ok = -0.1 < s_van < np.log(2) + 0.2
        checks.append({
            "name": "VAN_entropy_plausible",
            "pass": s_van_ok,
            "detail": f"s_VAN_∞ = {s_van:.6f}, expected roughly in [0, ln2]."
        })
        if not s_van_ok:
            warnings.append(f"VAN entropy {s_van:.6f} outside expected range.")
    else:
        warnings.append("VAN results not present (PyTorch may not be available).")

    # ── Overall pass ──────────────────────────────────────────────────
    all_pass = all(c["pass"] for c in checks)

    return {
        "pass": all_pass,
        "checks": checks,
        "missing": missing,
        "warnings": warnings,
    }


if __name__ == "__main__":
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vi_solver",
        "/home/elsafelts/Bachelor thesis/Quantum Chemistry/triangular_ising_pbc_solver_vi.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.solver()

    report = verify(result)
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT (Variational Inference)")
    print("=" * 60)
    print(f"Overall: {'PASS' if report['pass'] else 'FAIL'}")
    for c in report["checks"]:
        status = "✓" if c["pass"] else "✗"
        print(f"  {status} {c['name']}: {c['detail']}")
    if report["warnings"]:
        print(f"\nWarnings: {report['warnings']}")
