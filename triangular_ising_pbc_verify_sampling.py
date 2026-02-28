def verify(result: dict) -> dict:
    """Verify the sampling-based solver output for the 2D triangular Ising model."""
    import numpy as np

    checks = []
    missing = []
    warnings = []

    # Known exact values for reference
    beta_c_exact = 0.25 * np.log(3)  # ≈ 0.274653

    # ── Check 1: Required top-level keys ──────────────────────────────
    required_keys = [
        "beta_c", "beta_c_exact", "beta_c_error",
        "mean_energy_per_spin", "free_energy_per_spin", "entropy_per_spin",
        "sizes_used", "f_values", "u_values", "s_values",
    ]
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

    # ── Check 2: All values finite (no NaN/inf) ──────────────────────
    finite_ok = True
    for key in ["beta_c", "mean_energy_per_spin", "free_energy_per_spin",
                "entropy_per_spin"]:
        v = result[key]
        if not np.isfinite(v):
            finite_ok = False
            warnings.append(f"{key} is not finite: {v}")
    checks.append({
        "name": "values_finite",
        "pass": finite_ok,
        "detail": "All scalar values are finite." if finite_ok else "Some values are NaN/inf."
    })

    # ── Check 3: β_c in physically reasonable range ───────────────────
    bc = result["beta_c"]
    bc_ok = 0.15 < bc < 0.40
    checks.append({
        "name": "beta_c_range",
        "pass": bc_ok,
        "detail": f"β_c = {bc:.6f}, expected near {beta_c_exact:.6f}."
    })

    # ── Check 4: β_c accuracy (within 5% of exact) ───────────────────
    bc_error_pct = abs(bc - beta_c_exact) / beta_c_exact * 100
    bc_accurate = bc_error_pct < 5.0
    checks.append({
        "name": "beta_c_accuracy",
        "pass": bc_accurate,
        "detail": f"β_c relative error = {bc_error_pct:.2f}% (threshold: 5%)."
    })
    if not bc_accurate:
        warnings.append(f"β_c estimate {bc:.6f} deviates {bc_error_pct:.2f}% from exact.")

    # ── Check 5: Energy per spin domain ───────────────────────────────
    # For triangular lattice (z=6), energy per spin u = -⟨Σ σ_iσ_j⟩/N
    # Range: u ∈ [-3, 0] (3 edges per spin, each contributing -1 to +1)
    u = result["mean_energy_per_spin"]
    u_ok = -3.5 < u < 0.5
    checks.append({
        "name": "energy_domain",
        "pass": u_ok,
        "detail": f"u = {u:.6f}, expected in [-3, 0]."
    })

    # ── Check 6: Energy plausibility at β_c ───────────────────────────
    # At β_c, the system is partially ordered; u should be between
    # the ground state (-3) and disordered state (0), roughly [-2.5, -1.0]
    u_plausible = -2.5 < u < -1.0
    checks.append({
        "name": "energy_plausibility",
        "pass": u_plausible,
        "detail": f"u = {u:.6f} at β_c, expected roughly in [-2.5, -1.0]."
    })
    if not u_plausible:
        warnings.append(f"Energy {u:.6f} at β_c seems outside expected range.")

    # ── Check 7: Free energy domain ───────────────────────────────────
    # f = -(1/β) ln Z / N. At β_c ≈ 0.275, and Z > 1, f should be negative.
    # At β=0: f = -T*ln2 = -(1/β)*ln2 → -∞. At β→∞: f → u_gs = -3.
    # At β_c: f should be somewhere in [-4, -2]
    f = result["free_energy_per_spin"]
    f_ok = -5.0 < f < -1.0
    checks.append({
        "name": "free_energy_domain",
        "pass": f_ok,
        "detail": f"f = {f:.6f}, expected in [-5, -1]."
    })

    # ── Check 8: Entropy per spin domain ──────────────────────────────
    # s = β(u - f). Must be non-negative (s ≥ 0).
    # At β=0: s = ln2 ≈ 0.693. At β→∞: s → 0.
    # At β_c: s should be in (0, ln2)
    s = result["entropy_per_spin"]
    s_ok = -0.01 < s < np.log(2) + 0.05
    checks.append({
        "name": "entropy_domain",
        "pass": s_ok,
        "detail": f"s = {s:.6f}, expected in [0, ln2 ≈ 0.693]."
    })

    # ── Check 9: Thermodynamic consistency: s = β(u - f) ─────────────
    s_check = bc * (u - f)
    s_consistent = abs(s - s_check) < 0.05
    checks.append({
        "name": "thermodynamic_consistency",
        "pass": s_consistent,
        "detail": f"s = {s:.6f}, β(u-f) = {s_check:.6f}, diff = {abs(s - s_check):.6f}."
    })

    # ── Check 10: f < u (free energy ≤ energy, since -Ts ≤ 0) ────────
    f_le_u = f < u + 0.01
    checks.append({
        "name": "free_energy_le_energy",
        "pass": f_le_u,
        "detail": f"f = {f:.6f} should be ≤ u = {u:.6f} (since -Ts ≤ 0)."
    })

    # ── Check 11: Finite-size data arrays consistent ──────────────────
    sizes = result["sizes_used"]
    fv = result["f_values"]
    uv = result["u_values"]
    sv = result["s_values"]
    arrays_ok = len(fv) == len(sizes) and len(uv) == len(sizes) and len(sv) == len(sizes)
    checks.append({
        "name": "array_lengths_consistent",
        "pass": arrays_ok,
        "detail": f"sizes={len(sizes)}, f={len(fv)}, u={len(uv)}, s={len(sv)}."
    })

    # ── Check 12: Finite-size data all finite ─────────────────────────
    all_data_finite = all(np.isfinite(v) for v in fv + uv + sv)
    checks.append({
        "name": "finite_size_data_finite",
        "pass": all_data_finite,
        "detail": "All finite-size data points are finite." if all_data_finite
                  else "Some finite-size data points are NaN/inf."
    })

    # ── Check 13: Monotonic trend in energy (should approach bulk) ────
    # Energy magnitude should decrease (less negative) as L increases
    if len(uv) >= 3:
        u_trend_ok = uv[-1] > uv[0]  # less negative for larger L
        checks.append({
            "name": "energy_monotonic_trend",
            "pass": u_trend_ok,
            "detail": f"u[L_min]={uv[0]:.6f}, u[L_max]={uv[-1]:.6f}. "
                      "Should trend toward bulk (less negative)."
        })

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
        "sampling_solver",
        "/home/elsafelts/Bachelor thesis/Quantum Chemistry/triangular_ising_pbc_solver_sampling.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.solver()

    report = verify(result)
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT (Sampling)")
    print("=" * 60)
    print(f"Overall: {'PASS' if report['pass'] else 'FAIL'}")
    for c in report["checks"]:
        status = "✓" if c["pass"] else "✗"
        print(f"  {status} {c['name']}: {c['detail']}")
    if report["warnings"]:
        print(f"\nWarnings: {report['warnings']}")
