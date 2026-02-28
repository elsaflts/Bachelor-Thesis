def verify(result):
    """
    Verify inference results for the 2D triangular-lattice Ising model.
    Checks schema, finiteness, domain constraints, and plausibility.
    """
    import math

    checks = []
    missing = []
    warnings = []

    # 1) Required keys
    required_keys = ["f_infty_est", "u_infty_est"]
    optional_keys = ["beta_c", "sizes_used", "f_values", "u_values",
                     "f_fit_residual_std", "u_fit_residual_std"]

    for k in required_keys:
        if k not in result:
            missing.append(k)

    if missing:
        return {
            "pass": False,
            "checks": checks,
            "missing": missing,
            "warnings": warnings
        }

    # Check for error key
    if "error" in result:
        checks.append({
            "name": "no_error",
            "pass": False,
            "detail": f"Solver returned error: {result['error']}"
        })
        return {
            "pass": False,
            "checks": checks,
            "missing": missing,
            "warnings": warnings
        }

    # 2) Type checks
    for k in required_keys:
        v = result[k]
        ok = isinstance(v, (int, float))
        checks.append({
            "name": f"{k}_type",
            "pass": ok,
            "detail": f"{k} is {type(v).__name__}, expected numeric"
        })

    # 3) Finiteness
    for k in required_keys:
        v = result[k]
        if not isinstance(v, (int, float)):
            continue
        finite = not (math.isnan(v) or math.isinf(v))
        checks.append({
            "name": f"{k}_finite",
            "pass": finite,
            "detail": f"{k} = {v}"
        })

    f = result["f_infty_est"]
    u = result["u_infty_est"]

    # 4) Domain / support checks

    # f_infty should be > ln(2) ≈ 0.693 (since f(0)=ln2 and increases with β)
    ln2 = math.log(2)
    f_lower_ok = f > ln2 * 0.9  # allow 10% slack for VI bias
    checks.append({
        "name": "f_lower_bound",
        "pass": f_lower_ok,
        "detail": f"f_infty={f:.6f}, expected > ~{ln2:.4f} (ln 2)"
    })

    # f_infty should be < ~1.5 (well below theoretical maximum)
    f_upper_ok = f < 1.5
    checks.append({
        "name": "f_upper_bound",
        "pass": f_upper_ok,
        "detail": f"f_infty={f:.6f}, expected < 1.5"
    })

    # u_infty should be positive (ferromagnetic, u = <Σ σσ>/S > 0)
    u_pos_ok = u > 0
    checks.append({
        "name": "u_positive",
        "pass": u_pos_ok,
        "detail": f"u_infty={u:.6f}, expected > 0 (ferromagnetic)"
    })

    # u_infty should be < 3 (max 3 bonds/site in bulk, max correlation 1)
    u_upper_ok = u < 3.5
    checks.append({
        "name": "u_upper_bound",
        "pass": u_upper_ok,
        "detail": f"u_infty={u:.6f}, expected < 3.5"
    })

    # 5) Plausibility checks (soft, generate warnings only)

    # f_infty plausible range: ~[0.85, 0.92] from known exact results
    if not (0.7 < f < 1.1):
        warnings.append(
            f"f_infty={f:.6f} outside plausible range [0.7, 1.1]; "
            f"exact bulk value is approximately 0.87-0.88"
        )

    # u_infty plausible range: ~[1.5, 2.2] at criticality
    if not (1.0 < u < 2.5):
        warnings.append(
            f"u_infty={u:.6f} outside plausible range [1.0, 2.5]; "
            f"expected approximately 1.8-1.9 at criticality"
        )

    # Check beta_c if present
    if "beta_c" in result:
        bc = result["beta_c"]
        bc_expected = 0.25 * math.log(3)
        bc_ok = abs(bc - bc_expected) < 1e-6
        checks.append({
            "name": "beta_c_correct",
            "pass": bc_ok,
            "detail": f"beta_c={bc:.10f}, expected {bc_expected:.10f}"
        })

    # Check f_values monotonicity if present (f_N should increase with N)
    if "f_values" in result and "sizes_used" in result:
        fv = result["f_values"]
        if len(fv) >= 3:
            # Check if roughly increasing (allow small violations from noise)
            increasing = sum(1 for i in range(len(fv)-1) if fv[i+1] > fv[i])
            mostly_increasing = increasing >= len(fv) // 2
            if not mostly_increasing:
                warnings.append(
                    f"f_values not consistently increasing with N: {fv}"
                )

    # Check u_values monotonicity (u_N should increase with N towards u_∞)
    if "u_values" in result and "sizes_used" in result:
        uv = result["u_values"]
        if len(uv) >= 3:
            increasing = sum(1 for i in range(len(uv)-1) if uv[i+1] > uv[i])
            mostly_increasing = increasing >= len(uv) // 2
            if not mostly_increasing:
                warnings.append(
                    f"u_values not consistently increasing with N: {uv}"
                )

    # Check fit residuals if present
    for key in ["f_fit_residual_std", "u_fit_residual_std"]:
        if key in result:
            r = result[key]
            if isinstance(r, (int, float)) and r > 0.1:
                warnings.append(
                    f"{key}={r:.4f} is large, extrapolation may be unreliable"
                )

    all_pass = all(c["pass"] for c in checks) and len(missing) == 0

    return {
        "pass": all_pass,
        "checks": checks,
        "missing": missing,
        "warnings": warnings
    }
