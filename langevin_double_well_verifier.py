def verify(result: dict) -> dict:
    """
    Verifier for the Overdamped Langevin Double-Well inference outputs.

    Checks:
      1. Bundle-level structure (result is a dict, strategy_outputs present,
         S1/S2/S3 all present).
      2. Per-strategy required keys exist and are correct types.
      3. All numerical values are finite (no NaN/inf).
      4. Domain/support constraints:
           - n_samples >= 1,000,000
           - empirical_mean is 2D and finite
           - covariance is 2×2, PSD, variances > 0
           - Var(y) close to theoretical sigma^2/2 = 0.245
           - KDE density is non-negative
           - R-hat in [1, inf) — values > 1.1 warn non-convergence
           - acceptance_rate in [0, 1] (S2/S3 only)
           - ESS values > 0
           - energy errors finite and non-negative (S3 only)
      5. All requested quantities of interest are present.
    """
    import numpy as np
    import math

    # ---------------------------------------------------------------------- #
    # Problem constants
    # ---------------------------------------------------------------------- #
    SIGMA           = 0.7
    # y-marginal: pi_y(y) proportional to exp(-2y^2/sigma^2) = N(0, sigma^2/4)
    # Var(y) = sigma^2/4 = T/kappa where T=sigma^2/2, kappa=d^2V/dy^2=2
    THEORY_VAR_Y    = SIGMA**2 / 4.0          # = 0.1225 (exact, y is Gaussian)
    VAR_Y_TOL       = 0.04                    # absolute tolerance on Var(y)
    MEAN_TOL        = 0.5                     # soft |mean| tolerance
    MIN_SAMPLES     = 1_000_000               # hard requirement from problem
    RHAT_WARN       = 1.10                    # R-hat warning threshold
    RHAT_HARD_FAIL  = 5.00                    # R-hat hard-fail (clearly diverged)

    # Required keys common to all strategies
    COMMON_REQUIRED = [
        "strategy_id",
        "n_samples",
        "empirical_mean",
        "empirical_covariance",
        "kde_bandwidth_scott",
        "kde_grid_x",
        "kde_grid_y",
        "kde_density",
        "gelman_rubin_x",
        "gelman_rubin_y",
        "ess_per_chain_x",
        "ess_per_chain_y",
        "total_ess_x",
        "total_ess_y",
        "inter_well_transitions",
        "simulation_time_s",
        "throughput_samples_per_s",
    ]

    # Extra required keys per strategy
    STRATEGY_EXTRA = {
        "S1": ["dt"],
        "S2": ["acceptance_rate", "tuned_dt"],
        "S3": ["acceptance_rate", "tuned_eps", "leapfrog_steps_L",
               "mean_energy_error", "max_energy_error",
               "total_grad_evaluations", "ess_per_grad_eval_x"],
    }

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #
    def is_finite_scalar(v):
        """True iff v is a numeric, finite scalar."""
        try:
            return math.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    def _pass(msg):
        return f"PASS: {msg}"

    def _fail(msg):
        return f"FAIL: {msg}"

    def _warn(msg):
        return f"WARN: {msg}"

    # Accumulators
    bundle_checks = []
    bundle_missing = []
    bundle_warnings = []
    per_strategy = {}
    overall_pass = True
    top_error = None

    # ====================================================================== #
    # 1. Bundle-level structure checks
    # ====================================================================== #
    if not isinstance(result, dict):
        return {
            "pass": False,
            "per_strategy": {},
            "checks": [_fail("result is not a dict")],
            "missing": ["result dict"],
            "warnings": [],
            "error": "result is not a dict"
        }
    bundle_checks.append(_pass("result is a dict"))

    if "strategy_outputs" not in result:
        overall_pass = False
        top_error = "'strategy_outputs' key missing"
        bundle_checks.append(_fail("'strategy_outputs' key missing"))
        bundle_missing.append("strategy_outputs")
        return {
            "pass": False,
            "per_strategy": {},
            "checks": bundle_checks,
            "missing": bundle_missing,
            "warnings": bundle_warnings,
            "error": top_error
        }
    bundle_checks.append(_pass("'strategy_outputs' key present"))

    strat_out = result["strategy_outputs"]
    if not isinstance(strat_out, dict):
        overall_pass = False
        top_error = "'strategy_outputs' is not a dict"
        bundle_checks.append(_fail(top_error))
        return {
            "pass": False,
            "per_strategy": {},
            "checks": bundle_checks,
            "missing": bundle_missing,
            "warnings": bundle_warnings,
            "error": top_error
        }
    bundle_checks.append(_pass("'strategy_outputs' is a dict"))

    for sid in ["S1", "S2", "S3"]:
        if sid not in strat_out:
            overall_pass = False
            bundle_checks.append(_fail(f"strategy '{sid}' missing from strategy_outputs"))
            bundle_missing.append(sid)
        else:
            bundle_checks.append(_pass(f"strategy '{sid}' present in strategy_outputs"))

    # ====================================================================== #
    # 2. Per-strategy checks
    # ====================================================================== #
    for sid in ["S1", "S2", "S3"]:
        if sid not in strat_out:
            per_strategy[sid] = {
                "pass": False,
                "checks": [_fail("Strategy output missing entirely")],
                "warnings": []
            }
            continue

        s = strat_out[sid]
        s_checks  = []
        s_warnings = []
        s_pass    = True

        # ── 2a. Error key check ─────────────────────────────────────────── #
        if isinstance(s, dict) and "error" in s:
            s_pass = False
            overall_pass = False
            s_checks.append(_fail(f"Solver returned error: {s['error']}"))
            per_strategy[sid] = {"pass": False, "checks": s_checks, "warnings": s_warnings}
            continue

        if not isinstance(s, dict):
            s_pass = False
            overall_pass = False
            s_checks.append(_fail("Strategy output is not a dict"))
            per_strategy[sid] = {"pass": False, "checks": s_checks, "warnings": s_warnings}
            continue

        # ── 2b. Required key presence ───────────────────────────────────── #
        all_required = COMMON_REQUIRED + STRATEGY_EXTRA.get(sid, [])
        for key in all_required:
            if key not in s:
                s_pass = False
                s_checks.append(_fail(f"Required key '{key}' missing"))
                bundle_missing.append(f"{sid}.{key}")
            else:
                s_checks.append(_pass(f"Key '{key}' present"))

        # ── 2c. strategy_id matches ──────────────────────────────────────── #
        if "strategy_id" in s:
            if s["strategy_id"] != sid:
                s_pass = False
                s_checks.append(
                    _fail(f"strategy_id mismatch: expected '{sid}', got '{s['strategy_id']}'")
                )
            else:
                s_checks.append(_pass(f"strategy_id='{sid}' matches"))

        # ── 2d. n_samples >= 1,000,000 ──────────────────────────────────── #
        if "n_samples" in s:
            n = s["n_samples"]
            if not is_finite_scalar(n):
                s_pass = False
                s_checks.append(_fail(f"n_samples={n!r} is not a finite scalar"))
            elif int(n) < MIN_SAMPLES:
                s_pass = False
                s_checks.append(
                    _fail(f"n_samples={int(n):,} < {MIN_SAMPLES:,} (problem requirement)")
                )
            else:
                s_checks.append(_pass(f"n_samples={int(n):,} >= {MIN_SAMPLES:,}"))

        # ── 2e. empirical_mean: finite 2-vector, soft |mean| tolerance ───── #
        if "empirical_mean" in s:
            m = s["empirical_mean"]
            try:
                mv = [float(m[0]), float(m[1])]
                if len(m) != 2:
                    raise ValueError(f"length={len(m)}")
                if not all(math.isfinite(v) for v in mv):
                    s_pass = False
                    s_checks.append(_fail(f"empirical_mean contains non-finite values: {mv}"))
                else:
                    s_checks.append(
                        _pass(f"empirical_mean={[round(v,4) for v in mv]} is finite and 2D")
                    )
                    if abs(mv[0]) > MEAN_TOL:
                        s_warnings.append(
                            _warn(
                                f"mean(x)={mv[0]:.4f} deviates from expected 0 by >{MEAN_TOL}; "
                                "possible non-convergence or initialisation bias"
                            )
                        )
                    if abs(mv[1]) > MEAN_TOL:
                        s_warnings.append(
                            _warn(f"mean(y)={mv[1]:.4f} deviates from expected 0 by >{MEAN_TOL}")
                        )
            except Exception as exc:
                s_pass = False
                s_checks.append(_fail(f"empirical_mean invalid ({exc})"))

        # ── 2f. empirical_covariance: 2×2, finite, PSD, Var(y) check ─────── #
        if "empirical_covariance" in s:
            try:
                cov_arr = np.array(s["empirical_covariance"], dtype=float)
                if cov_arr.shape != (2, 2):
                    s_pass = False
                    s_checks.append(
                        _fail(f"empirical_covariance shape={cov_arr.shape}, expected (2,2)")
                    )
                elif not np.all(np.isfinite(cov_arr)):
                    s_pass = False
                    s_checks.append(_fail("empirical_covariance contains non-finite values"))
                else:
                    s_checks.append(_pass("empirical_covariance is 2×2 and finite"))

                    # Variances must be strictly positive
                    for i, var_name in enumerate(["Var(x)", "Var(y)"]):
                        if cov_arr[i, i] <= 0:
                            s_pass = False
                            s_checks.append(
                                _fail(f"{var_name}={cov_arr[i,i]:.6f} <= 0 (invalid variance)")
                            )
                        else:
                            s_checks.append(_pass(f"{var_name}={cov_arr[i,i]:.6f} > 0"))

                    # Positive semi-definite check via determinant
                    det = float(np.linalg.det(cov_arr))
                    if det < -1e-10:
                        s_pass = False
                        s_checks.append(
                            _fail(f"Covariance matrix not PSD (det={det:.6f} < 0)")
                        )
                    else:
                        s_checks.append(
                            _pass(f"Covariance matrix is PSD (det={det:.6f} >= 0)")
                        )

                    # Symmetry check
                    if abs(cov_arr[0, 1] - cov_arr[1, 0]) > 1e-8:
                        s_warnings.append(
                            _warn(
                                f"Covariance not symmetric: "
                                f"cov[0,1]={cov_arr[0,1]:.6f} vs cov[1,0]={cov_arr[1,0]:.6f}"
                            )
                        )

                    # Var(y) should be close to sigma^2/2 = 0.245 (y is Gaussian)
                    var_y = cov_arr[1, 1]
                    if abs(var_y - THEORY_VAR_Y) > VAR_Y_TOL:
                        s_warnings.append(
                            _warn(
                                f"Var(y)={var_y:.4f} deviates from theory "
                                f"sigma^2/2={THEORY_VAR_Y:.4f} by >{VAR_Y_TOL:.2f}"
                            )
                        )
                    else:
                        s_checks.append(
                            _pass(
                                f"Var(y)={var_y:.4f} within {VAR_Y_TOL} of "
                                f"theoretical sigma^2/2={THEORY_VAR_Y:.4f}"
                            )
                        )

                    # x and y should be approximately independent (Cov(x,y) ~ 0)
                    if abs(cov_arr[0, 1]) > 0.1:
                        s_warnings.append(
                            _warn(
                                f"Cov(x,y)={cov_arr[0,1]:.4f} is unexpectedly large; "
                                "x and y are independent under pi"
                            )
                        )

            except Exception as exc:
                s_pass = False
                s_checks.append(_fail(f"empirical_covariance invalid ({exc})"))

        # ── 2g. KDE bandwidth: positive finite scalar ────────────────────── #
        if "kde_bandwidth_scott" in s:
            bw = s["kde_bandwidth_scott"]
            if not is_finite_scalar(bw) or float(bw) <= 0:
                s_pass = False
                s_checks.append(
                    _fail(f"kde_bandwidth_scott={bw!r} must be a positive finite scalar")
                )
            else:
                s_checks.append(_pass(f"kde_bandwidth_scott={float(bw):.4f} > 0"))

        # ── 2h. KDE grids: finite 1D arrays, correct range ───────────────── #
        for grid_key, expected_range in [
            ("kde_grid_x", (-2.5, 2.5)),
            ("kde_grid_y", (-2.2, 2.2))
        ]:
            if grid_key in s:
                try:
                    g = np.array(s[grid_key], dtype=float)
                    if g.ndim != 1 or len(g) < 2:
                        s_pass = False
                        s_checks.append(
                            _fail(f"{grid_key} must be a 1D array with >= 2 points")
                        )
                    elif not np.all(np.isfinite(g)):
                        s_pass = False
                        s_checks.append(_fail(f"{grid_key} contains non-finite values"))
                    else:
                        s_checks.append(
                            _pass(f"{grid_key} has {len(g)} finite points, "
                                  f"range=[{g.min():.2f}, {g.max():.2f}]")
                        )
                        lo, hi = expected_range
                        if g.min() > lo + 0.5 or g.max() < hi - 0.5:
                            s_warnings.append(
                                _warn(
                                    f"{grid_key} range [{g.min():.2f},{g.max():.2f}] "
                                    f"may not cover the support adequately"
                                )
                            )
                except Exception as exc:
                    s_pass = False
                    s_checks.append(_fail(f"{grid_key} invalid ({exc})"))

        # ── 2i. KDE density: non-negative, finite 2D array ───────────────── #
        if "kde_density" in s:
            try:
                zz = np.array(s["kde_density"], dtype=float)
                if zz.ndim != 2:
                    s_pass = False
                    s_checks.append(
                        _fail(f"kde_density must be a 2D array, got {zz.ndim}D")
                    )
                elif not np.all(np.isfinite(zz)):
                    s_pass = False
                    n_bad = int(np.sum(~np.isfinite(zz)))
                    s_checks.append(
                        _fail(f"kde_density has {n_bad} non-finite values")
                    )
                elif np.any(zz < -1e-12):
                    s_pass = False
                    s_checks.append(
                        _fail(
                            f"kde_density has negative values (min={zz.min():.4e}); "
                            "density must be >= 0"
                        )
                    )
                else:
                    s_checks.append(
                        _pass(
                            f"kde_density is 2D, finite, non-negative; "
                            f"shape={zz.shape}, max={zz.max():.4f}"
                        )
                    )
                    if float(zz.max()) == 0.0:
                        s_warnings.append(_warn("kde_density is identically zero"))
            except Exception as exc:
                s_pass = False
                s_checks.append(_fail(f"kde_density invalid ({exc})"))

        # ── 2j. Gelman-Rubin R-hat ───────────────────────────────────────── #
        for rhat_key in ["gelman_rubin_x", "gelman_rubin_y"]:
            if rhat_key in s:
                rh = s[rhat_key]
                if not is_finite_scalar(rh):
                    s_pass = False
                    s_checks.append(_fail(f"{rhat_key}={rh!r} is not finite"))
                elif float(rh) < 1.0 - 1e-4:
                    s_pass = False
                    s_checks.append(
                        _fail(
                            f"{rhat_key}={float(rh):.4f} < 1.0 (R-hat cannot be < 1; "
                            "implementation error)"
                        )
                    )
                elif float(rh) > RHAT_HARD_FAIL:
                    s_pass = False
                    s_checks.append(
                        _fail(
                            f"{rhat_key}={float(rh):.4f} > {RHAT_HARD_FAIL} "
                            "(chain has clearly diverged)"
                        )
                    )
                elif float(rh) > RHAT_WARN:
                    s_checks.append(
                        _warn(
                            f"{rhat_key}={float(rh):.4f} > {RHAT_WARN} "
                            "(non-convergence; consider longer chains or larger dt)"
                        )
                    )
                    s_warnings.append(
                        _warn(f"{rhat_key}={float(rh):.4f} exceeds convergence threshold {RHAT_WARN}")
                    )
                else:
                    s_checks.append(
                        _pass(f"{rhat_key}={float(rh):.4f} in acceptable range [1.0, {RHAT_WARN}]")
                    )

        # ── 2k. ESS values: positive and finite ──────────────────────────── #
        for ess_key in [
            "ess_per_chain_x", "ess_per_chain_y",
            "total_ess_x", "total_ess_y"
        ]:
            if ess_key in s:
                ev = s[ess_key]
                if not is_finite_scalar(ev):
                    s_pass = False
                    s_checks.append(_fail(f"{ess_key}={ev!r} is not finite"))
                elif float(ev) <= 0:
                    s_pass = False
                    s_checks.append(_fail(f"{ess_key}={float(ev):.2f} <= 0 (invalid ESS)"))
                else:
                    s_checks.append(_pass(f"{ess_key}={float(ev):.1f} > 0"))

        # ── 2l. Acceptance rate (S2/S3): in [0, 1] ──────────────────────── #
        if "acceptance_rate" in s:
            ar = s["acceptance_rate"]
            if not is_finite_scalar(ar):
                s_pass = False
                s_checks.append(_fail(f"acceptance_rate={ar!r} is not finite"))
            else:
                ar_f = float(ar)
                if ar_f < 0.0 or ar_f > 1.0:
                    s_pass = False
                    s_checks.append(
                        _fail(f"acceptance_rate={ar_f:.4f} not in [0, 1]")
                    )
                else:
                    s_checks.append(_pass(f"acceptance_rate={ar_f:.4f} in [0, 1]"))
                    if ar_f < 0.05:
                        s_warnings.append(
                            _warn(
                                f"acceptance_rate={ar_f:.4f} extremely low; "
                                "step size may be too large"
                            )
                        )
                    if ar_f > 0.99:
                        s_warnings.append(
                            _warn(
                                f"acceptance_rate={ar_f:.4f} extremely high; "
                                "step size may be too small (low ESS)"
                            )
                        )

        # ── 2m. inter_well_transitions: non-negative integer ─────────────── #
        if "inter_well_transitions" in s:
            ntr = s["inter_well_transitions"]
            if not is_finite_scalar(ntr) or float(ntr) < 0:
                s_pass = False
                s_checks.append(_fail(f"inter_well_transitions={ntr!r} is invalid"))
            else:
                s_checks.append(_pass(f"inter_well_transitions={int(ntr)}"))
                if int(ntr) == 0:
                    s_warnings.append(
                        _warn(
                            "inter_well_transitions=0; chain may be entirely trapped; "
                            "distribution coverage is suspect"
                        )
                    )

        # ── 2n. simulation_time_s / throughput: positive finite ──────────── #
        for timing_key in ["simulation_time_s", "throughput_samples_per_s"]:
            if timing_key in s:
                tv = s[timing_key]
                if not is_finite_scalar(tv) or float(tv) <= 0:
                    s_pass = False
                    s_checks.append(_fail(f"{timing_key}={tv!r} must be a positive finite scalar"))
                else:
                    s_checks.append(_pass(f"{timing_key}={float(tv):.4g}"))

        # ── 2o. S3-specific: energy errors finite and non-negative ───────── #
        if sid == "S3":
            for err_key in ["mean_energy_error", "max_energy_error"]:
                if err_key in s:
                    ev = s[err_key]
                    if not is_finite_scalar(ev):
                        s_pass = False
                        s_checks.append(
                            _fail(
                                f"{err_key}={ev!r} is not finite — "
                                "leapfrog overflow detected"
                            )
                        )
                    elif float(ev) < 0:
                        s_pass = False
                        s_checks.append(
                            _fail(f"{err_key}={float(ev):.4f} < 0 (impossible for energy error)")
                        )
                    else:
                        s_checks.append(
                            _pass(f"{err_key}={float(ev):.4f} is finite and non-negative")
                        )
                        if float(ev) > 1.0:
                            s_warnings.append(
                                _warn(
                                    f"{err_key}={float(ev):.4f} > 1.0; "
                                    "leapfrog may be near instability threshold"
                                )
                            )

            # ess_per_grad_eval_x: finite, positive
            if "ess_per_grad_eval_x" in s:
                eg = s["ess_per_grad_eval_x"]
                if not is_finite_scalar(eg) or float(eg) <= 0:
                    s_pass = False
                    s_checks.append(
                        _fail(f"ess_per_grad_eval_x={eg!r} must be positive finite")
                    )
                else:
                    s_checks.append(_pass(f"ess_per_grad_eval_x={float(eg):.6f} > 0"))

        # ── Finalise strategy ────────────────────────────────────────────── #
        if not s_pass:
            overall_pass = False

        per_strategy[sid] = {
            "pass"    : s_pass,
            "checks"  : s_checks,
            "warnings": s_warnings,
        }

    # Collect all warnings at bundle level for convenience
    all_warnings = list(bundle_warnings)
    for sid, sv in per_strategy.items():
        all_warnings.extend(sv.get("warnings", []))

    return {
        "pass"        : overall_pass,
        "per_strategy": per_strategy,
        "checks"      : bundle_checks,
        "missing"     : bundle_missing,
        "warnings"    : all_warnings,
        "error"       : top_error,
    }


# --------------------------------------------------------------------------- #
# Self-test: build a combined result dict from the three solver files and run  #
# the verifier against it.                                                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import importlib.util, json, sys, os

    base = os.path.dirname(os.path.abspath(__file__))
    strategy_outputs = {}

    for sid, fname in [
        ("S1", "langevin_double_well_solver_S1.py"),
        ("S2", "langevin_double_well_solver_S2.py"),
        ("S3", "langevin_double_well_solver_S3.py"),
    ]:
        path = os.path.join(base, fname)
        spec  = importlib.util.spec_from_file_location(f"solver_{sid}", path)
        mod   = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"Running {sid} solver ...", flush=True)
        strategy_outputs[sid] = mod.solver()
        print(f"  {sid} done: n_samples={strategy_outputs[sid].get('n_samples','?')}")

    combined = {"strategy_outputs": strategy_outputs}
    report   = verify(combined)

    # Pretty-print everything except the large grid arrays
    def slim(d):
        if isinstance(d, dict):
            return {k: slim(v) for k, v in d.items() if k not in
                    ("kde_density", "kde_grid_x", "kde_grid_y")}
        return d

    print("\n" + "="*70)
    print("VERIFICATION REPORT")
    print("="*70)
    print(json.dumps(slim(report), indent=2))
    sys.exit(0 if report["pass"] else 1)
