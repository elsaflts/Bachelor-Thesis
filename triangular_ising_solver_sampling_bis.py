def solver():
    """
    Annealed-continuation TI for the 2D triangular-lattice Ising model.
    Fixes large-N upward drift via:
      A) Single annealed Markov chain per N (warm-start across β grid, no resets)
      B) Two-start mixing check near β_c with automatic retry
      C) Integration stability: K-point vs 2K-point trapz comparison
    """
    import numpy as np

    beta_c = 0.25 * np.log(3)   # exact critical β for 2D triangular lattice

    # ---------------------------------------------------------------
    # Lattice construction (identical to original)
    # ---------------------------------------------------------------
    def build_lattice(N):
        S = N * (N + 1) // 2
        def idx(r, c): return r * (r + 1) // 2 + c
        neighbors = [[] for _ in range(S)]
        edges_a, edges_b = [], []
        for r in range(N):
            for c in range(r + 1):
                i = idx(r, c)
                if c < r:
                    j = idx(r, c + 1)
                    neighbors[i].append(j); neighbors[j].append(i)
                    edges_a.append(i); edges_b.append(j)
                if r < N - 1:
                    j = idx(r + 1, c)
                    neighbors[i].append(j); neighbors[j].append(i)
                    edges_a.append(i); edges_b.append(j)
                if r < N - 1:
                    j = idx(r + 1, c + 1)
                    neighbors[i].append(j); neighbors[j].append(i)
                    edges_a.append(i); edges_b.append(j)
        ea = np.array(edges_a, dtype=np.int32)
        eb = np.array(edges_b, dtype=np.int32)
        max_nn = max((len(nb) for nb in neighbors), default=0)
        nn_arr = np.full((S, max_nn), -1, dtype=np.int32)
        nn_cnt = np.zeros(S, dtype=np.int32)
        for i in range(S):
            nn_cnt[i] = len(neighbors[i])
            for k in range(nn_cnt[i]):
                nn_arr[i, k] = neighbors[i][k]
        return S, ea, eb, nn_arr, nn_cnt

    # ---------------------------------------------------------------
    # Numba JIT (identical to original)
    # ---------------------------------------------------------------
    use_numba = False
    try:
        from numba import njit

        @njit
        def _wolff_mc(spins, nn_arr, nn_cnt, edges_a, edges_b,
                      p_add, S, n_equil, n_sample, n_skip):
            stack = np.empty(S, dtype=np.int32)
            in_cluster = np.zeros(S, dtype=np.bool_)
            energies = np.empty(n_sample, dtype=np.float64)
            n_edges = len(edges_a)
            si = 0; since_last = 0
            total_steps = n_equil + n_sample * n_skip
            for step in range(total_steps):
                seed = np.random.randint(0, S)
                sv = spins[seed]
                for x in range(S): in_cluster[x] = False
                in_cluster[seed] = True
                stack[0] = seed; sp = 1
                while sp > 0:
                    sp -= 1; site = stack[sp]
                    for k in range(nn_cnt[site]):
                        nb = nn_arr[site, k]
                        if (not in_cluster[nb] and spins[nb] == sv
                                and np.random.random() < p_add):
                            in_cluster[nb] = True
                            stack[sp] = nb; sp += 1
                for x in range(S):
                    if in_cluster[x]: spins[x] = -spins[x]
                if step >= n_equil:
                    since_last += 1
                    if since_last >= n_skip:
                        since_last = 0
                        E = 0.0
                        for e in range(n_edges):
                            E += spins[edges_a[e]] * spins[edges_b[e]]
                        energies[si] = E; si += 1
                        if si >= n_sample: break
            return energies[:si]

        _s = np.ones(3, dtype=np.int8)
        _nn = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32)
        _nc = np.array([2, 2, 2], dtype=np.int32)
        _ea = np.array([0, 0, 1], dtype=np.int32)
        _eb = np.array([1, 2, 2], dtype=np.int32)
        _wolff_mc(_s, _nn, _nc, _ea, _eb, 0.5, 3, 5, 5, 1)
        use_numba = True
        print("Using numba-accelerated Wolff cluster algorithm.")
    except Exception:
        print("Numba unavailable; using pure-Python Wolff cluster algorithm.")

    # ---------------------------------------------------------------
    # Tunable parameters (exposed near top)
    # ---------------------------------------------------------------
    if use_numba:
        sizes         = [10, 20, 30, 40, 60, 80, 100, 200, 300]
        K             = 24      # base grid; 2K points used for annealing
        n_equil_cold  = 5000    # equil steps for cold / first-β start
        n_equil_warm  = 1000    # equil steps for warm-start β transitions
        n_samp        = 10000
        thinning      = 3
    else:
        sizes         = [8, 12, 16, 20, 25, 30, 40, 50]
        K             = 16
        n_equil_cold  = 1500
        n_equil_warm  = 400
        n_samp        = 3000
        thinning      = 2

    n_check     = 4    # last n_check of 2K β pts get two-start mixing check
    max_retries = 2    # max doubling retries on mixing failure
    mix_sigma   = 3.0  # mixing threshold (units of combined SE)

    # ---------------------------------------------------------------
    # β grid: 2K points, quadratic spacing denser near β_c
    # β(t) = β_c · (2t − t²), t ∈ [0,1]  →  dβ/dt = 0 at t=1, so more points near β_c
    # ---------------------------------------------------------------
    def make_grid(n_pts):
        t = np.linspace(0.0, 1.0, n_pts)
        return beta_c * (2.0 * t - t ** 2)

    print(f"beta_c = {beta_c:.10f}")
    print(f"Grid: K={K} base, 2K={2*K} for annealing + stability check")
    print(f"System sizes N: {sizes}\n")

    # ---------------------------------------------------------------
    # MC helper: run Wolff on spins (modified in-place).
    # Returns (u_mean, u_sem, raw_E_total) where raw_E_total[i] = Σ_{<ij>} σ_i σ_j
    # at sample i (total, not per-site).
    # ---------------------------------------------------------------
    def mc_sample(spins, S, ea, eb, nn_arr, nn_cnt, beta, n_eq, n_sp):
        if beta < 1e-14:
            return 0.0, 0.0, np.zeros(n_sp)
        p_add = 1.0 - np.exp(-2.0 * beta)
        if use_numba:
            raw = _wolff_mc(spins, nn_arr, nn_cnt, ea, eb,
                            p_add, S, n_eq, n_sp, thinning)
        else:
            rng = np.random.default_rng()
            def wolff_step():
                seed = int(rng.integers(0, S))
                sv = int(spins[seed]); cluster = {seed}; stk = [seed]
                while stk:
                    site = stk.pop()
                    for ki in range(nn_cnt[site]):
                        nb = int(nn_arr[site, ki])
                        if nb not in cluster and spins[nb] == sv and rng.random() < p_add:
                            cluster.add(nb); stk.append(nb)
                for s in cluster: spins[s] = -spins[s]
            for _ in range(n_eq): wolff_step()
            raw = np.empty(n_sp)
            for i in range(n_sp):
                for _ in range(thinning): wolff_step()
                raw[i] = float(np.sum(spins[ea] * spins[eb]))
        u  = float(np.mean(raw)) / S
        se = float(np.std(raw, ddof=1) / np.sqrt(len(raw))) / S
        return u, se, raw

    # ---------------------------------------------------------------
    # Main loop over system sizes
    # ---------------------------------------------------------------
    f_data, u_data, c_data, diag_list = [], [], [], []

    for N in sizes:
        S, ea, eb, nn_arr, nn_cnt = build_lattice(N)
        n_edges    = len(ea)
        edge_per_S = n_edges / S
        print(f"N={N:3d}  S={S:5d}  edges={n_edges:6d} ...", end="", flush=True)

        betas   = make_grid(2 * K)   # 2K β points for this N
        u_arr   = np.zeros(2 * K)
        se_arr  = np.zeros(2 * K)
        last_raw     = None
        mixing_checks = []

        # A) Random initialisation — no ordered-state bias at any β
        spins = np.random.choice(np.array([-1, 1], dtype=np.int8), size=S)

        for k in range(2 * K):
            beta_k = betas[k]
            n_eq   = n_equil_cold if k == 0 else n_equil_warm

            if k >= 2 * K - n_check:
                # B) Two-start mixing check
                # Save spin state before warm-start run so we can restore on retry.
                spins_saved = spins.copy()
                u_w = se_w = raw_w = None

                for retry in range(max_retries + 1):
                    spins[:] = spins_saved          # restore for (re)try
                    n_eq_r   = n_eq * (2 ** retry)  # double equil each retry
                    u_w, se_w, raw_w = mc_sample(
                        spins, S, ea, eb, nn_arr, nn_cnt, beta_k, n_eq_r, n_samp)
                    # spins now holds final warm-start state for this retry

                    spins_fresh = np.random.choice(
                        np.array([-1, 1], dtype=np.int8), size=S)
                    u_r, se_r, _ = mc_sample(
                        spins_fresh, S, ea, eb, nn_arr, nn_cnt,
                        beta_k, n_equil_cold, n_samp)

                    diff   = abs(u_w - u_r)
                    tol    = mix_sigma * np.sqrt(se_w ** 2 + se_r ** 2)
                    passed = diff <= tol

                    if passed or retry == max_retries:
                        mixing_checks.append({
                            "beta": float(beta_k), "pass": passed,
                            "diff": float(diff), "tol": float(tol),
                            "retries": retry,
                        })
                        break
                # After loop, spins = final warm-start state (best equil)

                # Sanity: u ∈ [0, edge_per_S]
                if not (0.0 <= u_w <= edge_per_S + 1e-9):
                    print(f"\nWARN: u={u_w:.4f} out of [0, {edge_per_S:.3f}] at β={beta_k:.4f}")
                u_w = float(np.clip(u_w, 0.0, edge_per_S))

            else:
                u_w, se_w, raw_w = mc_sample(
                    spins, S, ea, eb, nn_arr, nn_cnt, beta_k, n_eq, n_samp)

                # Sanity: u ∈ [0, edge_per_S]
                if not (0.0 <= u_w <= edge_per_S + 1e-9):
                    print(f"\nWARN: u={u_w:.4f} out of [0, {edge_per_S:.3f}] at β={beta_k:.4f}")
                u_w = float(np.clip(u_w, 0.0, edge_per_S))

            u_arr[k]  = u_w
            se_arr[k] = se_w
            if k == 2 * K - 1:
                last_raw = raw_w   # save energy samples at β_c for c_N

        # C) Integration stability: f using all 2K pts vs every-other K pts
        f_2K    = np.log(2.0) + np.trapezoid(u_arr,      betas)
        f_K     = np.log(2.0) + np.trapezoid(u_arr[::2], betas[::2])
        delta_f = abs(f_K - f_2K)
        if delta_f > 1e-3:
            print(f"\nWARN: integration unstable Δf={delta_f:.4f} for N={N}")

        # E) Runaway check
        if len(f_data) >= 2 and f_2K > f_data[-1] + 0.1:
            print(f"\nWARN: f_N growing: f_N={f_2K:.5f}, prev={f_data[-1]:.5f}")

        # u_N at β_c (last β point)
        u_N = float(u_arr[-1])

        # c_N = β_c² · Var(E) / S  from raw energy samples at β_c
        if last_raw is not None and len(last_raw) > 1:
            E   = last_raw   # total Σ σ_i σ_j per sample
            c_N = beta_c ** 2 * (float(np.mean(E ** 2)) - float(np.mean(E)) ** 2) / S
        else:
            c_N = 0.0
        c_N = max(0.0, c_N)   # E) c_N ≥ 0

        f_data.append(f_2K)
        u_data.append(u_N)
        c_data.append(c_N)

        mix_ok = all(mc["pass"] for mc in mixing_checks)
        diag_list.append({
            "N": N,
            "delta_f": float(delta_f),
            "mixing_pass": mix_ok,
            "mixing_checks": mixing_checks,
        })

        print(f" f_N={f_2K:.7f}  Δf={delta_f:.2e}  u_N={u_N:.6f}"
              f"  c_N={c_N:.4f}  mix={'OK' if mix_ok else 'FAIL'}")

    # ---------------------------------------------------------------
    # Finite-size extrapolation — paper free-boundary form
    # Triangle geometry: S = N(N+1)/2,  p(N) = 3N
    # ---------------------------------------------------------------
    sizes_arr = np.array(sizes, dtype=np.float64)
    f_arr     = np.array(f_data)
    u_arr_all = np.array(u_data)

    # Exclude sizes where mixing diagnostics failed
    mix_mask = np.array([d["mixing_pass"] for d in diag_list], dtype=bool)
    N_fit    = sizes_arr[mix_mask]
    u_fit    = u_arr_all[mix_mask]
    n_ok     = len(N_fit)

    S_fit = N_fit * (N_fit + 1.0) / 2.0
    p_fit = 3.0 * N_fit
    lnN   = np.log(N_fit)

    # ── u fit: Eq.(8) — full model (7 cols) with fallback to truncated (4 cols)
    # u(N) = u_∞ + a1*(p·lnN)/S + a2*(lnN)/S + a3/√S + a4/S + a5/S^(3/2) + a6/S²
    COND_MAX = 1e8

    X_full = np.column_stack([
        np.ones(n_ok),
        (p_fit * lnN) / S_fit,   # surface log  [Eq.(8) u_surf term]
        lnN / S_fit,             # corner log   [Eq.(8) u_corn term]
        1.0 / np.sqrt(S_fit),   # k=1
        1.0 / S_fit,             # k=2
        1.0 / S_fit ** 1.5,     # k=3
        1.0 / S_fit ** 2.0,     # k=4
    ])

    cond_full = float(np.linalg.cond(X_full)) if n_ok >= 7 else float("inf")
    use_full  = (n_ok >= 7 and cond_full < COND_MAX)

    # Truncated model (4 cols): always numerically safe
    # u(N) = u_∞ + a1*(p·lnN)/S + a2*(lnN)/S + a3/√S
    X_trunc = np.column_stack([
        np.ones(n_ok),
        (p_fit * lnN) / S_fit,
        lnN / S_fit,
        1.0 / np.sqrt(S_fit),
    ])

    if use_full:
        Xu, u_cond, u_model = X_full, cond_full, "full (7-col, k=1..4)"
    else:
        Xu, u_cond, u_model = X_trunc, float(np.linalg.cond(X_trunc)), "truncated (4-col, k=1)"
        if n_ok >= 7:
            print(f"\nWARN: full u-model ill-conditioned (cond={cond_full:.1e}); "
                  "falling back to truncated 4-col model.")

    cu, _, _, _ = np.linalg.lstsq(Xu, u_fit, rcond=None)
    u_infty    = float(cu[0])
    u_residual = float(np.std(u_fit - Xu @ cu))
    sizes_fit  = [int(n) for n in N_fit]

    # ── f fit: Eq.(5) — triangle has l_k = 0 (no higher-order log corrections)
    # f(N) = f_∞ + f_surf*p/S + f_corn*lnN/S + f_2/S + Σ f_k/S^(k/2)
    f_fit_arr = f_arr[mix_mask]
    S_fit_f   = N_fit * (N_fit + 1.0) / 2.0
    Xf = np.column_stack([
        np.ones(n_ok),
        p_fit / S_fit_f,         # surface (no log) [Eq.(5) f_surf term]
        lnN / S_fit_f,           # corner log       [Eq.(5) f_corn term]
        1.0 / S_fit_f,           # f_2 and f_3 combined at 1/S
        1.0 / np.sqrt(S_fit_f), # k=1
    ])
    cf, _, _, _ = np.linalg.lstsq(Xf, f_fit_arr, rcond=None)
    f_infty    = float(cf[0])
    f_residual = float(np.std(f_fit_arr - Xf @ cf))

    print()
    print("=" * 65)
    print("FINITE-SIZE EXTRAPOLATION  (paper free-boundary form)")
    print("=" * 65)
    print(f"  u model : {u_model}")
    print(f"  Sizes in fit ({n_ok}/{len(sizes)}): {sizes_fit}")
    print(f"  u_infty = {u_infty:.8f}   (known bulk u_∞ = 2.0)")
    print(f"  u residual std = {u_residual:.2e}   |   cond(Xu) = {u_cond:.2e}")
    print(f"  f_infty = {f_infty:.8f}   (known bulk f_∞ = 0.87958539)")
    print(f"  f residual std = {f_residual:.2e}")
    print("=" * 65)
    print("\nDiagnostics (integration stability and mixing):")
    for d in diag_list:
        mix_str = "OK" if d["mixing_pass"] else "FAIL"
        print(f"  N={d['N']:3d}  Δf={d['delta_f']:.2e}  mixing={mix_str}")

    result = {
        "f_infty_est":        f_infty,
        "u_infty_est":        u_infty,
        "beta_c":             float(beta_c),
        "sizes_used":         sizes,
        "sizes_fit":          sizes_fit,
        "u_model":            u_model,
        "f_values":           [float(v) for v in f_data],
        "u_values":           [float(v) for v in u_data],
        "c_values":           [float(v) for v in c_data],
        "f_fit_residual_std": f_residual,
        "u_fit_residual_std": u_residual,
        "u_design_cond":      u_cond,
        "diagnostics":        diag_list,
    }

    print(f"\nFinal result: {result}")
    return result


solver()
