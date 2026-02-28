def solver():
    """
    Sampling-based inference for the 2D triangular-lattice Ising model
    on a triangle-shaped patch with free boundary conditions.

    Strategy: Wolff Cluster Algorithm + Thermodynamic Integration
    - Wolff cluster updates to efficiently sample at/near criticality
    - Gauss-Legendre quadrature for integral f_N(β_c) = ln2 + ∫₀^{β_c} u_N(β') dβ'
    - Finite-size extrapolation: q(N) = q_∞ + a/N + b/N²
    """
    import numpy as np
    from numpy.polynomial.legendre import leggauss

    # Exact critical inverse temperature for 2D triangular lattice
    beta_c = 0.25 * np.log(3)  # ≈ 0.274653

    # -----------------------------------------------------------------
    # Lattice construction
    # -----------------------------------------------------------------
    def build_lattice(N):
        """
        Build triangle-shaped patch of the 2D triangular lattice.
        Row r (0-indexed) has (r+1) sites.  S = N(N+1)/2 total spins.
        Returns neighbour arrays and edge arrays for fast energy computation.
        """
        S = N * (N + 1) // 2

        def idx(r, c):
            return r * (r + 1) // 2 + c

        neighbors = [[] for _ in range(S)]
        edges_a, edges_b = [], []

        for r in range(N):
            for c in range(r + 1):
                i = idx(r, c)
                # Horizontal right
                if c < r:
                    j = idx(r, c + 1)
                    neighbors[i].append(j)
                    neighbors[j].append(i)
                    edges_a.append(i)
                    edges_b.append(j)
                # Lower-left diagonal
                if r < N - 1:
                    j = idx(r + 1, c)
                    neighbors[i].append(j)
                    neighbors[j].append(i)
                    edges_a.append(i)
                    edges_b.append(j)
                # Lower-right diagonal
                if r < N - 1:
                    j = idx(r + 1, c + 1)
                    neighbors[i].append(j)
                    neighbors[j].append(i)
                    edges_a.append(i)
                    edges_b.append(j)

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

    # -----------------------------------------------------------------
    # Try numba JIT for the hot loop
    # -----------------------------------------------------------------
    use_numba = False
    try:
        from numba import njit

        @njit
        def _wolff_mc(spins, nn_arr, nn_cnt, edges_a, edges_b,
                      p_add, S, n_equil, n_sample, n_skip):
            """Numba-accelerated Wolff cluster MC returning energy samples."""
            stack = np.empty(S, dtype=np.int32)
            in_cluster = np.zeros(S, dtype=np.bool_)
            energies = np.empty(n_sample, dtype=np.float64)
            n_edges = len(edges_a)
            si = 0
            since_last = 0
            total_steps = n_equil + n_sample * n_skip

            for step in range(total_steps):
                # --- Wolff cluster step ---
                seed = np.random.randint(0, S)
                sv = spins[seed]
                for x in range(S):
                    in_cluster[x] = False
                in_cluster[seed] = True
                stack[0] = seed
                sp = 1

                while sp > 0:
                    sp -= 1
                    site = stack[sp]
                    nc = nn_cnt[site]
                    for k in range(nc):
                        nb = nn_arr[site, k]
                        if (not in_cluster[nb] and spins[nb] == sv
                                and np.random.random() < p_add):
                            in_cluster[nb] = True
                            stack[sp] = nb
                            sp += 1

                for x in range(S):
                    if in_cluster[x]:
                        spins[x] = -spins[x]

                # --- Collect sample ---
                if step >= n_equil:
                    since_last += 1
                    if since_last >= n_skip:
                        since_last = 0
                        E = 0.0
                        for e in range(n_edges):
                            E += spins[edges_a[e]] * spins[edges_b[e]]
                        energies[si] = E
                        si += 1
                        if si >= n_sample:
                            break

            return energies[:si]

        # Warm-up compile on tiny triangle (N=2, 3 spins)
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

    # -----------------------------------------------------------------
    # Monte Carlo driver
    # -----------------------------------------------------------------
    def run_mc(S, edges_a, edges_b, nn_arr, nn_cnt,
               beta, n_equil, n_sample, n_skip):
        """
        Run Wolff cluster MC at inverse temperature beta.
        Returns (mean of Σ_{<i,j>} σ_i σ_j,  standard error).
        """
        if beta < 1e-14:
            return 0.0, 0.0

        p_add = 1.0 - np.exp(-2.0 * beta)
        spins = np.ones(S, dtype=np.int8)

        if use_numba:
            energies = _wolff_mc(spins, nn_arr, nn_cnt, edges_a, edges_b,
                                 p_add, S, n_equil, n_sample, n_skip)
        else:
            rng = np.random.default_rng()

            def wolff_step():
                seed = int(rng.integers(0, S))
                sv = int(spins[seed])
                cluster = {seed}
                stk = [seed]
                while stk:
                    site = stk.pop()
                    for k in range(nn_cnt[site]):
                        nb = int(nn_arr[site, k])
                        if (nb not in cluster and spins[nb] == sv
                                and rng.random() < p_add):
                            cluster.add(nb)
                            stk.append(nb)
                for s in cluster:
                    spins[s] = -spins[s]

            for _ in range(n_equil):
                wolff_step()

            energies = np.empty(n_sample)
            for i in range(n_sample):
                for _ in range(n_skip):
                    wolff_step()
                energies[i] = np.sum(spins[edges_a] * spins[edges_b])

        mean_E = float(np.mean(energies))
        std_E = float(np.std(energies, ddof=1) / np.sqrt(len(energies)))
        return mean_E, std_E

    # -----------------------------------------------------------------
    # Simulation parameters
    # -----------------------------------------------------------------
    if use_numba:
        sizes = [10, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400, 500]
        n_quad = 24
        n_equil = 5000
        n_sample = 10000
        n_skip = 3
    else:
        sizes = [8, 12, 16, 20, 25, 30, 40, 50]
        n_quad = 16
        n_equil = 1500
        n_sample = 3000
        n_skip = 2

    # Gauss-Legendre quadrature on [0, β_c]
    gl_nodes, gl_weights = leggauss(n_quad)
    beta_pts = 0.5 * beta_c * (gl_nodes + 1.0)
    w_pts = 0.5 * beta_c * gl_weights

    print(f"beta_c = {beta_c:.10f}")
    print(f"Quadrature points: {n_quad}")
    print(f"System sizes N: {sizes}\n")

    f_data = []
    u_data = []

    for N in sizes:
        S, ea, eb, nn_arr, nn_cnt = build_lattice(N)
        n_edges = len(ea)
        print(f"N={N:3d}  S={S:5d}  edges={n_edges:6d} ... ",
              end="", flush=True)

        # Thermodynamic integration for f_N(β_c)
        # f_N(β_c) = ln(2) + ∫₀^{β_c} (1/S)⟨Σ σ_i σ_j⟩_β dβ
        integral = 0.0
        for k in range(n_quad):
            mE, _ = run_mc(S, ea, eb, nn_arr, nn_cnt,
                           beta_pts[k], n_equil, n_sample, n_skip)
            integral += w_pts[k] * (mE / S)

        f_N = np.log(2.0) + integral

        # Direct measurement of u_N(β_c) = (1/S)⟨Σ σ_i σ_j⟩_{β_c}
        mE_c, sE_c = run_mc(S, ea, eb, nn_arr, nn_cnt,
                            beta_c, n_equil, n_sample * 2, n_skip)
        u_N = mE_c / S
        u_N_err = sE_c / S

        f_data.append(f_N)
        u_data.append(u_N)
        print(f"f_N = {f_N:.7f}   u_N = {u_N:.7f} +/- {u_N_err:.7f}")

    # -----------------------------------------------------------------
    # Finite-size extrapolation
    # -----------------------------------------------------------------
    sizes_arr = np.array(sizes, dtype=np.float64)
    f_arr = np.array(f_data)
    u_arr = np.array(u_data)

    # --- f baseline: q = q_inf + a/N + b/N^2 ---
    A_f = np.column_stack([
        np.ones(len(sizes)),
        1.0 / sizes_arr,
        1.0 / sizes_arr**2,
    ])
    cf, _, _, _ = np.linalg.lstsq(A_f, f_arr, rcond=None)
    f_infty = float(cf[0])
    f_fit = A_f @ cf
    f_residual = float(np.std(f_arr - f_fit))

    # --- u FIT A (baseline): u = u_inf + a/N + b/N^2 ---
    A_u = np.column_stack([
        np.ones(len(sizes)),
        1.0 / sizes_arr,
        1.0 / sizes_arr**2,
    ])
    cu, _, _, _ = np.linalg.lstsq(A_u, u_arr, rcond=None)
    u_infty = float(cu[0])
    u_fit = A_u @ cu
    u_residual = float(np.std(u_arr - u_fit))

    # --- u FIT B (log-corrected, minimal + stable) ---
    # Model: u(N) = u_inf + a*(log N)/N + b*(1/N) + c*(1/N^2)
    logN = np.log(sizes_arr)

    A_log = np.column_stack([
        np.ones(len(sizes_arr)),          # u_inf
        (logN) / sizes_arr,               # a * (log N)/N
        1.0 / sizes_arr,                  # b * 1/N
        1.0 / (sizes_arr**2),             # c * 1/N^2
    ])

    cu_log, _, _, _ = np.linalg.lstsq(A_log, u_arr, rcond=None)
    u_infty_log = float(cu_log[0])
    u_fit_log = A_log @ cu_log
    u_residual_log = float(np.std(u_arr - u_fit_log))

    # (optional but useful) print stability info
    cond_baseline = float(np.linalg.cond(A_u))
    cond_log = float(np.linalg.cond(A_log))


    # --- Summary ---
    print()
    print("=" * 62)
    print("FINITE-SIZE EXTRAPOLATION COMPARISON")
    print("=" * 62)
    print(f"  f_infty   (baseline)  = {f_infty:.8f}   residual std: {f_residual:.2e}")
    print(f"  u_infty   (baseline)  = {u_infty:.8f}   residual std: {u_residual:.2e}")
    print(f"  u_infty   (log-fit)   = {u_infty_log:.8f}   residual std: {u_residual_log:.2e}")
    print(f"  cond(A_u) baseline     = {cond_baseline:.2e}")
    print(f"  cond(A_log) log-fit    = {cond_log:.2e}")
    print(f"  sizes used: {sizes}")
    print("=" * 62)

    result = {
        "f_infty_est": f_infty,
        "u_infty_est": u_infty,
        "u_infty_logfit": u_infty_log,
        "beta_c": float(beta_c),
        "sizes_used": sizes,
        "f_values": [float(v) for v in f_data],
        "u_values": [float(v) for v in u_data],
        "f_fit_residual_std": f_residual,
        "u_fit_residual_std": u_residual,
        "u_logfit_residual_std": u_residual_log,
    }

    print(f"\nFinal result: {result}")
    return result


solver()
