def solver():
    """
    Sampling-based inference for the 2D triangular-lattice Ising model
    on a periodic L×L lattice (torus) with ferromagnetic coupling J=1, h=0.

    Strategy 1: Wolff Cluster Algorithm + Thermodynamic Integration
    ---------------------------------------------------------------
    1. Wolff cluster updates for efficient sampling near criticality.
    2. β_c estimated from data via Binder cumulant crossing across system sizes.
    3. Gauss-Legendre quadrature for free energy:
       f(β_c) = −(1/(Nβ_c))[N ln2 + ∫₀^{β_c} ⟨−H⟩_β dβ]
    4. Finite-size extrapolation: q(L) = q_∞ + a/L + b/L² for L→∞.
    """
    import numpy as np
    from numpy.polynomial.legendre import leggauss

    # -----------------------------------------------------------------
    # Lattice construction: periodic L×L triangular lattice
    # Each site (r,c) has 6 neighbors on the torus.
    # -----------------------------------------------------------------
    def build_lattice(L):
        """
        Periodic L×L triangular lattice.
        N = L*L spins, each with exactly 6 neighbors.
        Edges = 3*N (3 edge directions, each counted once).
        Returns N, edges_a, edges_b, nn_arr (N×6).
        """
        N = L * L

        def idx(r, c):
            return (r % L) * L + (c % L)

        neighbors = [[] for _ in range(N)]
        edges_a, edges_b = [], []

        for r in range(L):
            for c in range(L):
                i = idx(r, c)
                # 3 edges per site (to avoid double counting)
                # Right: (r, c) - (r, c+1)
                j = idx(r, c + 1)
                neighbors[i].append(j)
                neighbors[j].append(i)
                edges_a.append(i)
                edges_b.append(j)

                # Down: (r, c) - (r+1, c)
                j = idx(r + 1, c)
                neighbors[i].append(j)
                neighbors[j].append(i)
                edges_a.append(i)
                edges_b.append(j)

                # Down-left diagonal: (r, c) - (r+1, c-1)
                j = idx(r + 1, c - 1)
                neighbors[i].append(j)
                neighbors[j].append(i)
                edges_a.append(i)
                edges_b.append(j)

        ea = np.array(edges_a, dtype=np.int32)
        eb = np.array(edges_b, dtype=np.int32)

        # Each site has exactly 6 neighbors
        nn_arr = np.empty((N, 6), dtype=np.int32)
        for i in range(N):
            for k in range(6):
                nn_arr[i, k] = neighbors[i][k]

        return N, ea, eb, nn_arr

    # -----------------------------------------------------------------
    # Numba-accelerated Wolff cluster MC
    # -----------------------------------------------------------------
    use_numba = False
    try:
        from numba import njit

        @njit
        def _wolff_mc(spins, nn_arr, edges_a, edges_b,
                      p_add, N, n_equil, n_sample, n_skip):
            """Wolff cluster MC returning bond-energy samples Σ σ_iσ_j."""
            stack = np.empty(N, dtype=np.int32)
            in_cluster = np.zeros(N, dtype=np.bool_)
            energies = np.empty(n_sample, dtype=np.float64)
            n_edges = len(edges_a)
            si = 0
            since_last = 0
            total_steps = n_equil + n_sample * n_skip

            for step in range(total_steps):
                # --- Wolff cluster step ---
                seed = np.random.randint(0, N)
                sv = spins[seed]
                for x in range(N):
                    in_cluster[x] = False
                in_cluster[seed] = True
                stack[0] = seed
                sp = 1

                while sp > 0:
                    sp -= 1
                    site = stack[sp]
                    for k in range(6):
                        nb = nn_arr[site, k]
                        if (not in_cluster[nb] and spins[nb] == sv
                                and np.random.random() < p_add):
                            in_cluster[nb] = True
                            stack[sp] = nb
                            sp += 1

                for x in range(N):
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

        @njit
        def _wolff_mc_observables(spins, nn_arr, edges_a, edges_b,
                                  p_add, N, n_equil, n_sample, n_skip):
            """Wolff MC returning (bond_energy, mag², |M|, M⁴) samples."""
            stack = np.empty(N, dtype=np.int32)
            in_cluster = np.zeros(N, dtype=np.bool_)
            n_edges = len(edges_a)
            energies = np.empty(n_sample, dtype=np.float64)
            mag2 = np.empty(n_sample, dtype=np.float64)
            mag_abs = np.empty(n_sample, dtype=np.float64)
            mag4 = np.empty(n_sample, dtype=np.float64)
            si = 0
            since_last = 0
            total_steps = n_equil + n_sample * n_skip

            for step in range(total_steps):
                seed = np.random.randint(0, N)
                sv = spins[seed]
                for x in range(N):
                    in_cluster[x] = False
                in_cluster[seed] = True
                stack[0] = seed
                sp = 1

                while sp > 0:
                    sp -= 1
                    site = stack[sp]
                    for k in range(6):
                        nb = nn_arr[site, k]
                        if (not in_cluster[nb] and spins[nb] == sv
                                and np.random.random() < p_add):
                            in_cluster[nb] = True
                            stack[sp] = nb
                            sp += 1

                for x in range(N):
                    if in_cluster[x]:
                        spins[x] = -spins[x]

                if step >= n_equil:
                    since_last += 1
                    if since_last >= n_skip:
                        since_last = 0
                        E = 0.0
                        for e in range(n_edges):
                            E += spins[edges_a[e]] * spins[edges_b[e]]
                        energies[si] = E
                        M = 0.0
                        for x in range(N):
                            M += spins[x]
                        m = M / N
                        mag2[si] = m * m
                        mag_abs[si] = abs(m)
                        mag4[si] = m * m * m * m
                        si += 1
                        if si >= n_sample:
                            break

            return energies[:si], mag2[:si], mag_abs[:si], mag4[:si]

        # Warm-up compile with tiny 2×2 lattice
        _N2, _ea2, _eb2, _nn2 = build_lattice(2)
        _s2 = np.ones(_N2, dtype=np.int8)
        _wolff_mc(_s2.copy(), _nn2, _ea2, _eb2, 0.5, _N2, 2, 2, 1)
        _wolff_mc_observables(_s2.copy(), _nn2, _ea2, _eb2, 0.5, _N2, 2, 2, 1)

        use_numba = True
        print("Using numba-accelerated Wolff cluster algorithm.")
    except Exception:
        print("Numba unavailable; using pure-Python Wolff cluster algorithm.")

    # -----------------------------------------------------------------
    # MC driver functions
    # -----------------------------------------------------------------
    def run_mc_energy(N, edges_a, edges_b, nn_arr, beta,
                      n_equil, n_sample, n_skip):
        """Run Wolff MC, return (mean bond sum, std error)."""
        if beta < 1e-14:
            return 0.0, 0.0

        p_add = 1.0 - np.exp(-2.0 * beta)
        spins = np.ones(N, dtype=np.int8)

        if use_numba:
            energies = _wolff_mc(spins, nn_arr, edges_a, edges_b,
                                 p_add, N, n_equil, n_sample, n_skip)
        else:
            rng = np.random.default_rng()

            def wolff_step():
                seed = int(rng.integers(0, N))
                sv = int(spins[seed])
                cluster = {seed}
                stk = [seed]
                while stk:
                    site = stk.pop()
                    for k in range(6):
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
                energies[i] = float(np.sum(
                    spins[edges_a].astype(np.int32) * spins[edges_b].astype(np.int32)
                ))

        mean_E = float(np.mean(energies))
        std_E = float(np.std(energies, ddof=1) / np.sqrt(len(energies)))
        return mean_E, std_E

    def run_mc_observables(N, edges_a, edges_b, nn_arr, beta,
                           n_equil, n_sample, n_skip):
        """Run Wolff MC, return dict of observable means for Binder analysis."""
        if beta < 1e-14:
            return {"bond_sum": 0.0, "m2": 0.0, "m_abs": 0.0, "m4": 0.0}

        p_add = 1.0 - np.exp(-2.0 * beta)
        spins = np.ones(N, dtype=np.int8)

        if use_numba:
            energies, mag2, mag_abs, mag4 = _wolff_mc_observables(
                spins, nn_arr, edges_a, edges_b,
                p_add, N, n_equil, n_sample, n_skip
            )
        else:
            rng = np.random.default_rng()

            def wolff_step():
                seed = int(rng.integers(0, N))
                sv = int(spins[seed])
                cluster = {seed}
                stk = [seed]
                while stk:
                    site = stk.pop()
                    for k in range(6):
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
            mag2 = np.empty(n_sample)
            mag_abs = np.empty(n_sample)
            mag4 = np.empty(n_sample)
            for i in range(n_sample):
                for _ in range(n_skip):
                    wolff_step()
                energies[i] = float(np.sum(
                    spins[edges_a].astype(np.int32) * spins[edges_b].astype(np.int32)
                ))
                m = float(np.sum(spins)) / N
                mag2[i] = m * m
                mag_abs[i] = abs(m)
                mag4[i] = m ** 4

        return {
            "bond_sum": float(np.mean(energies)),
            "m2": float(np.mean(mag2)),
            "m_abs": float(np.mean(mag_abs)),
            "m4": float(np.mean(mag4)),
        }

    # -----------------------------------------------------------------
    # Simulation parameters
    # -----------------------------------------------------------------
    if use_numba:
        binder_sizes = [8, 12, 16, 24, 32]
        thermo_sizes = [8, 12, 16, 24, 32, 48, 64]
        n_quad = 20
        n_equil = 3000
        n_sample = 8000
        n_skip = 2
        n_equil_binder = 2000
        n_sample_binder = 5000
        n_skip_binder = 2
    else:
        binder_sizes = [6, 8, 12, 16]
        thermo_sizes = [6, 8, 12, 16, 20, 24, 32]
        n_quad = 16
        n_equil = 1500
        n_sample = 3000
        n_skip = 2
        n_equil_binder = 1000
        n_sample_binder = 2000
        n_skip_binder = 2

    # =================================================================
    # STEP 1: Estimate β_c from Binder cumulant crossing
    # =================================================================
    beta_exact = 0.25 * np.log(3)  # ≈ 0.274653

    n_beta_scan = 15
    beta_scan = np.linspace(0.20, 0.35, n_beta_scan)

    print("=" * 60)
    print("STEP 1: Estimating β_c via Binder cumulant crossing")
    print("=" * 60)

    binder_data = {}

    for L in binder_sizes:
        N, ea, eb, nn_arr = build_lattice(L)
        U4_vals = np.zeros(n_beta_scan)
        print(f"  L={L:3d} (N={N:5d}): scanning {n_beta_scan} β values ...",
              end="", flush=True)
        for ib, beta in enumerate(beta_scan):
            obs = run_mc_observables(N, ea, eb, nn_arr, beta,
                                     n_equil_binder, n_sample_binder,
                                     n_skip_binder)
            m2 = obs["m2"]
            m4 = obs["m4"]
            # Binder cumulant: U4 = 1 - <m^4>/(3<m^2>^2)
            if m2 > 1e-15:
                U4_vals[ib] = 1.0 - m4 / (3.0 * m2 * m2)
            else:
                U4_vals[ib] = 0.0
        binder_data[L] = U4_vals
        print(" done.")

    # Find crossing points between all adjacent size pairs
    all_crossings = []
    sorted_sizes = sorted(binder_sizes)
    for i in range(len(sorted_sizes) - 1):
        L1, L2 = sorted_sizes[i], sorted_sizes[i + 1]
        diff = binder_data[L2] - binder_data[L1]
        for j in range(len(diff) - 1):
            if diff[j] * diff[j + 1] < 0:
                beta_cross = beta_scan[j] - diff[j] * (
                    beta_scan[j + 1] - beta_scan[j]
                ) / (diff[j + 1] - diff[j])
                all_crossings.append((beta_cross, L1, L2))

    # Filter: keep only crossings in the central region of the scan
    # (discard spurious crossings near the edges of the β range)
    beta_mid = 0.5 * (beta_scan[0] + beta_scan[-1])
    beta_half_width = 0.3 * (beta_scan[-1] - beta_scan[0])
    crossings = [bc for bc, L1, L2 in all_crossings
                 if abs(bc - beta_mid) < beta_half_width]

    if len(crossings) > 0:
        beta_c_est = float(np.median(crossings))
        beta_c_std = float(np.std(crossings)) if len(crossings) > 1 else 0.0
    else:
        print("  WARNING: No Binder crossings found; using exact β_c.")
        beta_c_est = beta_exact
        beta_c_std = 0.0

    print(f"\n  Binder crossings found: {len(crossings)}")
    if len(crossings) > 0:
        print(f"  Crossing values: {[f'{x:.6f}' for x in crossings]}")
    print(f"  Estimated β_c = {beta_c_est:.6f} ± {beta_c_std:.6f}")
    print(f"  Exact     β_c = {beta_exact:.6f}")
    print(f"  Relative error = {abs(beta_c_est - beta_exact)/beta_exact * 100:.2f}%")

    beta_c = beta_c_est

    # =================================================================
    # STEP 2: Thermodynamic integration + energy measurement at β_c
    # =================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Thermodynamic integration & energy at β_c")
    print("=" * 60)

    # Gauss-Legendre quadrature on [0, β_c]
    gl_nodes, gl_weights = leggauss(n_quad)
    beta_pts = 0.5 * beta_c * (gl_nodes + 1.0)
    w_pts = 0.5 * beta_c * gl_weights

    print(f"  β_c (used) = {beta_c:.10f}")
    print(f"  Quadrature points: {n_quad}")
    print(f"  System sizes L: {thermo_sizes}\n")

    # Free energy via thermodynamic integration:
    #   ln Z(β) = N ln2 + ∫₀^β ⟨Σ σ_iσ_j⟩_{β'} dβ'
    #   f(β) = -(1/(Nβ)) ln Z(β) = -(1/β)[ln2 + (1/N)∫₀^β ⟨Σ σ_iσ_j⟩_{β'} dβ']
    # Energy per spin: u = ⟨H⟩/N = -⟨Σ σ_iσ_j⟩/N

    f_data = []
    u_data = []
    s_data = []

    for L in thermo_sizes:
        N, ea, eb, nn_arr = build_lattice(L)
        n_edges = len(ea)
        print(f"  L={L:3d}  N={N:5d}  edges={n_edges:6d} ... ",
              end="", flush=True)

        # Thermodynamic integration: (1/N) ∫₀^{β_c} ⟨Σ σ_iσ_j⟩_β dβ
        integral = 0.0
        for k in range(n_quad):
            mE, _ = run_mc_energy(N, ea, eb, nn_arr, beta_pts[k],
                                  n_equil, n_sample, n_skip)
            integral += w_pts[k] * (mE / N)

        # (1/N) ln Z(β_c) = ln2 + integral
        # f(β_c) = -(1/β_c)(ln2 + integral)
        lnZ_per_spin = np.log(2.0) + integral
        f_N = -(1.0 / beta_c) * lnZ_per_spin

        # Direct measurement of energy at β_c
        mE_c, sE_c = run_mc_energy(N, ea, eb, nn_arr, beta_c,
                                    n_equil, n_sample * 2, n_skip)
        u_N = -mE_c / N  # u = ⟨H⟩/N = -bond_sum/N
        u_N_err = sE_c / N

        # Entropy per spin: s = β(u - f)
        s_N = beta_c * (u_N - f_N)

        f_data.append(f_N)
        u_data.append(u_N)
        s_data.append(s_N)
        print(f"f={f_N:.7f}  u={u_N:.7f}±{u_N_err:.7f}  s={s_N:.7f}")

    # =================================================================
    # STEP 3: Finite-size extrapolation: q(L) = q_∞ + a/L + b/L²
    # =================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Finite-size extrapolation (q = q_∞ + a/L + b/L²)")
    print("=" * 60)

    L_arr = np.array(thermo_sizes, dtype=np.float64)
    f_arr = np.array(f_data)
    u_arr = np.array(u_data)
    s_arr = np.array(s_data)

    A = np.column_stack([
        np.ones(len(thermo_sizes)),
        1.0 / L_arr,
        1.0 / L_arr**2,
    ])

    cf, _, _, _ = np.linalg.lstsq(A, f_arr, rcond=None)
    f_infty = float(cf[0])
    f_residual = float(np.std(f_arr - A @ cf))

    cu, _, _, _ = np.linalg.lstsq(A, u_arr, rcond=None)
    u_infty = float(cu[0])
    u_residual = float(np.std(u_arr - A @ cu))

    cs, _, _, _ = np.linalg.lstsq(A, s_arr, rcond=None)
    s_infty = float(cs[0])
    s_residual = float(np.std(s_arr - A @ cs))

    # Log-corrected fit for energy (2D Ising has log divergence in C_v)
    A_log = np.column_stack([
        np.ones(len(thermo_sizes)),
        np.log(L_arr) / L_arr,
        1.0 / L_arr,
        1.0 / L_arr**2,
    ])
    cu_log, _, _, _ = np.linalg.lstsq(A_log, u_arr, rcond=None)
    u_infty_log = float(cu_log[0])
    u_residual_log = float(np.std(u_arr - A_log @ cu_log))

    # Entropy from best estimates
    s_from_thermo = beta_c * (u_infty - f_infty)
    s_log_from_thermo = beta_c * (u_infty_log - f_infty)

    print(f"  f_∞ = {f_infty:.8f}   (residual std: {f_residual:.2e})")
    print(f"  u_∞ = {u_infty:.8f}   (residual std: {u_residual:.2e})")
    print(f"  u_∞ (log-fit) = {u_infty_log:.8f}   (residual std: {u_residual_log:.2e})")
    print(f"  s_∞ (direct fit) = {s_infty:.8f}   (residual std: {s_residual:.2e})")
    print(f"  s_∞ (from u,f)   = {s_from_thermo:.8f}")
    print(f"  s_∞ (log u, f)   = {s_log_from_thermo:.8f}")

    # =================================================================
    # Assemble final results
    # =================================================================
    if u_residual_log < u_residual and len(thermo_sizes) >= 5:
        u_best = u_infty_log
        u_method = "log-corrected fit"
    else:
        u_best = u_infty
        u_method = "polynomial fit"

    s_best = beta_c * (u_best - f_infty)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  β_c (estimated from Binder cumulant) = {beta_c:.6f}")
    print(f"  β_c (exact)                          = {beta_exact:.6f}")
    print(f"  Mean energy per spin at β_c           = {u_best:.6f}  ({u_method})")
    print(f"  Free energy per spin at β_c           = {f_infty:.6f}")
    print(f"  Entropy per spin at β_c               = {s_best:.6f}")
    print("=" * 60)

    result = {
        "beta_c": float(beta_c),
        "beta_c_exact": float(beta_exact),
        "beta_c_error": float(abs(beta_c - beta_exact)),
        "mean_energy_per_spin": float(u_best),
        "free_energy_per_spin": float(f_infty),
        "entropy_per_spin": float(s_best),
        "energy_method": u_method,
        "u_infty_polynomial": float(u_infty),
        "u_infty_logfit": float(u_infty_log),
        "u_polynomial_residual": float(u_residual),
        "u_logfit_residual": float(u_residual_log),
        "f_infty": float(f_infty),
        "f_residual": float(f_residual),
        "s_infty_direct": float(s_infty),
        "s_from_thermodynamics": float(s_best),
        "binder_crossings": [float(x) for x in crossings],
        "sizes_used": thermo_sizes,
        "f_values": [float(v) for v in f_data],
        "u_values": [float(v) for v in u_data],
        "s_values": [float(v) for v in s_data],
    }

    print(f"\nFinal result: {result}")
    return result
