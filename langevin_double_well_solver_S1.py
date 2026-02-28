def solver():
    """
    S1 (DEBUGGED): Vectorised Unadjusted Langevin Algorithm (ULA).

    Fix applied (Code Critic S1):
      - dt increased from 0.005 to 0.025 (still within stability limit
        dt < 2/max_eigenvalue_Hessian ~ 0.045).  At dt=0.025 the expected
        number of inter-well crossings per chain rises from ~0.4 to ~6,
        allowing individual chains to mix and R-hat(x) to converge.
      - N_samples increased from 5500 to 15000 per chain (3,000,000 total)
        to further ensure each chain observes multiple well transitions.
      - N_burnin reduced to 2000 (faster equilibration at larger dt).
    """
    import numpy as np
    import time
    from scipy.stats import gaussian_kde

    try:
        # ------------------------------------------------------------------ #
        # Problem parameters
        # ------------------------------------------------------------------ #
        sigma   = 0.7
        beta    = 2.0 / sigma**2    # effective inverse temperature ~ 4.0816
        # dt=0.025: stability OK (limit ~0.045); ~6 inter-well crossings/chain
        dt      = 0.025
        sqrt_dt = np.sqrt(dt)

        # ------------------------------------------------------------------ #
        # Simulation parameters  (FIX: larger dt + more steps per chain)
        # ------------------------------------------------------------------ #
        N_chains  = 200
        N_burnin  = 2_000           # burn-in steps per chain
        N_samples = 15_000          # kept steps per chain -> 3,000,000 total
        np.random.seed(2024)

        # Validate parameters
        assert dt < 0.045,  "dt exceeds stability limit for this potential"
        assert N_samples >= 10_000, "Too few samples for reliable R-hat"

        # ------------------------------------------------------------------ #
        # Initialise chains: half near (+1,0), half near (-1,0)
        # ------------------------------------------------------------------ #
        half = N_chains // 2
        x = np.concatenate([
            1.0 + 0.15 * np.random.randn(half),
           -1.0 + 0.15 * np.random.randn(N_chains - half)
        ])
        y = 0.15 * np.random.randn(N_chains)

        # Pre-allocate sample storage (float32 halves memory usage)
        samples_x = np.empty((N_samples, N_chains), dtype=np.float32)
        samples_y = np.empty((N_samples, N_chains), dtype=np.float32)

        # ------------------------------------------------------------------ #
        # Euler-Maruyama loop -- vectorised over all chains
        # ------------------------------------------------------------------ #
        t_start = time.perf_counter()

        for i in range(N_burnin + N_samples):
            # Gradient of V(x,y) = (x^2-1)^2 + y^2
            gx = 4.0 * x * (x * x - 1.0)   # dV/dx = 4x(x^2-1)
            gy = 2.0 * y                      # dV/dy = 2y

            # Euler-Maruyama step
            x = x - gx * dt + sigma * sqrt_dt * np.random.randn(N_chains)
            y = y - gy * dt + sigma * sqrt_dt * np.random.randn(N_chains)

            if i >= N_burnin:
                k = i - N_burnin
                samples_x[k] = x.astype(np.float32)
                samples_y[k] = y.astype(np.float32)

        t_sim = time.perf_counter() - t_start

        # ------------------------------------------------------------------ #
        # Flatten all chains into one sample pool
        # ------------------------------------------------------------------ #
        X_flat = samples_x.ravel().astype(np.float64)
        Y_flat = samples_y.ravel().astype(np.float64)
        n_total = len(X_flat)

        # Basic domain validation (all samples should be finite)
        if not (np.all(np.isfinite(X_flat)) and np.all(np.isfinite(Y_flat))):
            return {"error": "Non-finite values in samples; reduce dt."}

        # ------------------------------------------------------------------ #
        # Empirical statistics
        # ------------------------------------------------------------------ #
        empirical_mean = [float(np.mean(X_flat)), float(np.mean(Y_flat))]
        cov_mat        = np.cov(X_flat, Y_flat)

        # ------------------------------------------------------------------ #
        # Kernel density estimation (subsample for speed)
        # ------------------------------------------------------------------ #
        n_kde = min(200_000, n_total)
        idx   = np.random.choice(n_total, n_kde, replace=False)
        kde   = gaussian_kde(
            np.vstack([X_flat[idx], Y_flat[idx]]),
            bw_method='scott'
        )

        xg = np.linspace(-2.5, 2.5, 100)
        yg = np.linspace(-2.2, 2.2, 100)
        XX, YY = np.meshgrid(xg, yg)
        ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

        # ------------------------------------------------------------------ #
        # Convergence diagnostics
        # ------------------------------------------------------------------ #
        def gelman_rubin(chains):
            """Gelman-Rubin R-hat for a (T, M) array."""
            T, M = chains.shape
            cm   = chains.mean(axis=0)
            cv   = chains.var(axis=0, ddof=1)
            W    = cv.mean()
            B    = T * cm.var(ddof=1)
            if W == 0.0:
                return float('nan')
            return float(np.sqrt(((T - 1) / T * W + B / T) / W))

        def batch_ess(chain, n_batch=50):
            """Batch-means ESS estimate for a single chain."""
            T = len(chain)
            b = T // n_batch
            if b == 0:
                return float(T)
            bm = np.array([chain[i*b:(i+1)*b].mean() for i in range(n_batch)])
            vt = chain[:n_batch * b].var()
            vb = bm.var(ddof=1)
            if vb == 0 or vt == 0:
                return float(T)
            return T * vt / (b * vb)

        n_diag  = min(40, N_chains)
        r_hat_x = gelman_rubin(samples_x[:, :n_diag].astype(np.float64))
        r_hat_y = gelman_rubin(samples_y[:, :n_diag].astype(np.float64))

        ess_x = float(np.mean([
            batch_ess(samples_x[:, j].astype(np.float64))
            for j in range(n_diag)
        ]))
        ess_y = float(np.mean([
            batch_ess(samples_y[:, j].astype(np.float64))
            for j in range(n_diag)
        ]))

        # Count inter-well sign-change transitions in x per chain
        n_transitions = int(sum(
            np.sum(np.diff(np.sign(samples_x[:, j])) != 0)
            for j in range(N_chains)
        ))

        return {
            "strategy_id"             : "S1",
            "algorithm"               : "Unadjusted Langevin Algorithm (Euler-Maruyama)",
            "n_samples"               : n_total,
            "n_chains"                : N_chains,
            "dt"                      : dt,
            "empirical_mean"          : empirical_mean,
            "empirical_covariance"    : cov_mat.tolist(),
            "kde_bandwidth_scott"     : float(kde.factor),
            "kde_grid_x"              : xg.tolist(),
            "kde_grid_y"              : yg.tolist(),
            "kde_density"             : ZZ.tolist(),
            "gelman_rubin_x"          : r_hat_x,
            "gelman_rubin_y"          : r_hat_y,
            "ess_per_chain_x"         : ess_x,
            "ess_per_chain_y"         : ess_y,
            "total_ess_x"             : ess_x * N_chains,
            "total_ess_y"             : ess_y * N_chains,
            "ess_efficiency_x"        : (ess_x * N_chains) / n_total,
            "inter_well_transitions"  : n_transitions,
            "simulation_time_s"       : float(t_sim),
            "throughput_samples_per_s": n_total / t_sim,
        }

    except Exception as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    import json
    result = solver()
    summary = {k: v for k, v in result.items()
               if k not in ("kde_density", "kde_grid_x", "kde_grid_y")}
    print(json.dumps(summary, indent=2))
