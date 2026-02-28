def solver():
    """
    S2 (DEBUGGED): Metropolis-Adjusted Langevin Algorithm (MALA).

    Fixes applied (Code Critic S2):
      - N_adapt extended to equal N_burnin (3000) so step-size tuning
        continues across the entire burn-in period, preventing premature
        freezing of dt before the chain has reached stationarity.
      - N_samples increased from 10500 to 15000 per chain (1,500,000 total)
        to improve ESS and push R-hat(x) below 1.01.
      - adapt_interval kept at 100 steps; upper cap on dt tightened to 0.5
        to avoid over-large steps during early adaptation.
    """
    import numpy as np
    import time
    from scipy.stats import gaussian_kde

    try:
        # ------------------------------------------------------------------ #
        # Problem parameters
        # ------------------------------------------------------------------ #
        sigma = 0.7
        beta  = 2.0 / sigma**2            # ~4.0816

        # ------------------------------------------------------------------ #
        # Simulation parameters  (FIX: N_adapt = N_burnin, larger N_samples)
        # ------------------------------------------------------------------ #
        dt             = 0.10             # initial step size
        N_chains       = 100
        N_burnin       = 3_000            # burn-in steps per chain
        N_samples      = 15_000           # kept steps -> 1,500,000 total
        N_adapt        = 3_000            # FIX: now equals N_burnin
        target_accept  = 0.57             # optimal MALA rate in d=2
        adapt_interval = 100
        np.random.seed(2024)

        # Validate
        assert N_adapt <= N_burnin, "N_adapt must not exceed N_burnin"
        assert N_samples >= 10_000, "Too few samples per chain"

        # ------------------------------------------------------------------ #
        # Helper functions
        # ------------------------------------------------------------------ #
        def grad_V(x, y):
            """Gradient of V(x,y) = (x^2-1)^2 + y^2."""
            return 4.0 * x * (x * x - 1.0), 2.0 * y

        def log_pi(x, y):
            """Unnormalised log-density: log pi ~ -beta * V(x,y)."""
            return -beta * ((x * x - 1.0) ** 2 + y * y)

        # ------------------------------------------------------------------ #
        # Initialise chains: half near (+1,0), half near (-1,0)
        # ------------------------------------------------------------------ #
        half = N_chains // 2
        x = np.concatenate([
            1.0 + 0.15 * np.random.randn(half),
           -1.0 + 0.15 * np.random.randn(N_chains - half)
        ])
        y = 0.15 * np.random.randn(N_chains)

        # Cache current log-density and gradient (avoids redundant re-computation)
        lp_curr          = log_pi(x, y)
        gx_curr, gy_curr = grad_V(x, y)

        samples_x     = np.empty((N_samples, N_chains), dtype=np.float32)
        samples_y     = np.empty((N_samples, N_chains), dtype=np.float32)
        accept_count  = np.zeros(N_chains)
        recent_accept = np.zeros(N_chains)
        total_steps   = 0

        t_start = time.perf_counter()

        # ------------------------------------------------------------------ #
        # MALA loop -- vectorised over all chains
        # ------------------------------------------------------------------ #
        for i in range(N_burnin + N_samples):
            sqrt_dt = np.sqrt(dt)

            # -- Proposal: X* = X - (dt/2)*grad_V(X) + sqrt(dt)*xi ----------
            mx_fwd = x - 0.5 * dt * gx_curr
            my_fwd = y - 0.5 * dt * gy_curr
            x_prop = mx_fwd + sqrt_dt * np.random.randn(N_chains)
            y_prop = my_fwd + sqrt_dt * np.random.randn(N_chains)

            # -- Reverse proposal mean at proposed point --------------------
            gx_prop, gy_prop = grad_V(x_prop, y_prop)
            mx_rev = x_prop - 0.5 * dt * gx_prop
            my_rev = y_prop - 0.5 * dt * gy_prop

            # -- Log-density of proposed point ------------------------------
            lp_prop = log_pi(x_prop, y_prop)

            # -- Log-proposal ratios (normalisation constant cancels) -------
            log_q_fwd = -0.5 / dt * (
                (x_prop - mx_fwd) ** 2 + (y_prop - my_fwd) ** 2
            )
            log_q_rev = -0.5 / dt * (
                (x - mx_rev) ** 2 + (y - my_rev) ** 2
            )

            # -- Log acceptance ratio: log pi(X*) - log pi(X)
            #    + log q(X|X*) - log q(X*|X) --------------------------------
            log_alpha = np.minimum(
                0.0,
                lp_prop - lp_curr + log_q_rev - log_q_fwd
            )
            accept = np.log(np.random.uniform(0.0, 1.0, N_chains)) < log_alpha

            # -- Update state (cache accepted gradient for next iteration) ---
            x       = np.where(accept, x_prop,  x)
            y       = np.where(accept, y_prop,  y)
            lp_curr = np.where(accept, lp_prop, lp_curr)
            gx_curr = np.where(accept, gx_prop, gx_curr)
            gy_curr = np.where(accept, gy_prop, gy_curr)

            accept_count  += accept.astype(float)
            recent_accept += accept.astype(float)
            total_steps   += 1

            # -- Multiplicative step-size adaptation (FIX: runs through all
            #    N_adapt=3000 burn-in steps, cap tightened to 0.5) ----------
            if i < N_adapt and (i + 1) % adapt_interval == 0:
                batch_rate = recent_accept.mean() / adapt_interval
                if batch_rate > target_accept:
                    dt = min(dt * 1.10, 0.5)    # FIX: cap at 0.5, was 2.0
                else:
                    dt = max(dt * 0.90, 1e-4)
                recent_accept[:] = 0.0

            # -- Store post burn-in ------------------------------------------
            if i >= N_burnin:
                k = i - N_burnin
                samples_x[k] = x.astype(np.float32)
                samples_y[k] = y.astype(np.float32)

        t_sim = time.perf_counter() - t_start
        acceptance_rate = float(accept_count.mean() / total_steps)

        # ------------------------------------------------------------------ #
        # Flatten sample pool
        # ------------------------------------------------------------------ #
        X_flat = samples_x.ravel().astype(np.float64)
        Y_flat = samples_y.ravel().astype(np.float64)
        n_total = len(X_flat)

        if not (np.all(np.isfinite(X_flat)) and np.all(np.isfinite(Y_flat))):
            return {"error": "Non-finite values detected in MALA samples."}

        # ------------------------------------------------------------------ #
        # Empirical statistics
        # ------------------------------------------------------------------ #
        empirical_mean = [float(np.mean(X_flat)), float(np.mean(Y_flat))]
        cov_mat        = np.cov(X_flat, Y_flat)

        # ------------------------------------------------------------------ #
        # Kernel density estimation
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
            T, M = chains.shape
            cm   = chains.mean(axis=0)
            cv   = chains.var(axis=0, ddof=1)
            W    = cv.mean()
            B    = T * cm.var(ddof=1)
            if W == 0.0:
                return float('nan')
            return float(np.sqrt(((T - 1) / T * W + B / T) / W))

        def batch_ess(chain, n_batch=50):
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

        n_transitions = int(sum(
            np.sum(np.diff(np.sign(samples_x[:, j])) != 0)
            for j in range(N_chains)
        ))

        return {
            "strategy_id"             : "S2",
            "algorithm"               : "Metropolis-Adjusted Langevin Algorithm (MALA)",
            "n_samples"               : n_total,
            "n_chains"                : N_chains,
            "tuned_dt"                : float(dt),
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
            "acceptance_rate"         : acceptance_rate,
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
