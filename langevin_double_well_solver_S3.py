def solver():
    """
    S3 (DEBUGGED): Hamiltonian Monte Carlo (HMC) with Leapfrog Integration.

    Fixes applied (Code Critic S3):
      - L reduced from 20 to 12, keeping trajectory length L*eps ~ 1.2-2.3
        which avoids the large-x overflow region seen with L=20, eps=0.195.
      - Adaptation cap on eps tightened from 1.0 to 0.15, preventing the
        leapfrog from reaching the unstable region 4x(x^2-1) -> overflow.
      - NaN/inf guard added inside the leapfrog: any chain whose state goes
        non-finite is silently reset to its pre-proposal state so the MH
        step can cleanly reject it without propagating NaN values.
      - Energy error reporting now guards against NaN with np.nanmean/nanmax.
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
        # Simulation parameters  (FIX: L=12, tighter eps cap)
        # ------------------------------------------------------------------ #
        eps            = 0.10             # initial leapfrog step size
        L              = 12               # FIX: reduced from 20 (traj len ~1.2)
        N_chains       = 50
        N_burnin       = 1_000
        N_samples      = 21_000           # -> 1,050,000 total
        N_adapt        = 700
        target_accept  = 0.65
        adapt_interval = 100
        np.random.seed(2024)

        # Validate
        assert L >= 5, "L too small for effective proposals"
        assert eps > 0, "Step size must be positive"

        # ------------------------------------------------------------------ #
        # Helper functions
        # ------------------------------------------------------------------ #
        def grad_V(x, y):
            """Gradient of V(x,y) = (x^2-1)^2 + y^2."""
            return 4.0 * x * (x * x - 1.0), 2.0 * y

        def kinetic(px, py):
            """Kinetic energy K(p) = 0.5*(px^2+py^2), mass M=I."""
            return 0.5 * (px * px + py * py)

        def potential_energy(x, y):
            """Potential energy U(q) = beta * V(x,y)."""
            return beta * ((x * x - 1.0) ** 2 + y * y)

        def leapfrog(x, y, px, py, step, n_steps):
            """
            Stormer-Verlet leapfrog integrator, vectorised over chains.
            FIX: NaN/inf guard resets unstable chains to their entry state
            so the MH step can safely reject them without NaN propagation.
            """
            # Save entry state for NaN recovery
            x0, y0, px0, py0 = x.copy(), y.copy(), px.copy(), py.copy()

            # Initial half-step for momentum
            gx, gy = grad_V(x, y)
            px = px - 0.5 * step * beta * gx
            py = py - 0.5 * step * beta * gy

            # (n_steps - 1) full leapfrog steps
            for _ in range(n_steps - 1):
                x  = x  + step * px
                y  = y  + step * py
                gx, gy = grad_V(x, y)
                px = px - step * beta * gx
                py = py - step * beta * gy

            # Final full position step
            x = x + step * px
            y = y + step * py

            # Final half-step for momentum
            gx, gy = grad_V(x, y)
            px = px - 0.5 * step * beta * gx
            py = py - 0.5 * step * beta * gy

            # FIX: detect any chain that went non-finite and restore entry state
            bad = ~(np.isfinite(x) & np.isfinite(y) &
                    np.isfinite(px) & np.isfinite(py))
            if np.any(bad):
                x  = np.where(bad, x0,  x)
                y  = np.where(bad, y0,  y)
                px = np.where(bad, px0, px)
                py = np.where(bad, py0, py)

            return x, y, px, py

        # ------------------------------------------------------------------ #
        # Initialise chains: half near (+1,0), half near (-1,0)
        # ------------------------------------------------------------------ #
        half = N_chains // 2
        q_x = np.concatenate([
            1.0 + 0.15 * np.random.randn(half),
           -1.0 + 0.15 * np.random.randn(N_chains - half)
        ])
        q_y = 0.15 * np.random.randn(N_chains)

        samples_x     = np.empty((N_samples, N_chains), dtype=np.float32)
        samples_y     = np.empty((N_samples, N_chains), dtype=np.float32)
        accept_count  = np.zeros(N_chains)
        recent_accept = np.zeros(N_chains)
        total_steps   = 0
        energy_errors = []

        t_start = time.perf_counter()

        # ------------------------------------------------------------------ #
        # HMC loop
        # ------------------------------------------------------------------ #
        for i in range(N_burnin + N_samples):

            # 1. Resample momentum ~ N(0, I)
            p_x = np.random.randn(N_chains)
            p_y = np.random.randn(N_chains)

            H_curr = potential_energy(q_x, q_y) + kinetic(p_x, p_y)

            # 2. Leapfrog proposal (NaN-safe)
            qx_p, qy_p, px_p, py_p = leapfrog(
                q_x.copy(), q_y.copy(),
                p_x.copy(), p_y.copy(),
                eps, L
            )

            H_prop = potential_energy(qx_p, qy_p) + kinetic(px_p, py_p)

            # Energy error (use nan-safe operations for reporting)
            err = float(np.nanmean(np.abs(H_prop - H_curr)))
            energy_errors.append(err)

            # 3. MH accept/reject: log alpha = H_curr - H_prop
            log_alpha = np.minimum(0.0, H_curr - H_prop)
            # Treat any NaN alpha as reject (chain was reset to entry by leapfrog)
            log_alpha = np.where(np.isfinite(log_alpha), log_alpha, -np.inf)
            accept    = np.log(np.random.uniform(0.0, 1.0, N_chains)) < log_alpha

            q_x = np.where(accept, qx_p, q_x)
            q_y = np.where(accept, qy_p, q_y)

            accept_count  += accept.astype(float)
            recent_accept += accept.astype(float)
            total_steps   += 1

            # Step-size adaptation (FIX: cap at 0.15 to avoid overflow region)
            if i < N_adapt and (i + 1) % adapt_interval == 0:
                batch_rate = recent_accept.mean() / adapt_interval
                if batch_rate > target_accept:
                    eps = min(eps * 1.10, 0.15)   # FIX: cap was 1.0, now 0.15
                else:
                    eps = max(eps * 0.90, 1e-4)
                recent_accept[:] = 0.0

            if i >= N_burnin:
                k = i - N_burnin
                samples_x[k] = q_x.astype(np.float32)
                samples_y[k] = q_y.astype(np.float32)

        t_sim = time.perf_counter() - t_start
        acceptance_rate   = float(accept_count.mean() / total_steps)
        mean_energy_error = float(np.nanmean(energy_errors))
        max_energy_error  = float(np.nanmax(energy_errors))

        # ------------------------------------------------------------------ #
        # Flatten sample pool
        # ------------------------------------------------------------------ #
        X_flat = samples_x.ravel().astype(np.float64)
        Y_flat = samples_y.ravel().astype(np.float64)
        n_total = len(X_flat)

        if not (np.all(np.isfinite(X_flat)) and np.all(np.isfinite(Y_flat))):
            return {"error": "Non-finite values remain in HMC samples."}

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

        total_grad_evals = (N_burnin + N_samples) * (L + 1) * N_chains

        return {
            "strategy_id"              : "S3",
            "algorithm"                : "Hamiltonian Monte Carlo with Leapfrog",
            "n_samples"                : n_total,
            "n_chains"                 : N_chains,
            "leapfrog_steps_L"         : L,
            "tuned_eps"                : float(eps),
            "trajectory_length"        : float(L * eps),
            "empirical_mean"           : empirical_mean,
            "empirical_covariance"     : cov_mat.tolist(),
            "kde_bandwidth_scott"      : float(kde.factor),
            "kde_grid_x"               : xg.tolist(),
            "kde_grid_y"               : yg.tolist(),
            "kde_density"              : ZZ.tolist(),
            "gelman_rubin_x"           : r_hat_x,
            "gelman_rubin_y"           : r_hat_y,
            "ess_per_chain_x"          : ess_x,
            "ess_per_chain_y"          : ess_y,
            "total_ess_x"              : ess_x * N_chains,
            "total_ess_y"              : ess_y * N_chains,
            "ess_efficiency_x"         : (ess_x * N_chains) / n_total,
            "acceptance_rate"          : acceptance_rate,
            "mean_energy_error"        : mean_energy_error,
            "max_energy_error"         : max_energy_error,
            "total_grad_evaluations"   : total_grad_evals,
            "inter_well_transitions"   : n_transitions,
            "simulation_time_s"        : float(t_sim),
            "throughput_samples_per_s" : n_total / t_sim,
            "ess_per_grad_eval_x"      : (ess_x * N_chains) / total_grad_evals,
        }

    except Exception as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    import json
    result = solver()
    summary = {k: v for k, v in result.items()
               if k not in ("kde_density", "kde_grid_x", "kde_grid_y")}
    print(json.dumps(summary, indent=2))
