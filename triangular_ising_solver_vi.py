def solver():
    """
    Variational Inference for the 2D triangular-lattice Ising model
    on a triangle-shaped patch with free boundary conditions.

    Strategy: Variational Autoregressive Network (VAN)
    - MADE-style masked autoregressive network parameterises q_θ(σ)
    - Minimise reverse KL via REINFORCE with EMA baseline
    - Free energy from importance-weighted Z estimate
    - Internal energy from IS-weighted expectation with ESS monitoring
    - Finite-size extrapolation for bulk limits
    """
    import numpy as np

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        return {"error": "PyTorch is required for the VI (VAN) solver."}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    beta_c = 0.25 * np.log(3)  # ≈ 0.274653

    # -----------------------------------------------------------------
    # Lattice construction
    # -----------------------------------------------------------------
    def build_lattice(N):
        S = N * (N + 1) // 2

        def idx(r, c):
            return r * (r + 1) // 2 + c

        edges_a, edges_b = [], []
        for r in range(N):
            for c in range(r + 1):
                i = idx(r, c)
                if c < r:
                    j = idx(r, c + 1)
                    edges_a.append(i)
                    edges_b.append(j)
                if r < N - 1:
                    j = idx(r + 1, c)
                    edges_a.append(i)
                    edges_b.append(j)
                if r < N - 1:
                    j = idx(r + 1, c + 1)
                    edges_a.append(i)
                    edges_b.append(j)

        ea = torch.tensor(edges_a, dtype=torch.long, device=device)
        eb = torch.tensor(edges_b, dtype=torch.long, device=device)
        return S, ea, eb

    # -----------------------------------------------------------------
    # Masked linear layer (no in-place weight mutation)
    # -----------------------------------------------------------------
    class MaskedLinear(nn.Module):
        def __init__(self, in_features, out_features, mask):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.register_buffer('mask', mask)

        def forward(self, x):
            return F.linear(x, self.linear.weight * self.mask,
                            self.linear.bias)

    # -----------------------------------------------------------------
    # MADE autoregressive model
    # -----------------------------------------------------------------
    class MADE(nn.Module):
        def __init__(self, S, hidden_sizes):
            super().__init__()
            self.S = S

            # Ordering assignments for masking
            m = {-1: torch.arange(S, device=device)}
            for l, h in enumerate(hidden_sizes):
                m[l] = torch.randint(0, S - 1, (h,), device=device)
            m[len(hidden_sizes)] = torch.arange(S, device=device)

            # Build masked layers
            layers = []
            dims = [S] + list(hidden_sizes) + [S]
            for l in range(len(dims) - 1):
                if l == 0:
                    mask = (m[0].unsqueeze(1) >= m[-1].unsqueeze(0)).float()
                elif l == len(dims) - 2:
                    # Strict inequality for output: autoregressive property
                    mask = (m[l].unsqueeze(1) > m[l - 1].unsqueeze(0)).float()
                else:
                    mask = (m[l].unsqueeze(1) >= m[l - 1].unsqueeze(0)).float()
                layers.append(MaskedLinear(dims[l], dims[l + 1], mask))

            self.layers = nn.ModuleList(layers)

        def forward(self, x):
            h = x
            for l, layer in enumerate(self.layers):
                h = layer(h)
                if l < len(self.layers) - 1:
                    h = torch.relu(h)
            return h

        def log_prob(self, x):
            """log q_θ(σ) for completed configs. x: (batch, S) in {-1,+1}."""
            logits = self.forward(x)
            labels = (x + 1.0) / 2.0
            log_p = -F.binary_cross_entropy_with_logits(
                logits, labels, reduction='none')
            return log_p.sum(dim=1)

        @torch.no_grad()
        def sample(self, batch_size):
            """Ancestral sampling, returns (samples, log_probs)."""
            x = torch.zeros(batch_size, self.S, device=device)
            log_probs = torch.zeros(batch_size, device=device)

            for i in range(self.S):
                logits = self.forward(x)
                logit_i = logits[:, i]
                prob_i = torch.sigmoid(logit_i)
                # Clamp to avoid log(0)
                prob_i = prob_i.clamp(1e-7, 1.0 - 1e-7)

                u = torch.rand(batch_size, device=device)
                s_i = torch.where(u < prob_i,
                                  torch.ones(batch_size, device=device),
                                  -torch.ones(batch_size, device=device))
                x[:, i] = s_i

                label_i = (s_i + 1.0) / 2.0
                lp_i = -F.binary_cross_entropy_with_logits(
                    logit_i, label_i, reduction='none')
                log_probs += lp_i

            return x, log_probs

    # -----------------------------------------------------------------
    # Energy computation
    # -----------------------------------------------------------------
    def compute_energy_sum(sigma, edges_a, edges_b):
        return (sigma[:, edges_a] * sigma[:, edges_b]).sum(dim=1)

    # -----------------------------------------------------------------
    # Train VAN for a given lattice size N
    # -----------------------------------------------------------------
    def train_van(N, n_epochs, batch_size, lr, hidden_sizes):
        S, ea, eb = build_lattice(N)

        model = MADE(S, hidden_sizes).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=n_epochs, eta_min=lr * 0.01)

        # EMA baseline for REINFORCE variance reduction
        ema_baseline = None
        ema_alpha = 0.05

        best_vfe = float('inf')
        patience_counter = 0
        patience = 500

        for epoch in range(n_epochs):
            model.train()

            # Sample from q_θ (no grad for discrete sampling)
            sigma, log_q_no_grad = model.sample(batch_size)

            # Recompute log q_θ WITH gradients on detached samples
            log_q = model.log_prob(sigma)

            # Energy of sampled configs
            energy_sum = compute_energy_sum(sigma, ea, eb)
            log_p_unnorm = beta_c * energy_sum

            # REINFORCE with EMA baseline
            with torch.no_grad():
                signal = log_q_no_grad - log_p_unnorm
                batch_mean = signal.mean().item()
                if ema_baseline is None:
                    ema_baseline = batch_mean
                else:
                    ema_baseline = (ema_alpha * batch_mean +
                                    (1 - ema_alpha) * ema_baseline)
                advantage = signal - ema_baseline

            # Score-function gradient: ∇_θ E_q[f] = E_q[f · ∇_θ log q]
            loss = (advantage * log_q).mean()

            with torch.no_grad():
                vfe = batch_mean

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            if vfe < best_vfe - 1e-5:
                best_vfe = vfe
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience and epoch > 1500:
                print(f"    Early stop at epoch {epoch}, VFE = {vfe:.4f}")
                break

            if (epoch + 1) % 1000 == 0:
                print(f"    Epoch {epoch+1:5d}  VFE = {vfe:.4f}  "
                      f"best = {best_vfe:.4f}")

        # ----- Extract estimates -----
        model.eval()
        with torch.no_grad():
            n_eval_batches = 8
            eval_batch = max(300, batch_size)
            all_log_q = []
            all_energy_sum = []

            for _ in range(n_eval_batches):
                sigma, log_q = model.sample(eval_batch)
                esum = compute_energy_sum(sigma, ea, eb)
                all_log_q.append(log_q)
                all_energy_sum.append(esum)

            all_log_q = torch.cat(all_log_q)
            all_energy_sum = torch.cat(all_energy_sum)

            # Importance weights: log w = log p̃ - log q = β Σσσ - log q
            log_w = beta_c * all_energy_sum - all_log_q

            # IS estimate of ln Z using log-sum-exp
            max_lw = log_w.max().item()
            ln_Z_iw = max_lw + np.log(
                torch.exp(log_w - max_lw).mean().item())
            f_N = ln_Z_iw / S

            # IS-weighted internal energy with ESS check
            w_norm = torch.softmax(log_w, dim=0)
            ess = 1.0 / (w_norm ** 2).sum().item()
            u_N_is = (w_norm * all_energy_sum).sum().item() / S

            # Unweighted (direct q) estimate as fallback
            u_N_q = all_energy_sum.mean().item() / S

            # Use IS estimate if ESS is reasonable, else blend
            if ess > 20:
                u_N = u_N_is
            elif ess > 5:
                # Blend between IS and direct q estimates
                alpha = (ess - 5) / 15.0
                u_N = alpha * u_N_is + (1 - alpha) * u_N_q
            else:
                u_N = u_N_q
                print(f"    Warning: very low ESS ({ess:.1f}), "
                      f"using direct q estimate for u")

        print(f"  N={N:3d}  S={S:4d}  f={f_N:.6f}  u={u_N:.6f}  "
              f"ESS={ess:.0f}/{len(all_log_q)}")
        return f_N, u_N

    # -----------------------------------------------------------------
    # Run for multiple system sizes
    # -----------------------------------------------------------------
    sizes = [4, 6, 8, 10, 12, 14]
    print(f"beta_c = {beta_c:.10f}")
    print(f"System sizes N: {sizes}\n")

    f_data = []
    u_data = []

    for N in sizes:
        S = N * (N + 1) // 2
        # Scale architecture and training to system size
        h_dim = min(max(48, 2 * S), 200)
        # More epochs for small systems (cheap), fewer for large
        n_ep = max(2000, min(4000, int(40000 / S)))
        bs = min(1000, max(300, 4 * S))

        print(f"Training VAN for N={N} (S={S}, hidden={h_dim}, "
              f"epochs={n_ep}, batch={bs})...")
        f_N, u_N = train_van(
            N, n_epochs=n_ep, batch_size=bs, lr=5e-4,
            hidden_sizes=[h_dim, h_dim])
        f_data.append(f_N)
        u_data.append(u_N)

    # -----------------------------------------------------------------
    # Finite-size extrapolation: q(N) = q_∞ + a/N + b/N²
    # -----------------------------------------------------------------
    sizes_arr = np.array(sizes, dtype=np.float64)
    f_arr = np.array(f_data)
    u_arr = np.array(u_data)

    A = np.column_stack([
        np.ones(len(sizes)),
        1.0 / sizes_arr,
        1.0 / sizes_arr**2,
    ])

    cf, _, _, _ = np.linalg.lstsq(A, f_arr, rcond=None)
    cu, _, _, _ = np.linalg.lstsq(A, u_arr, rcond=None)

    f_infty = float(cf[0])
    u_infty = float(cu[0])

    f_fit = A @ cf
    u_fit = A @ cu
    f_residual = float(np.std(f_arr - f_fit))
    u_residual = float(np.std(u_arr - u_fit))

    print()
    print("=" * 60)
    print("FINITE-SIZE EXTRAPOLATION (VAN)  q = q_inf + a/N + b/N^2")
    print("=" * 60)
    print(f"  f_infty = {f_infty:.8f}   (fit residual std: {f_residual:.2e})")
    print(f"  u_infty = {u_infty:.8f}   (fit residual std: {u_residual:.2e})")
    print("=" * 60)

    result = {
        "f_infty_est": f_infty,
        "u_infty_est": u_infty,
        "beta_c": float(beta_c),
        "sizes_used": sizes,
        "f_values": [float(v) for v in f_data],
        "u_values": [float(v) for v in u_data],
        "f_fit_residual_std": f_residual,
        "u_fit_residual_std": u_residual,
    }

    print(f"\nFinal result: {result}")
    return result


solver()
