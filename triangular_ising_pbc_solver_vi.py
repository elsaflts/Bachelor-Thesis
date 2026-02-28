def solver():
    """
    Variational Inference for the 2D triangular-lattice Ising model
    on a periodic L×L lattice (torus) with ferromagnetic coupling J=1, h=0.

    Strategy 4 (Mean-Field) + Strategy 6 (VAN/MADE):
    -------------------------------------------------
    1. Mean-field VI: analytic self-consistency m = tanh(6βm) for baseline estimates.
    2. VAN (MADE): autoregressive neural network q_θ(σ) trained via REINFORCE,
       with importance-weighted free energy estimation.
    3. Finite-size extrapolation of VAN results for L→∞.
    """
    import numpy as np

    # =================================================================
    # STRATEGY 4: Mean-Field Variational Inference (z=6 triangular)
    # =================================================================
    print("=" * 60)
    print("STRATEGY 4: Mean-Field Variational Inference (z=6)")
    print("=" * 60)

    z = 6  # coordination number for triangular lattice
    J = 1.0
    beta_exact = 0.25 * np.log(3)  # exact β_c ≈ 0.274653
    T_c_exact = 1.0 / beta_exact   # ≈ 3.6410

    # Mean-field critical temperature: T_c^MF = zJ = 6
    T_c_mf = z * J
    beta_c_mf = 1.0 / T_c_mf

    print(f"  Exact β_c = {beta_exact:.6f}  (T_c = {T_c_exact:.4f})")
    print(f"  MF    β_c = {beta_c_mf:.6f}  (T_c = {T_c_mf:.4f})")
    print(f"  MF overestimates T_c by {(T_c_mf - T_c_exact)/T_c_exact*100:.1f}%\n")

    def mf_solve_magnetization(beta, z=6, J=1.0, tol=1e-14, max_iter=500):
        """Solve m = tanh(z*beta*J*m) via damped fixed-point iteration."""
        if beta * z * J <= 1.0:
            return 0.0
        m = 0.99
        for _ in range(max_iter):
            m_new = np.tanh(z * beta * J * m)
            if abs(m_new - m) < tol:
                return m_new
            m = 0.5 * m + 0.5 * m_new  # damping
        return m

    def mf_free_energy(beta, m, z=6, J=1.0):
        """MF free energy per spin: f = -(z/2)Jm² - T*S_binary(m)."""
        T = 1.0 / beta if beta > 0 else float('inf')
        u = -(z / 2.0) * J * m * m
        if abs(m) > 1.0 - 1e-15:
            s = 0.0
        elif abs(m) < 1e-15:
            s = np.log(2.0)
        else:
            p = (1.0 + m) / 2.0
            s = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        f = u - T * s
        return f, u, s

    # Compute MF quantities at the exact β_c
    m_mf_at_bc = mf_solve_magnetization(beta_exact)
    f_mf, u_mf, s_mf = mf_free_energy(beta_exact, m_mf_at_bc)

    print(f"  MF at exact β_c = {beta_exact:.6f}:")
    print(f"    magnetization   m = {m_mf_at_bc:.6f}")
    print(f"    energy/spin     u = {u_mf:.6f}")
    print(f"    free energy/spin f = {f_mf:.6f}")
    print(f"    entropy/spin    s = {s_mf:.6f}")

    # Sweep over temperatures for full MF phase diagram
    T_grid = np.linspace(0.5, 10.0, 200)
    mf_results = {"T": [], "m": [], "u": [], "f": [], "s": []}
    for T in T_grid:
        beta = 1.0 / T
        m = mf_solve_magnetization(beta)
        f, u, s = mf_free_energy(beta, m)
        mf_results["T"].append(float(T))
        mf_results["m"].append(float(m))
        mf_results["u"].append(float(u))
        mf_results["f"].append(float(f))
        mf_results["s"].append(float(s))

    # =================================================================
    # STRATEGY 6: Variational Autoregressive Network (VAN / MADE)
    # =================================================================
    print("\n" + "=" * 60)
    print("STRATEGY 6: Variational Autoregressive Network (MADE)")
    print("=" * 60)

    van_results = None

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

        # -------------------------------------------------------------
        # Periodic triangular lattice edge list
        # -------------------------------------------------------------
        def build_edges_torch(L):
            """Build edge list for periodic L×L triangular lattice."""
            N = L * L
            ea, eb = [], []
            for r in range(L):
                for c in range(L):
                    i = r * L + c
                    # Right
                    j = r * L + (c + 1) % L
                    ea.append(i); eb.append(j)
                    # Down
                    j = ((r + 1) % L) * L + c
                    ea.append(i); eb.append(j)
                    # Down-left
                    j = ((r + 1) % L) * L + (c - 1) % L
                    ea.append(i); eb.append(j)
            return N, torch.tensor(ea, dtype=torch.long, device=device), \
                       torch.tensor(eb, dtype=torch.long, device=device)

        # -------------------------------------------------------------
        # Masked Linear layer for MADE
        # -------------------------------------------------------------
        class MaskedLinear(nn.Module):
            def __init__(self, in_features, out_features, mask):
                super().__init__()
                self.linear = nn.Linear(in_features, out_features)
                self.register_buffer('mask', mask)

            def forward(self, x):
                return F.linear(x, self.linear.weight * self.mask,
                                self.linear.bias)

        # -------------------------------------------------------------
        # MADE autoregressive model
        # -------------------------------------------------------------
        class MADE(nn.Module):
            def __init__(self, N, hidden=128):
                super().__init__()
                self.N = N

                # Assign ordering: input units get degrees 0..N-1
                # Two hidden layers with random degrees in [0, N-1]
                m_in = torch.arange(N, device=device)
                m_h1 = torch.randint(0, N - 1, (hidden,), device=device)
                m_h2 = torch.randint(0, N - 1, (hidden,), device=device)
                m_out = torch.arange(N, device=device)

                # Mask1: h1[j] connected to input[i] if m_h1[j] >= m_in[i]
                mask1 = (m_h1.unsqueeze(1) >= m_in.unsqueeze(0)).float()
                # Mask_mid: h2[j] connected to h1[i] if m_h2[j] >= m_h1[i]
                mask_mid = (m_h2.unsqueeze(1) >= m_h1.unsqueeze(0)).float()
                # Mask2: output[k] connected to h2[j] if m_out[k] > m_h2[j]
                mask2 = (m_out.unsqueeze(1) > m_h2.unsqueeze(0)).float()

                self.layer1 = MaskedLinear(N, hidden, mask1)
                self.layer_mid = MaskedLinear(hidden, hidden, mask_mid)
                self.layer2 = MaskedLinear(hidden, N, mask2)

            def forward(self, x):
                """x: (batch, N) in {-1, +1}. Returns logits (batch, N)."""
                # Map {-1, +1} -> {0, 1} for input
                h = (x + 1.0) / 2.0
                h = torch.relu(self.layer1(h))
                h = torch.relu(self.layer_mid(h))
                logits = self.layer2(h)
                return logits

            def log_prob(self, x):
                """Compute log q_θ(x) for x in {-1, +1}^N."""
                logits = self.forward(x)
                # Target: {-1,+1} -> {0,1}
                targets = (x + 1.0) / 2.0
                # Per-site log prob, sum over sites
                log_p = -F.binary_cross_entropy_with_logits(
                    logits, targets, reduction='none'
                )
                return log_p.sum(dim=1)  # (batch,)

            def sample(self, batch_size):
                """Ancestral sampling, one spin at a time."""
                x = torch.zeros(batch_size, self.N, device=device)
                with torch.no_grad():
                    for i in range(self.N):
                        logits = self.forward(x)
                        prob_i = torch.sigmoid(logits[:, i])
                        bit = torch.bernoulli(prob_i)
                        x[:, i] = 2.0 * bit - 1.0  # map {0,1} -> {-1,+1}
                return x

        # -------------------------------------------------------------
        # Energy computation
        # -------------------------------------------------------------
        def energy_batch(spins, ea, eb):
            """H = -J Σ σ_iσ_j. Returns (batch,) tensor."""
            return -torch.sum(spins[:, ea] * spins[:, eb], dim=1).float()

        # -------------------------------------------------------------
        # Train VAN for multiple sizes, extract thermodynamic quantities
        # -------------------------------------------------------------
        van_sizes = [4, 6, 8]
        beta_c_used = beta_exact  # Use exact β_c for VI evaluation
        n_epochs = 800
        batch_size = 1024
        n_eval = 20000

        van_f_data = []
        van_u_data = []
        van_s_data = []

        for L in van_sizes:
            N, ea, eb = build_edges_torch(L)
            hidden = min(512, max(128, N * 4))
            print(f"\n  L={L}, N={N}, hidden={hidden}")

            model = MADE(N, hidden=hidden).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

            beta = beta_c_used

            # Training loop
            best_loss = float('inf')
            for epoch in range(n_epochs):
                model.train()
                spins = model.sample(batch_size)

                # Z2 symmetry augmentation: include both σ and -σ
                spins_aug = torch.cat([spins, -spins], dim=0)

                E = energy_batch(spins_aug, ea, eb)
                log_q = model.log_prob(spins_aug)

                # Variational loss: E_q[β*H + log q]
                loss_vals = beta * E + log_q
                baseline = loss_vals.mean().detach()

                # REINFORCE gradient
                loss = ((loss_vals - baseline) * log_q).mean() + log_q.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                vf = loss_vals.mean().item()
                if vf < best_loss:
                    best_loss = vf

                if (epoch + 1) % 100 == 0:
                    print(f"    epoch {epoch+1:4d}  var_free_energy={vf:.4f}")

            # Evaluate: importance-weighted estimates
            model.eval()
            with torch.no_grad():
                samples = model.sample(n_eval)
                # Z2 augmentation for evaluation too
                samples_aug = torch.cat([samples, -samples], dim=0)
                E_s = energy_batch(samples_aug, ea, eb)
                lq = model.log_prob(samples_aug)

                # Mean energy per spin (direct from q)
                u_van = E_s.mean().item() / N

                # Importance-weighted free energy: f = -(1/(Nβ)) ln Z
                # log w = -β*E - log q = log[p(σ)/q(σ)] + const
                log_w = -beta * E_s - lq
                log_Z = torch.logsumexp(log_w, 0) - np.log(len(samples_aug))
                f_van = -(1.0 / (N * beta)) * log_Z.item()

                # IS-weighted energy
                w_norm = torch.softmax(log_w, 0)
                u_van_is = (w_norm * E_s).sum().item() / N

                # Entropy per spin
                s_van = beta * (u_van_is - f_van)

                # Effective sample size
                ess = 1.0 / (w_norm ** 2).sum().item()
                ess_frac = ess / len(samples_aug)

            print(f"    Results: f={f_van:.6f}  u={u_van_is:.6f}  "
                  f"s={s_van:.6f}  ESS={ess_frac:.3f}")

            van_f_data.append(f_van)
            van_u_data.append(u_van_is)
            van_s_data.append(s_van)

        # Finite-size extrapolation for VAN
        L_arr = np.array(van_sizes, dtype=np.float64)
        A_van = np.column_stack([
            np.ones(len(van_sizes)),
            1.0 / L_arr,
            1.0 / L_arr**2,
        ])

        cf_van, _, _, _ = np.linalg.lstsq(A_van, np.array(van_f_data), rcond=None)
        cu_van, _, _, _ = np.linalg.lstsq(A_van, np.array(van_u_data), rcond=None)
        cs_van, _, _, _ = np.linalg.lstsq(A_van, np.array(van_s_data), rcond=None)

        f_van_inf = float(cf_van[0])
        u_van_inf = float(cu_van[0])
        s_van_inf = float(cs_van[0])
        s_van_thermo = beta_c_used * (u_van_inf - f_van_inf)

        print(f"\n  VAN extrapolated (L→∞):")
        print(f"    f_∞ = {f_van_inf:.6f}")
        print(f"    u_∞ = {u_van_inf:.6f}")
        print(f"    s_∞ (direct)  = {s_van_inf:.6f}")
        print(f"    s_∞ (thermo)  = {s_van_thermo:.6f}")

        van_results = {
            "method": "VAN (MADE) with Z2 augmentation",
            "beta_c_used": float(beta_c_used),
            "sizes": van_sizes,
            "f_values": [float(v) for v in van_f_data],
            "u_values": [float(v) for v in van_u_data],
            "s_values": [float(v) for v in van_s_data],
            "f_infty": f_van_inf,
            "u_infty": u_van_inf,
            "s_infty": s_van_inf,
        }

    except ImportError:
        print("\n  [PyTorch not available — skipping VAN (Strategy 6)]")
    except Exception as e:
        print(f"\n  [VAN failed: {e}]")
        import traceback
        traceback.print_exc()

    # =================================================================
    # Final summary
    # =================================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  β_c (exact)         = {beta_exact:.6f}")
    print(f"  T_c (exact)         = {T_c_exact:.4f}")
    print(f"\n  --- Mean-Field (z=6) ---")
    print(f"  T_c^MF              = {T_c_mf:.4f}")
    print(f"  energy/spin at β_c  = {u_mf:.6f}")
    print(f"  free energy/spin    = {f_mf:.6f}")
    print(f"  entropy/spin        = {s_mf:.6f}")

    if van_results is not None:
        print(f"\n  --- VAN (MADE) extrapolated ---")
        print(f"  energy/spin at β_c  = {van_results['u_infty']:.6f}")
        print(f"  free energy/spin    = {van_results['f_infty']:.6f}")
        print(f"  entropy/spin        = {van_results['s_infty']:.6f}")

    print("=" * 60)

    result = {
        "beta_c_exact": float(beta_exact),
        "T_c_exact": float(T_c_exact),
        "mean_field": {
            "T_c_MF": float(T_c_mf),
            "magnetization_at_bc": float(m_mf_at_bc),
            "energy_per_spin": float(u_mf),
            "free_energy_per_spin": float(f_mf),
            "entropy_per_spin": float(s_mf),
            "T_grid": mf_results["T"],
            "magnetization_MF": mf_results["m"],
            "energy_MF": mf_results["u"],
            "free_energy_MF": mf_results["f"],
        },
    }

    if van_results is not None:
        result["van"] = van_results

    print(f"\nFinal result: {result}")
    return result
