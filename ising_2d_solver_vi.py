"""
2D Ising Model — Variational Inference
Implements both:
  - Strategy 6 (VAN): Variational Autoregressive Network with MADE (small lattice)
  - Strategy 4 (Mean-Field): Weiss mean-field as fast baseline
Computes: f(T), u(T), c(T), m(T) compared against exact Onsager/Yang.
"""

import numpy as np
from scipy.special import ellipk, ellipe
from scipy import integrate
import time
import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def solver():
    J = 1.0
    T_c_exact = 2.0 * J / np.log(1.0 + np.sqrt(2.0))

    # ── Exact Onsager/Yang ───────────────────────────────────────────────
    def exact_energy(T):
        K = J / T
        kap = 2.0 * np.sinh(2*K) / np.cosh(2*K)**2
        ks = np.clip(np.abs(kap), 0, 1-1e-15)
        K1 = ellipk(ks**2)
        return -J/np.tanh(2*K) * (1 + (2/np.pi)*(2*np.tanh(2*K)**2 - 1)*K1)

    def exact_specific_heat(T):
        K = J / T
        kap = 2.0*np.sinh(2*K)/np.cosh(2*K)**2
        ks = np.clip(np.abs(kap), 0, 1-1e-15)
        K1, E1 = ellipk(ks**2), ellipe(ks**2)
        c2K, t2K = 1/np.tanh(2*K), np.tanh(2*K)
        return (2/np.pi)*(K*c2K)**2*(2*(K1-E1)-(1-t2K**2)*(np.pi/2+(2*t2K**2-1)*K1))

    def exact_magnetization(T):
        K = J / T
        s = np.sinh(2*K)
        return (1-s**(-4))**(1/8) if T < T_c_exact else 0.0

    def exact_free_energy(T):
        K = J / T
        kap = 2.0*np.sinh(2*K)/np.cosh(2*K)**2
        ks = min(abs(kap), 1-1e-15)
        def integ(phi): return np.log(0.5*(1+np.sqrt(1-ks**2*np.sin(phi)**2)))
        I, _ = integrate.quad(integ, 0, np.pi/2)
        return -T * (np.log(2*np.cosh(2*K)) + I/np.pi)

    # Temperature grid
    T_grid = np.array([1.0, 1.5, 2.0, 2.1, 2.2, 2.269, 2.35, 2.5, 2.8, 3.2, 3.8, 4.5])

    # ══════════════════════════════════════════════════════════════════════
    # STRATEGY 4: Mean-Field Variational Inference (fast, analytical)
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("STRATEGY 4: Mean-Field Variational Inference")
    print("=" * 70)

    z = 4  # coordination number for square lattice
    T_c_mf = z * J  # mean-field T_c = 4.0 (vs exact 2.269)

    def mean_field_solve(T):
        """Solve m = tanh(z*beta*J*m) self-consistently."""
        beta = 1.0 / T
        if T >= T_c_mf:
            return 0.0
        # Newton iteration
        m = 0.99 if T < T_c_mf * 0.5 else 0.5
        for _ in range(500):
            arg = z * beta * J * m
            m_new = np.tanh(arg)
            if abs(m_new - m) < 1e-14:
                break
            # Damped update for stability
            m = 0.5 * m + 0.5 * m_new
        return m

    mf_results = {'T': [], 'm': [], 'u': [], 'f': [], 'c': []}

    print(f"Mean-field T_c = {T_c_mf:.4f}  (exact: {T_c_exact:.4f})")
    print(f"{'T':>7s} {'m_MF':>10s} {'m_exact':>10s} {'u_MF':>10s} {'u_exact':>10s} {'f_MF':>10s} {'f_exact':>10s}")
    print("-" * 75)

    for T in T_grid:
        m = mean_field_solve(T)
        # Mean-field energy per spin: u = -(z/2)*J*m^2
        u_mf = -(z / 2) * J * m**2
        # Mean-field entropy per spin
        if abs(m) < 1e-15:
            s_mf = np.log(2)
        else:
            p = (1 + m) / 2
            s_mf = -p * np.log(p) - (1 - p) * np.log(1 - p)
        # Mean-field free energy: f = u - T*s
        f_mf = u_mf - T * s_mf

        mf_results['T'].append(T)
        mf_results['m'].append(m)
        mf_results['u'].append(u_mf)
        mf_results['f'].append(f_mf)

        m_ex = exact_magnetization(T)
        u_ex = exact_energy(T)
        f_ex = exact_free_energy(T)
        print(f"{T:7.3f} {m:10.5f} {m_ex:10.5f} {u_mf:10.5f} {u_ex:10.5f} {f_mf:10.5f} {f_ex:10.5f}")

    # Mean-field specific heat via numerical differentiation
    T_dense = np.linspace(0.5, 5.0, 500)
    u_dense = np.array([-(z/2)*J*mean_field_solve(t)**2 for t in T_dense])
    c_dense = np.gradient(u_dense, T_dense)
    # Interpolate to our grid
    for T in T_grid:
        idx = np.argmin(np.abs(T_dense - T))
        mf_results['c'].append(c_dense[idx])

    # ══════════════════════════════════════════════════════════════════════
    # STRATEGY 6: VAN with MADE (if PyTorch available)
    # ══════════════════════════════════════════════════════════════════════
    van_results = None

    if TORCH_AVAILABLE:
        print("\n" + "=" * 70)
        print("STRATEGY 6: Variational Autoregressive Network (MADE)")
        print("=" * 70)

        L = 4   # very small lattice for tractability
        N = L * L

        # Neighbor table
        neighbors = np.empty((N, 4), dtype=np.int64)
        for i in range(N):
            x, y = divmod(i, L)
            neighbors[i, 0] = ((x+1)%L)*L + y
            neighbors[i, 1] = ((x-1)%L)*L + y
            neighbors[i, 2] = x*L + (y+1)%L
            neighbors[i, 3] = x*L + (y-1)%L
        nb = torch.from_numpy(neighbors).long()

        def energy_batch(spins):
            """Energy for batch of {-1,+1} configs. spins: (B, N)"""
            return -J * torch.sum(spins*spins[:, nb[:, 0]] + spins*spins[:, nb[:, 2]], dim=1)

        class MADE(nn.Module):
            def __init__(self, n_in, hidden=64):
                super().__init__()
                self.n_in = n_in
                # Simple 2-layer MADE
                self.fc1 = nn.Linear(n_in, hidden)
                self.fc2 = nn.Linear(hidden, hidden)
                self.fc3 = nn.Linear(hidden, n_in)
                # Create autoregressive masks
                m0 = torch.arange(n_in).float()
                m1 = torch.arange(hidden).float() % (n_in - 1)
                m2 = torch.arange(hidden).float() % (n_in - 1)
                self.register_buffer('mask1', (m0.unsqueeze(0) <= m1.unsqueeze(1)).float())
                self.register_buffer('mask2', (m1.unsqueeze(0) <= m2.unsqueeze(1)).float())
                self.register_buffer('mask3', (m2.unsqueeze(0) < m0.unsqueeze(1)).float())

            def forward(self, x):
                h = torch.relu(torch.mm(x, (self.fc1.weight * self.mask1).t()) + self.fc1.bias)
                h = torch.relu(torch.mm(h, (self.fc2.weight * self.mask2).t()) + self.fc2.bias)
                return torch.mm(h, (self.fc3.weight * self.mask3).t()) + self.fc3.bias

            def log_prob(self, x):
                logits = self.forward(x)
                targets = (x + 1) / 2  # {-1,+1} -> {0,1}
                return -nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction='none').sum(1)

            def sample(self, batch_size):
                x = torch.zeros(batch_size, self.n_in)
                with torch.no_grad():
                    for i in range(self.n_in):
                        logits = self.forward(x)
                        p = torch.sigmoid(logits[:, i])
                        x[:, i] = 2 * torch.bernoulli(p) - 1
                return x

        van_results = {'T': [], 'f': [], 'u': [], 'm': []}

        # Use fewer temperatures for VAN (it's slow)
        T_van_grid = np.array([1.0, 1.5, 2.0, 2.2, 2.4, 2.8, 3.5, 4.5])

        print(f"Lattice: {L}x{L}, N={N}")
        print(f"{'T':>7s} {'f_VAN':>10s} {'f_exact':>10s} {'u_VAN':>10s} {'u_exact':>10s} {'m_VAN':>10s} {'m_exact':>10s}")
        print("-" * 75)

        for T_val in T_van_grid:
            beta = 1.0 / T_val
            model = MADE(N, hidden=64)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            batch_size = 500
            n_epochs = 300

            # Training
            for epoch in range(n_epochs):
                spins = model.sample(batch_size)
                # Z2 symmetry augmentation
                spins_aug = torch.cat([spins, -spins], dim=0)
                E = energy_batch(spins_aug)
                log_q = model.log_prob(spins_aug)

                # Variational loss: E_q[beta*E + log q]
                loss_vals = beta * E + log_q
                baseline = loss_vals.mean().detach()
                loss = ((loss_vals - baseline) * log_q).mean() + log_q.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Evaluate
            with torch.no_grad():
                n_eval = 5000
                samples = model.sample(n_eval)
                E_s = energy_batch(samples)
                lq = model.log_prob(samples)

                u_van = E_s.mean().item() / N
                m_van = torch.abs(samples.sum(1)).mean().item() / N

                # Importance-weighted free energy
                log_w = -beta * E_s - lq
                log_Z = torch.logsumexp(log_w, 0) - np.log(n_eval)
                f_van = -T_val * log_Z.item() / N

            f_ex = exact_free_energy(T_val)
            u_ex = exact_energy(T_val)
            m_ex = exact_magnetization(T_val)

            van_results['T'].append(T_val)
            van_results['f'].append(f_van)
            van_results['u'].append(u_van)
            van_results['m'].append(m_van)

            print(f"{T_val:7.3f} {f_van:10.5f} {f_ex:10.5f} {u_van:10.5f} {u_ex:10.5f} {m_van:10.5f} {m_ex:10.5f}")

    else:
        print("\n[PyTorch not available — skipping VAN (Strategy 6)]")

    # ── Assemble output ──────────────────────────────────────────────────
    exact_vals = {
        'T': T_grid.tolist(),
        'f': [exact_free_energy(t) for t in T_grid],
        'u': [exact_energy(t) for t in T_grid],
        'c': [exact_specific_heat(t) for t in T_grid],
        'm': [exact_magnetization(t) for t in T_grid],
    }

    output = {
        "model": "2D Ising Model",
        "T_c_exact": float(T_c_exact),
        "mean_field": {
            "method": "Weiss Mean-Field VI (Strategy 4)",
            "T_c_MF": float(T_c_mf),
            "T_grid": mf_results['T'],
            "magnetization_MF": mf_results['m'],
            "energy_MF": mf_results['u'],
            "free_energy_MF": mf_results['f'],
            "specific_heat_MF": mf_results['c'],
        },
        "exact": exact_vals,
    }

    if van_results is not None:
        output["van"] = {
            "method": "Variational Autoregressive Network (MADE, Strategy 6)",
            "lattice_size": 4,
            "T_grid": van_results['T'],
            "free_energy_VAN": van_results['f'],
            "energy_VAN": van_results['u'],
            "magnetization_VAN": van_results['m'],
        }

    # Diagnostics
    mf_u_rmse = np.sqrt(np.mean((np.array(mf_results['u']) - np.array(exact_vals['u']))**2))
    mf_m_rmse = np.sqrt(np.mean((np.array(mf_results['m']) - np.array(exact_vals['m']))**2))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Exact T_c:    {T_c_exact:.6f}")
    print(f"MF T_c:       {T_c_mf:.6f} (error: {abs(T_c_mf - T_c_exact)/T_c_exact*100:.1f}%)")
    print(f"MF RMSE(u):   {mf_u_rmse:.4f}")
    print(f"MF RMSE(m):   {mf_m_rmse:.4f}")

    if van_results is not None:
        van_u_ex = [exact_energy(t) for t in van_results['T']]
        van_u_rmse = np.sqrt(np.mean((np.array(van_results['u']) - np.array(van_u_ex))**2))
        print(f"VAN RMSE(u):  {van_u_rmse:.4f}")

    output["diagnostics"] = {
        "MF_RMSE_energy": float(mf_u_rmse),
        "MF_RMSE_magnetization": float(mf_m_rmse),
    }

    print("\nInference completed successfully.")
    return output


if __name__ == "__main__":
    result = solver()
