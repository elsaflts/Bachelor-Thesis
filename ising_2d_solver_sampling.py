"""
2D Ising Model — Sampling-based Inference using the Wolff Cluster Algorithm
Strategy 2 (score 9): Wolff single-cluster MCMC

Uses a C-compiled Wolff kernel via ctypes for performance.
Falls back to a small pure-Python run if compilation fails.
Computes: f(T), T_c, u(T), c(T), m(T) compared against exact Onsager/Yang.
"""

import numpy as np
from scipy.special import ellipk, ellipe
from scipy import integrate
import time
import os
import ctypes
import tempfile
import subprocess


def solver():
    J = 1.0
    T_c_exact = 2.0 * J / np.log(1.0 + np.sqrt(2.0))
    beta_c_exact = 1.0 / T_c_exact

    # ── Exact Onsager/Yang solutions ─────────────────────────────────────
    def exact_energy(T):
        K = J / T
        kappa = 2.0 * np.sinh(2.0 * K) / np.cosh(2.0 * K)**2
        kappa_safe = np.clip(np.abs(kappa), 0, 1.0 - 1e-15)
        K1 = ellipk(kappa_safe**2)
        return -J / np.tanh(2.0 * K) * (
            1.0 + (2.0 / np.pi) * (2.0 * np.tanh(2.0 * K)**2 - 1.0) * K1
        )

    def exact_specific_heat(T):
        K = J / T
        kappa = 2.0 * np.sinh(2.0 * K) / np.cosh(2.0 * K)**2
        kappa_safe = np.clip(np.abs(kappa), 0, 1.0 - 1e-15)
        K1 = ellipk(kappa_safe**2)
        E1 = ellipe(kappa_safe**2)
        coth2K = 1.0 / np.tanh(2.0 * K)
        tanh2K = np.tanh(2.0 * K)
        return (2.0 / np.pi) * (K * coth2K)**2 * (
            2.0 * (K1 - E1)
            - (1.0 - tanh2K**2) * (np.pi / 2.0 + (2.0 * tanh2K**2 - 1.0) * K1)
        )

    def exact_magnetization(T):
        K = J / T
        s = np.sinh(2.0 * K)
        return (1.0 - s**(-4))**(1.0 / 8.0) if T < T_c_exact else 0.0

    def exact_free_energy(T):
        K = J / T
        kap = 2.0 * np.sinh(2.0 * K) / np.cosh(2.0 * K)**2
        kap_safe = min(abs(kap), 1.0 - 1e-15)
        def integrand(phi):
            return np.log(0.5 * (1.0 + np.sqrt(1.0 - kap_safe**2 * np.sin(phi)**2)))
        I, _ = integrate.quad(integrand, 0, np.pi / 2)
        neg_beta_f = np.log(2.0 * np.cosh(2.0 * K)) + I / np.pi
        return -T * neg_beta_f

    # ── Try to compile a C Wolff kernel for speed ────────────────────────
    c_source = r"""
#include <stdlib.h>
#include <math.h>
#include <string.h>

static unsigned long xorshift_state = 123456789UL;
static unsigned long xorshift64() {
    xorshift_state ^= xorshift_state << 13;
    xorshift_state ^= xorshift_state >> 7;
    xorshift_state ^= xorshift_state << 17;
    return xorshift_state;
}
static double rand_double() {
    return (xorshift64() & 0xFFFFFFFFFFFFFULL) / (double)(0x10000000000000ULL);
}

void seed_rng(unsigned long s) { xorshift_state = s ? s : 1; }

/* Wolff simulation: run n_steps cluster flips, then n_measure more,
   recording energy and |magnetization| per spin every skip steps.
   spins: L*L array of +1/-1 (int8). neighbors: N*4 array (int32).
   out_e, out_m: output arrays. Returns number of measurements. */
int wolff_run(signed char *spins, int *neighbors, int N, int L,
              double p_add, int n_equil, int n_measure, int skip,
              double *out_e, double *out_m, double *out_chi) {
    int *stack = (int*)malloc(N * sizeof(int));
    char *in_cluster = (char*)calloc(N, 1);

    /* Helper: one Wolff step, returns cluster size */
    #define WOLFF_STEP(csize_out) { \
        int seed = (int)(rand_double() * N); \
        if (seed >= N) seed = N-1; \
        signed char cspin = spins[seed]; \
        memset(in_cluster, 0, N); \
        in_cluster[seed] = 1; \
        int top = 0; \
        stack[top++] = seed; \
        int csize = 0; \
        while (top > 0) { \
            int si = stack[--top]; \
            csize++; \
            for (int d = 0; d < 4; d++) { \
                int j = neighbors[si*4+d]; \
                if (!in_cluster[j] && spins[j] == cspin && rand_double() < p_add) { \
                    in_cluster[j] = 1; \
                    stack[top++] = j; \
                } \
            } \
        } \
        for (int i = 0; i < N; i++) { \
            if (in_cluster[i]) spins[i] = -spins[i]; \
        } \
        csize_out = csize; \
    }

    /* Equilibration */
    int dummy;
    for (int s = 0; s < n_equil; s++) { WOLFF_STEP(dummy); }

    /* Measurement */
    int n_meas = 0;
    for (int s = 0; s < n_measure; s++) {
        int cs;
        WOLFF_STEP(cs);
        if (s % skip == 0) {
            /* Compute energy */
            double E = 0;
            for (int i = 0; i < N; i++) {
                int j1 = neighbors[i*4+0]; /* right */
                int j2 = neighbors[i*4+2]; /* up */
                E -= (double)(spins[i] * spins[j1] + spins[i] * spins[j2]);
            }
            /* Magnetization */
            int M = 0;
            for (int i = 0; i < N; i++) M += spins[i];
            out_e[n_meas] = E / N;
            out_m[n_meas] = abs(M) / (double)N;
            out_chi[n_meas] = cs / (double)N;
            n_meas++;
        }
    }
    free(stack);
    free(in_cluster);
    return n_meas;
}
"""

    use_c = False
    lib = None
    tmpdir = tempfile.mkdtemp()
    c_path = os.path.join(tmpdir, "wolff.c")
    so_path = os.path.join(tmpdir, "wolff.so")

    try:
        with open(c_path, "w") as f:
            f.write(c_source)
        ret = subprocess.run(
            ["gcc", "-O3", "-shared", "-fPIC", "-o", so_path, c_path, "-lm"],
            capture_output=True, timeout=30
        )
        if ret.returncode == 0:
            lib = ctypes.CDLL(so_path)
            lib.seed_rng(ctypes.c_ulong(np.random.randint(1, 2**32)))
            lib.wolff_run.restype = ctypes.c_int
            lib.wolff_run.argtypes = [
                ctypes.POINTER(ctypes.c_int8),    # spins
                ctypes.POINTER(ctypes.c_int32),    # neighbors
                ctypes.c_int, ctypes.c_int,        # N, L
                ctypes.c_double,                   # p_add
                ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n_equil, n_measure, skip
                ctypes.POINTER(ctypes.c_double),   # out_e
                ctypes.POINTER(ctypes.c_double),   # out_m
                ctypes.POINTER(ctypes.c_double),   # out_chi
            ]
            use_c = True
            print("C Wolff kernel compiled successfully.")
    except Exception as e:
        print(f"C compilation failed ({e}), using pure Python fallback.")

    # ── Simulation parameters ────────────────────────────────────────────
    if use_c:
        L = 32
        n_equilib = 2000
        n_measure = 10000
        n_skip = 1
    else:
        L = 8
        n_equilib = 200
        n_measure = 1000
        n_skip = 1

    N = L * L

    T_grid = np.concatenate([
        np.linspace(0.5, 2.0, 6, endpoint=False),
        np.linspace(2.0, 2.6, 10, endpoint=False),
        np.linspace(2.6, 4.5, 6)
    ])

    # Precompute neighbors
    neighbors = np.empty((N, 4), dtype=np.int32)
    for i in range(N):
        x, y = divmod(i, L)
        neighbors[i, 0] = ((x + 1) % L) * L + y
        neighbors[i, 1] = ((x - 1) % L) * L + y
        neighbors[i, 2] = x * L + (y + 1) % L
        neighbors[i, 3] = x * L + (y - 1) % L

    # ── Pure Python fallback ─────────────────────────────────────────────
    def wolff_step_py(spins_flat, p_add):
        seed = np.random.randint(N)
        cluster_spin = spins_flat[seed]
        in_cluster = np.zeros(N, dtype=np.bool_)
        in_cluster[seed] = True
        stack = [seed]
        csize = 0
        while stack:
            si = stack.pop()
            csize += 1
            for d in range(4):
                j = neighbors[si, d]
                if not in_cluster[j] and spins_flat[j] == cluster_spin:
                    if np.random.random() < p_add:
                        in_cluster[j] = True
                        stack.append(j)
        spins_flat[in_cluster] *= -1
        return csize

    def run_python(T_val):
        beta = 1.0 / T_val
        p_add = 1.0 - np.exp(-2.0 * beta * J)
        spins = np.ones(N, dtype=np.int8) if T_val < T_c_exact else \
                np.random.choice(np.array([-1, 1], dtype=np.int8), size=N)
        for _ in range(n_equilib):
            wolff_step_py(spins, p_add)
        e_list, m_list, chi_list = [], [], []
        for step in range(n_measure):
            cs = wolff_step_py(spins, p_add)
            if step % n_skip == 0:
                E = 0.0
                for i in range(N):
                    E -= J * (spins[i] * spins[neighbors[i, 0]] + spins[i] * spins[neighbors[i, 2]])
                e_list.append(E / N)
                m_list.append(np.abs(np.sum(spins)) / N)
                chi_list.append(cs / N)
        return np.array(e_list), np.array(m_list), np.array(chi_list)

    # ── Run simulation ───────────────────────────────────────────────────
    print(f"2D Ising Model — Wolff Cluster Algorithm ({'C kernel' if use_c else 'Python'})")
    print(f"Lattice: {L}x{L}, J={J}")
    print(f"Exact T_c = {T_c_exact:.6f}")
    print(f"Equilib: {n_equilib}, Measure: {n_measure}")
    print(f"{'T':>7s} {'u_MC':>10s} {'u_exact':>10s} {'m_MC':>10s} {'m_exact':>10s} {'c_MC':>10s} {'c_exact':>10s}")
    print("-" * 75)

    t_start = time.time()
    all_T, all_u, all_u_ex, all_c, all_c_ex = [], [], [], [], []
    all_m, all_m_ex, all_chi, all_f_ex = [], [], [], []

    max_meas = n_measure // n_skip + 1

    for T_val in T_grid:
        beta = 1.0 / T_val
        p_add = 1.0 - np.exp(-2.0 * beta * J)

        if use_c:
            spins = np.ones(N, dtype=np.int8) if T_val < T_c_exact else \
                    np.random.choice(np.array([-1, 1], dtype=np.int8), size=N)
            out_e = np.zeros(max_meas)
            out_m = np.zeros(max_meas)
            out_chi = np.zeros(max_meas)
            nm = lib.wolff_run(
                spins.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                neighbors.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                N, L, p_add, n_equilib, n_measure, n_skip,
                out_e.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_chi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            e_arr = out_e[:nm]
            m_arr = out_m[:nm]
            chi_arr_t = out_chi[:nm]
        else:
            e_arr, m_arr, chi_arr_t = run_python(T_val)

        u_mean = np.mean(e_arr)
        c_mc = beta**2 * N * np.var(e_arr)
        m_mean = np.mean(m_arr)
        chi_mean = np.mean(chi_arr_t)

        u_ex = exact_energy(T_val)
        c_ex = exact_specific_heat(T_val)
        m_ex = exact_magnetization(T_val)
        f_ex = exact_free_energy(T_val)

        all_T.append(T_val)
        all_u.append(u_mean); all_u_ex.append(u_ex)
        all_c.append(c_mc); all_c_ex.append(c_ex)
        all_m.append(m_mean); all_m_ex.append(m_ex)
        all_chi.append(chi_mean); all_f_ex.append(f_ex)

        print(f"{T_val:7.3f} {u_mean:10.5f} {u_ex:10.5f} {m_mean:10.5f} {m_ex:10.5f} {c_mc:10.4f} {c_ex:10.4f}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")

    all_T = np.array(all_T)
    all_u = np.array(all_u)
    all_c = np.array(all_c)
    all_m = np.array(all_m)
    all_chi = np.array(all_chi)
    all_u_ex = np.array(all_u_ex)
    all_c_ex = np.array(all_c_ex)
    all_m_ex = np.array(all_m_ex)
    all_f_ex = np.array(all_f_ex)

    # Free energy via thermodynamic integration
    beta_arr = 1.0 / all_T
    sort_idx = np.argsort(beta_arr)
    beta_sorted = beta_arr[sort_idx]
    u_sorted = all_u[sort_idx]
    f_mc = np.zeros(len(all_T))
    for i in range(len(beta_sorted)):
        integral = np.trapezoid(u_sorted[:i + 1], beta_sorted[:i + 1])
        beta_f = -np.log(2) + integral
        f_mc[sort_idx[i]] = beta_f / beta_sorted[i] if beta_sorted[i] > 1e-10 else -all_T[sort_idx[i]] * np.log(2)

    # Estimate T_c from peaks
    T_c_chi = all_T[np.argmax(all_chi)]
    T_c_c = all_T[np.argmax(all_c)]

    u_rmse = np.sqrt(np.mean((all_u - all_u_ex)**2))
    m_rmse = np.sqrt(np.mean((all_m - all_m_ex)**2))
    f_rmse = np.sqrt(np.mean((f_mc - all_f_ex)**2))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Exact T_c:                 {T_c_exact:.6f}")
    print(f"Estimated T_c (chi peak):  {T_c_chi:.4f}")
    print(f"Estimated T_c (c peak):    {T_c_c:.4f}")
    print(f"u(T_c) exact = -√2 =       {-np.sqrt(2):.6f}")
    print(f"RMSE(u):                   {u_rmse:.6f}")
    print(f"RMSE(m):                   {m_rmse:.6f}")
    print(f"RMSE(f):                   {f_rmse:.6f}")

    output = {
        "model": f"2D Ising Model (L={L}, J={J})",
        "method": "Wolff Cluster Algorithm (Strategy 2)",
        "T_c_exact": float(T_c_exact),
        "beta_c_exact": float(beta_c_exact),
        "T_c_estimated_chi": float(T_c_chi),
        "T_c_estimated_c": float(T_c_c),
        "T_grid": all_T.tolist(),
        "energy_per_spin_MC": all_u.tolist(),
        "energy_per_spin_exact": all_u_ex.tolist(),
        "specific_heat_MC": all_c.tolist(),
        "specific_heat_exact": all_c_ex.tolist(),
        "magnetization_MC": all_m.tolist(),
        "magnetization_exact": all_m_ex.tolist(),
        "free_energy_MC": f_mc.tolist(),
        "free_energy_exact": all_f_ex.tolist(),
        "susceptibility_MC": all_chi.tolist(),
        "diagnostics": {
            "RMSE_energy": float(u_rmse),
            "RMSE_magnetization": float(m_rmse),
            "RMSE_free_energy": float(f_rmse),
            "lattice_size": L,
            "n_equilib": n_equilib,
            "n_measure": n_measure,
            "runtime_seconds": float(elapsed),
            "used_c_kernel": use_c,
        }
    }

    print("\nInference completed successfully.")
    return output


if __name__ == "__main__":
    result = solver()
