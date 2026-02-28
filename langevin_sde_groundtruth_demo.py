"""
Demonstration: recovering the stationary density of the overdamped Langevin SDE
purely by running Euler-Maruyama (no analytic formula used).

SDE:  dX_t = -grad V(X_t) dt + sigma dW_t
V(x,y) = (x^2-1)^2 + y^2,  sigma = 0.7

We compare:
  (A) Analytic ground truth:   pi(x,y) prop to exp(-2V/sigma^2)
  (B) EM-recovered density:    long EM run -> KDE (purely empirical)
  (C) Effect of step size:     dt=0.1 (large bias) vs dt=0.001 (paper h=10⁻³, negligible bias)
      to show the O(dt) discretization error explicitly.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time, os

BASE  = os.path.dirname(os.path.abspath(__file__))
SIGMA = 0.7
BETA  = 2.0 / SIGMA**2      # = 4.0816...

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def grad_V(x, y):
    return 4.0 * x * (x * x - 1.0), 2.0 * y

def run_EM(dt, n_chains, n_burnin, n_samples, seed=42):
    """Vectorised Euler-Maruyama over n_chains parallel chains."""
    np.random.seed(seed)
    sqrt_dt = np.sqrt(dt)
    half = n_chains // 2
    x = np.concatenate([
        1.0 + 0.15 * np.random.randn(half),
       -1.0 + 0.15 * np.random.randn(n_chains - half)
    ])
    y = 0.15 * np.random.randn(n_chains)

    sx = np.empty((n_samples, n_chains), dtype=np.float32)
    sy = np.empty((n_samples, n_chains), dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(n_burnin + n_samples):
        gx, gy = grad_V(x, y)
        x = x - gx * dt + SIGMA * sqrt_dt * np.random.randn(n_chains)
        y = y - gy * dt + SIGMA * sqrt_dt * np.random.randn(n_chains)
        if i >= n_burnin:
            k = i - n_burnin
            sx[k] = x.astype(np.float32)
            sy[k] = y.astype(np.float32)

    elapsed = time.perf_counter() - t0
    X = sx.ravel().astype(np.float64)
    Y = sy.ravel().astype(np.float64)
    return X, Y, elapsed

def make_kde(X, Y, n_sub=200_000):
    """Fit a 2D KDE on a random subsample."""
    idx = np.random.choice(len(X), min(n_sub, len(X)), replace=False)
    return gaussian_kde(np.vstack([X[idx], Y[idx]]), bw_method='scott')

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation grid
# ─────────────────────────────────────────────────────────────────────────────
xg = np.linspace(-2.5, 2.5, 200)
yg = np.linspace(-2.2, 2.2, 200)
XX, YY = np.meshgrid(xg, yg)
pts    = np.vstack([XX.ravel(), YY.ravel()])

# --- (A) Analytic ground truth (used ONLY for comparison here) ---------------
V_grid  = (XX**2 - 1.0)**2 + YY**2
ZZ_true = np.exp(-BETA * V_grid)
ZZ_true_norm = ZZ_true / ZZ_true.max()

# Analytic x-marginal: pi_x(x) prop exp(-beta*(x^2-1)^2), normalised
x1d    = np.linspace(-2.5, 2.5, 500)
log_px = -BETA * (x1d**2 - 1)**2;  log_px -= log_px.max()
px_true = np.exp(log_px)
px_true /= np.trapz(px_true, x1d)

# Analytic y-marginal: N(0, sigma^2/4)
y1d    = np.linspace(-2.2, 2.2, 500)
THEORY_VAR_Y = SIGMA**2 / 4.0
py_true = np.exp(-0.5 * y1d**2 / THEORY_VAR_Y)
py_true /= np.trapz(py_true, y1d)

# ─────────────────────────────────────────────────────────────────────────────
# (B) Long EM run with small dt -> should match analytic ground truth closely
# ─────────────────────────────────────────────────────────────────────────────
print("Running long EM (dt=0.001 = h=10⁻³ as in paper, 5M samples) to recover ground truth ...")
X_fine, Y_fine, t_fine = run_EM(
    dt=0.001, n_chains=200, n_burnin=20_000, n_samples=25_000
)
n_fine = len(X_fine)
print(f"  {n_fine:,} samples in {t_fine:.2f}s")
kde_fine = make_kde(X_fine, Y_fine)
ZZ_fine  = kde_fine(pts).reshape(XX.shape)
ZZ_fine_norm = ZZ_fine / ZZ_fine.max()

# ─────────────────────────────────────────────────────────────────────────────
# (C) Compare three step sizes to visualise discretization bias
# ─────────────────────────────────────────────────────────────────────────────
configs = [
    (0.001, 200, 10_000,  5_000,  "dt=0.001  (paper: h=10⁻³, negligible bias)"),
    (0.025, 200,  2_000,  5_000,  "dt=0.025  (S1 ULA, moderate bias)"),
    (0.10,  200,  2_000,  5_000,  "dt=0.10   (large bias)"),
]

kdes_dt = {}
for dt, nc, nb, ns, label in configs:
    print(f"Running EM  {label} ...")
    X_, Y_, t_ = run_EM(dt=dt, n_chains=nc, n_burnin=nb, n_samples=ns)
    k_ = make_kde(X_, Y_)
    Z_ = k_(pts).reshape(XX.shape)
    kdes_dt[label] = {
        "Z": Z_ / Z_.max(),
        "X": X_, "Y": Y_,
        "time": t_, "n": len(X_),
        "var_y": float(np.var(Y_)),
        "mean_x": float(np.mean(X_)),
        "dt": dt,
    }
    print(f"  {len(X_):,} samples, Var(y)={kdes_dt[label]['var_y']:.4f}  "
          f"(theory {THEORY_VAR_Y:.4f}), mean(x)={kdes_dt[label]['mean_x']:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — EM-recovered density vs. analytic ground truth (small dt)
# ═════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
fig1.suptitle(
    r"Ground Truth Recovery via Euler-Maruyama (dt=0.001 = $h=10^{-3}$, 5M samples)"
    "\nNo analytic formula used in sampling — KDE built purely from SDE trajectory",
    fontsize=11, fontweight='bold'
)

# Panel 1: Analytic ground truth
cf = axes[0].contourf(XX, YY, ZZ_true_norm, levels=40, cmap='plasma')
axes[0].contour(XX, YY, ZZ_true_norm, levels=10, colors='w', lw=0.5, alpha=0.4)
plt.colorbar(cf, ax=axes[0], label='Norm. density')
axes[0].scatter([1,-1],[0,0], c='red', s=60, zorder=5)
axes[0].scatter([0],[0], c='yellow', s=60, zorder=5, marker='^')
axes[0].set_title("(A) Analytic ground truth\n"
                  r"$\pi \propto \exp(-\beta V)$, evaluated on grid", fontsize=10)
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')

# Panel 2: EM-recovered density
cf2 = axes[1].contourf(XX, YY, ZZ_fine_norm, levels=40, cmap='viridis')
axes[1].contour(XX, YY, ZZ_fine_norm, levels=10, colors='w', lw=0.5, alpha=0.4)
plt.colorbar(cf2, ax=axes[1], label='Norm. density')
axes[1].scatter([1,-1],[0,0], c='red', s=60, zorder=5)
axes[1].scatter([0],[0], c='yellow', s=60, zorder=5, marker='^')
axes[1].set_title(f"(B) EM-recovered density\n"
                  f"{n_fine/1e6:.1f}M samples, " r"dt=0.001 ($h=10^{-3}$), KDE only", fontsize=10)
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')

# Panel 3: Absolute difference |A - B|
diff = np.abs(ZZ_true_norm - ZZ_fine_norm)
cf3 = axes[2].contourf(XX, YY, diff, levels=30, cmap='hot_r')
plt.colorbar(cf3, ax=axes[2], label='|True - EM-KDE|')
axes[2].set_title(f"(C) Pointwise error |A − B|\n"
                  f"max error = {diff.max():.4f}", fontsize=10)
axes[2].set_xlabel('x'); axes[2].set_ylabel('y')

plt.tight_layout()
out1 = os.path.join(BASE, "plot_gt_recovery_main.png")
fig1.savefig(out1, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out1}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Effect of step size on discretization bias (contours + marginals)
# ═════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 4, figsize=(18, 9))
fig2.suptitle(
    "Effect of Euler-Maruyama Step Size on Recovered Density\n"
    "Larger dt -> O(dt) discretization bias -> density deviates from true pi",
    fontsize=12, fontweight='bold'
)

cmaps = ["Blues", "Greens", "Reds"]

# ── Row 0: 2D contours ───────────────────────────────────────────────────────
# True density reference
cf_t = axes2[0, 0].contourf(XX, YY, ZZ_true_norm, levels=35, cmap='plasma')
axes2[0, 0].contour(XX, YY, ZZ_true_norm, levels=8, colors='w', lw=0.5, alpha=0.4)
plt.colorbar(cf_t, ax=axes2[0,0], label='Density')
axes2[0, 0].scatter([1,-1],[0,0], c='red', s=50, zorder=5)
axes2[0, 0].set_title("Analytic ground truth", fontsize=10)
axes2[0, 0].set_xlabel('x'); axes2[0, 0].set_ylabel('y')

for col, (label, d) in enumerate(kdes_dt.items(), start=1):
    ax = axes2[0, col]
    Z  = d["Z"]
    cfi = ax.contourf(XX, YY, Z, levels=35, cmap=cmaps[col-1])
    ax.contour(XX, YY, Z, levels=8, colors='w', lw=0.5, alpha=0.4)
    # True density as dashed overlay
    ax.contour(XX, YY, ZZ_true_norm, levels=6, colors='black',
               linewidths=1.0, linestyles='--', alpha=0.7)
    plt.colorbar(cfi, ax=ax, label='Density')
    ax.scatter([1,-1],[0,0], c='red', s=50, zorder=5)
    ax.set_title(
        f"EM: {label}\n"
        f"Var(y)={d['var_y']:.4f}  (theory {THEORY_VAR_Y:.4f})",
        fontsize=9
    )
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.text(0.02, 0.97, "dashed = true pi", transform=ax.transAxes,
            fontsize=7, va='top')

# ── Row 1: X and Y marginals vs. theory ──────────────────────────────────────
# X marginal
ax_x = axes2[1, 0]
ax_x.plot(x1d, px_true, 'k-', lw=2.5, label='Analytic truth')
colors_dt = ['steelblue', 'seagreen', 'crimson']
for (label, d), c in zip(kdes_dt.items(), colors_dt):
    kde_   = make_kde(d["X"], d["Y"])
    px_em_ = np.array([kde_([xi, 0.0])[0] for xi in x1d])  # slice at y=0... no
    # Marginalise over y using the KDE evaluated on the grid
    Z_  = kde_(pts).reshape(XX.shape)
    px_ = np.trapz(Z_, yg, axis=0)
    px_ /= np.trapz(px_, xg)
    ax_x.plot(xg, px_, color=c, lw=1.8, ls='--', label=label.split('(')[0].strip())
ax_x.set_xlabel('x', fontsize=11); ax_x.set_ylabel('Density', fontsize=10)
ax_x.set_title('X Marginal vs. Theory', fontsize=10)
ax_x.legend(fontsize=8); ax_x.grid(alpha=0.3)

# Y marginal
ax_y = axes2[1, 1]
ax_y.plot(y1d, py_true, 'k-', lw=2.5, label=f'Analytic N(0,{THEORY_VAR_Y:.4f})')
for (label, d), c in zip(kdes_dt.items(), colors_dt):
    kde_   = make_kde(d["X"], d["Y"])
    Z_     = kde_(pts).reshape(XX.shape)
    py_    = np.trapz(Z_, xg, axis=1)
    py_   /= np.trapz(py_, yg)
    ax_y.plot(yg, py_, color=c, lw=1.8, ls='--', label=label.split('(')[0].strip())
ax_y.set_xlabel('y', fontsize=11); ax_y.set_ylabel('Density', fontsize=10)
ax_y.set_title('Y Marginal vs. Theory', fontsize=10)
ax_y.legend(fontsize=8); ax_y.grid(alpha=0.3)

# Var(y) vs dt bar chart — shows O(dt) bias directly
ax_bias = axes2[1, 2]
dt_vals  = [d["dt"] for d in kdes_dt.values()]
vary_vals = [d["var_y"] for d in kdes_dt.values()]
bars = ax_bias.bar([str(dt) for dt in dt_vals], vary_vals,
                   color=colors_dt, edgecolor='k')
ax_bias.axhline(THEORY_VAR_Y, color='black', lw=2, ls='--',
                label=f'Theory {THEORY_VAR_Y:.4f}')
for b, v in zip(bars, vary_vals):
    ax_bias.text(b.get_x() + b.get_width()/2, v + 0.001,
                 f"{v:.4f}", ha='center', fontsize=9)
ax_bias.set_xlabel('Step size dt', fontsize=10)
ax_bias.set_ylabel('Empirical Var(y)', fontsize=10)
ax_bias.set_title('Var(y) vs. dt  — O(dt) bias\n(theory = σ²/4 = 0.1225)', fontsize=10)
ax_bias.legend(fontsize=9); ax_bias.grid(axis='y', alpha=0.3)

# Absolute density error vs dt
ax_err = axes2[1, 3]
for (label, d), c in zip(kdes_dt.items(), colors_dt):
    Z_ = d["Z"]
    err_ = np.abs(ZZ_true_norm - Z_)
    ax_err.bar(label.split('(')[0].strip(), err_.mean(),
               color=c, edgecolor='k',
               label=f"max={err_.max():.3f}")
ax_err.set_ylabel('Mean |True - KDE|', fontsize=10)
ax_err.set_title('Mean Pointwise KDE Error vs. dt', fontsize=10)
ax_err.legend(fontsize=8); ax_err.grid(axis='y', alpha=0.3)

plt.tight_layout()
out2 = os.path.join(BASE, "plot_gt_recovery_bias.png")
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Saved: {out2}")

# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP SIZE vs. RECOVERY ACCURACY")
print("="*65)
print(f"{'dt':<8} {'Var(y)':<10} {'|Var(y)-theory|':<18} {'mean(x)':<12} {'n_samples'}")
print("-"*65)
print(f"{'analytic':<8} {THEORY_VAR_Y:<10.4f} {'0.0000':<18} {'0.0000':<12} {'exact (formula)'}")
for label, d in kdes_dt.items():
    dt_s = str(d['dt'])
    print(f"{dt_s:<8} {d['var_y']:<10.4f} "
          f"{abs(d['var_y']-THEORY_VAR_Y):<18.5f} "
          f"{d['mean_x']:<12.4f} {d['n']:,}")
print("="*65)
print("\nConclusion: as dt -> 0, EM-KDE density converges to the true stationary density.")
print("The O(dt) bias is visible in Var(y): larger dt -> larger deviation from theory.")
print("Paper specifies h=10⁻³; dt=0.001 gives negligible bias.")
