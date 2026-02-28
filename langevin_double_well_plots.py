"""
Langevin Double-Well — Comprehensive Plots
Generates all figures requested in the original problem statement:
  1. 2D KDE contour plots (one per strategy + true density)
  2. 2D surface plots of estimated density
  3. Marginal distributions (x and y) vs. theory
  4. Convergence diagnostics summary bar chart
  5. Computational efficiency comparison
"""

import importlib.util
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401 (registers 3d projection)
from scipy.stats import gaussian_kde

BASE   = os.path.dirname(os.path.abspath(__file__))
SIGMA  = 0.7
BETA   = 2.0 / SIGMA**2      # ~4.0816
THEORY_VAR_Y = SIGMA**2 / 4.0   # 0.1225

def _gt_em(dt=0.001, n_chains=200, n_burnin=20_000, n_samples=15_000, seed=2025):
    """
    Vectorised Euler-Maruyama ground truth.
    dt = 0.001 = h = 10^{-3} as specified in the paper.
    n_burnin = 20 000 steps  ->  20 time units of burn-in
    n_chains * n_samples     ->  3 000 000 samples total
    """
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
    for i in range(n_burnin + n_samples):
        gx = 4.0 * x * (x * x - 1.0)
        gy = 2.0 * y
        x = x - gx * dt + SIGMA * sqrt_dt * np.random.randn(n_chains)
        y = y - gy * dt + SIGMA * sqrt_dt * np.random.randn(n_chains)
        if i >= n_burnin:
            k = i - n_burnin
            sx[k] = x.astype(np.float32)
            sy[k] = y.astype(np.float32)
    return sx.ravel().astype(np.float64), sy.ravel().astype(np.float64)

# ─────────────────────────────────────────────────────────────────────────────
# Load results from each solver
# ─────────────────────────────────────────────────────────────────────────────
results = {}
for sid, fname in [
    ("S1", "langevin_double_well_solver_S1.py"),
    ("S2", "langevin_double_well_solver_S2.py"),
    ("S3", "langevin_double_well_solver_S3.py"),
]:
    path = os.path.join(BASE, fname)
    spec = importlib.util.spec_from_file_location(f"solver_{sid}", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"Running {sid} ...", flush=True)
    results[sid] = mod.solver()
    n = results[sid].get("n_samples", "?")
    print(f"  {sid} done  (n={n:,})")

# ─────────────────────────────────────────────────────────────────────────────
# Shared grid for all density plots
# ─────────────────────────────────────────────────────────────────────────────
xg  = np.array(results["S1"]["kde_grid_x"])
yg  = np.array(results["S1"]["kde_grid_y"])
XX, YY = np.meshgrid(xg, yg)

# Numerical ground truth: long EM run with dt=0.001 (paper h=10^{-3}) -> KDE
print("Computing numerical ground truth (EM, dt=0.001, 3M samples) ...", flush=True)
_X_gt, _Y_gt = _gt_em()
_n_gt    = len(_X_gt)
GT_VAR_Y = float(np.var(_Y_gt))
GT_LABEL = r"GT: EM  $h=10^{-3}$" + f"  ({_n_gt // 1_000_000:.0f}M samples)"
_idx_gt  = np.random.default_rng(0).choice(_n_gt, min(200_000, _n_gt), replace=False)
_kde_gt  = gaussian_kde(
    np.vstack([_X_gt[_idx_gt], _Y_gt[_idx_gt]]), bw_method='scott'
)
ZZ_true  = _kde_gt(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
ZZ_true /= ZZ_true.max()
print(f"  GT done  (Var(y)={GT_VAR_Y:.4f},  theory σ²/4={THEORY_VAR_Y:.4f})", flush=True)

def kde_arr(sid):
    return np.array(results[sid]["kde_density"])

# Normalise each KDE to peak=1 for visual comparability
def norm(Z):
    m = Z.max()
    return Z / m if m > 0 else Z

ZZ = {s: norm(kde_arr(s)) for s in ("S1", "S2", "S3")}

COLORS  = {"S1": "Blues", "S2": "Greens", "S3": "Oranges"}
LABELS  = {
    "S1": "S1 — ULA (Euler-Maruyama)",
    "S2": "S2 — MALA",
    "S3": "S3 — HMC / Leapfrog",
}

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 2D Contour Plots: each strategy + true density
# ═════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(14, 11))
fig1.suptitle(
    r"Overdamped Langevin Double-Well  —  $V(x,y)=(x^2{-}1)^2+y^2,\;\sigma=0.7$"
    "\n2D KDE Estimated Stationary Density (contour)",
    fontsize=13, fontweight='bold'
)

panel_data = [
    ("S1", axes[0, 0]),
    ("S2", axes[0, 1]),
    ("S3", axes[1, 0]),
    (None, axes[1, 1]),   # true density
]

for sid, ax in panel_data:
    if sid is None:
        Z  = ZZ_true
        cm_use = "plasma"
        title  = GT_LABEL
    else:
        Z  = ZZ[sid]
        cm_use = COLORS[sid]
        title  = LABELS[sid]

    cf = ax.contourf(XX, YY, Z, levels=40, cmap=cm_use)
    ax.contour(XX, YY, Z, levels=10, colors='white', linewidths=0.5, alpha=0.4)
    plt.colorbar(cf, ax=ax, label='Normalised density')

    # Mark modes and saddle
    ax.scatter([1, -1], [0, 0], c='red',    s=70, zorder=5, label='Modes (±1,0)')
    ax.scatter([0],    [0],    c='yellow', s=70, zorder=5, marker='^', label='Saddle (0,0)')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, loc='upper right')

plt.tight_layout()
out1 = os.path.join(BASE, "plot1_contour_2d.png")
fig1.savefig(out1, dpi=150, bbox_inches='tight')
print(f"  Saved: {out1}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — 3D Surface Plots of KDE estimated density
# ═════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(18, 5))
fig2.suptitle(
    "2D Surface Plots of KDE Estimated Stationary Density",
    fontsize=13, fontweight='bold'
)

# Downsample grid for 3D rendering clarity
step = 2
XXs, YYs = XX[::step, ::step], YY[::step, ::step]

for k, sid in enumerate(("S1", "S2", "S3")):
    ax3d = fig2.add_subplot(1, 3, k + 1, projection='3d')
    Zs   = ZZ[sid][::step, ::step]
    surf = ax3d.plot_surface(XXs, YYs, Zs, cmap=COLORS[sid],
                              linewidth=0, antialiased=True, alpha=0.9)
    ax3d.set_xlabel('x', fontsize=9, labelpad=4)
    ax3d.set_ylabel('y', fontsize=9, labelpad=4)
    ax3d.set_zlabel('Density', fontsize=9, labelpad=4)
    ax3d.set_title(LABELS[sid], fontsize=10)
    ax3d.view_init(elev=30, azim=-60)
    fig2.colorbar(surf, ax=ax3d, shrink=0.5, label='Norm. density')

plt.tight_layout()
out2 = os.path.join(BASE, "plot2_surface_3d.png")
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"  Saved: {out2}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Marginal distributions vs. theory
# ═════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle("Marginal Distributions vs. Theory", fontsize=13, fontweight='bold')

# Numerical GT marginals: marginalise the GT KDE grid (already peak-normalised)
px_true = np.trapz(ZZ_true, yg, axis=0)
px_true /= np.trapz(px_true, xg)

py_true = np.trapz(ZZ_true, xg, axis=1)
py_true /= np.trapz(py_true, yg)

ax_x, ax_y = axes3

# ── X marginal ──────────────────────────────────────────────────────────────
LINE_STYLES = {"S1": ("steelblue",  "-",  0.85),
               "S2": ("seagreen",   "--", 0.85),
               "S3": ("darkorange", ":",  0.95)}

for sid in ("S1", "S2", "S3"):
    Z = ZZ[sid]  # shape (len(yg), len(xg))
    # Marginalise KDE grid over y by trapezoidal integration
    px_kde = np.trapz(Z, yg, axis=0)
    px_kde /= np.trapz(px_kde, xg)
    c, ls, lw = LINE_STYLES[sid]
    r = results[sid]
    ax_x.plot(xg, px_kde, color=c, ls=ls, lw=lw*1.8,
              label=f"{sid}  (R̂={r['gelman_rubin_x']:.3f})")

ax_x.plot(xg, px_true, 'k-', lw=2.5, label=GT_LABEL)
ax_x.set_xlabel('x', fontsize=12)
ax_x.set_ylabel('Density', fontsize=12)
ax_x.set_title('X Marginal', fontsize=12)
ax_x.legend(fontsize=9)
ax_x.grid(alpha=0.3)

# ── Y marginal ──────────────────────────────────────────────────────────────
for sid in ("S1", "S2", "S3"):
    Z = ZZ[sid]
    py_kde = np.trapz(Z, xg, axis=1)
    py_kde /= np.trapz(py_kde, yg)
    c, ls, lw = LINE_STYLES[sid]
    r = results[sid]
    ax_y.plot(yg, py_kde, color=c, ls=ls, lw=lw*1.8,
              label=f"{sid}  Var(y)={r['empirical_covariance'][1][1]:.4f}")

ax_y.plot(yg, py_true, 'k-', lw=2.5, label=GT_LABEL)
ax_y.set_xlabel('y', fontsize=12)
ax_y.set_ylabel('Density', fontsize=12)
ax_y.set_title(
    f'Y Marginal  (GT Var(y)={GT_VAR_Y:.4f},  theory σ²/4={THEORY_VAR_Y:.4f})',
    fontsize=12
)
ax_y.legend(fontsize=9)
ax_y.grid(alpha=0.3)

plt.tight_layout()
out3 = os.path.join(BASE, "plot3_marginals.png")
fig3.savefig(out3, dpi=150, bbox_inches='tight')
print(f"  Saved: {out3}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Convergence diagnostics comparison
# ═════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(2, 3, figsize=(15, 8))
fig4.suptitle("Convergence Diagnostics & Efficiency Comparison", fontsize=13, fontweight='bold')

strategies   = ["S1", "S2", "S3"]
short_labels = ["S1\n(ULA)", "S2\n(MALA)", "S3\n(HMC)"]
bar_colors   = ["steelblue", "seagreen", "darkorange"]

def bar(ax, values, ylabel, title, lower_better=False, ref_line=None, fmt=".4f"):
    bars = ax.bar(short_labels, values, color=bar_colors, edgecolor='k', linewidth=0.8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    if ref_line is not None:
        ax.axhline(ref_line, color='red', lw=1.5, ls='--', label=f'ref={ref_line}')
        ax.legend(fontsize=8)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
                f"{v:{fmt}}", ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, max(values) * 1.20)
    ax.grid(axis='y', alpha=0.3)

# ── R-hat x ─────────────────────────────────────────────────────────────────
rhat_x = [results[s]["gelman_rubin_x"] for s in strategies]
bar(axes4[0,0], rhat_x, "R̂(x)", "Gelman-Rubin R̂(x)  [lower=better]",
    lower_better=True, ref_line=1.10, fmt=".4f")

# ── Total ESS(x) ─────────────────────────────────────────────────────────────
total_ess_x = [results[s]["total_ess_x"] for s in strategies]
bar(axes4[0,1], total_ess_x, "Total ESS(x)", "Total ESS(x)  [higher=better]", fmt=".0f")

# ── ESS(x) per second ───────────────────────────────────────────────────────
ess_per_sec = [results[s]["total_ess_x"] / results[s]["simulation_time_s"]
               for s in strategies]
bar(axes4[0,2], ess_per_sec, "ESS(x)/sec", "ESS(x) per Second  [higher=better]", fmt=".0f")

# ── Inter-well transitions per 1000 samples ─────────────────────────────────
trans_rate = [results[s]["inter_well_transitions"] /
              (results[s]["n_samples"] / 1000.0) for s in strategies]
bar(axes4[1,0], trans_rate, "Transitions/1000 samples",
    "Inter-well Mixing Rate  [higher=better]", fmt=".2f")

# ── |mean(x)| ───────────────────────────────────────────────────────────────
mean_err = [abs(results[s]["empirical_mean"][0]) for s in strategies]
bar(axes4[1,1], mean_err, "|mean(x)|", "|Empirical mean(x) - 0|  [lower=better]",
    lower_better=True, fmt=".4f")

# ── Simulation time ─────────────────────────────────────────────────────────
sim_times = [results[s]["simulation_time_s"] for s in strategies]
bar(axes4[1,2], sim_times, "Seconds", "Wall-clock Runtime  [lower=better]",
    lower_better=True, fmt=".2f")

plt.tight_layout()
out4 = os.path.join(BASE, "plot4_diagnostics.png")
fig4.savefig(out4, dpi=150, bbox_inches='tight')
print(f"  Saved: {out4}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Side-by-side KDE vs. True density overlay (per strategy)
# ═════════════════════════════════════════════════════════════════════════════
fig5, axes5 = plt.subplots(1, 3, figsize=(17, 5))
fig5.suptitle(
    "KDE Estimated Density vs. True Density  (overlaid contours)",
    fontsize=13, fontweight='bold'
)

for k, sid in enumerate(("S1", "S2", "S3")):
    ax = axes5[k]
    Z  = ZZ[sid]
    r  = results[sid]

    # KDE density filled
    cf = ax.contourf(XX, YY, Z, levels=30, cmap=COLORS[sid], alpha=0.75)
    plt.colorbar(cf, ax=ax, shrink=0.8, label='KDE density')

    # True density outline
    ax.contour(XX, YY, ZZ_true, levels=8, colors='black',
               linewidths=1.2, linestyles='--', alpha=0.85)

    ax.scatter([1, -1], [0, 0], c='red',    s=60, zorder=6)
    ax.scatter([0],    [0],    c='yellow', s=60, zorder=6, marker='^')

    rhat = r["gelman_rubin_x"]
    ess  = r["total_ess_x"]
    t    = r["simulation_time_s"]
    ax.set_title(
        f"{LABELS[sid]}\n"
        f"R̂(x)={rhat:.3f}  ESS(x)={ess:,.0f}  t={t:.2f}s",
        fontsize=9
    )
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.text(0.02, 0.97, r"dashed = GT (EM, $h=10^{-3}$)", transform=ax.transAxes,
            fontsize=7, va='top', color='black')

plt.tight_layout()
out5 = os.path.join(BASE, "plot5_kde_vs_true.png")
fig5.savefig(out5, dpi=150, bbox_inches='tight')
print(f"  Saved: {out5}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Benchmark summary radar / scatter
# ═════════════════════════════════════════════════════════════════════════════
fig6, axes6 = plt.subplots(1, 2, figsize=(13, 5))
fig6.suptitle("Benchmark Summary", fontsize=13, fontweight='bold')

# Scatter: ESS(x)/sec vs R-hat(x)
ax_sc = axes6[0]
for sid, c in zip(strategies, bar_colors):
    r = results[sid]
    ax_sc.scatter(
        r["gelman_rubin_x"],
        r["total_ess_x"] / r["simulation_time_s"],
        s=200, color=c, zorder=5, label=LABELS[sid], edgecolors='k'
    )
    ax_sc.annotate(
        sid,
        (r["gelman_rubin_x"], r["total_ess_x"] / r["simulation_time_s"]),
        textcoords="offset points", xytext=(8, 4), fontsize=10, fontweight='bold'
    )
ax_sc.axvline(1.10, color='red', ls='--', lw=1.2, label='R̂=1.10 threshold')
ax_sc.set_xlabel("Gelman-Rubin R̂(x)  [lower=better]", fontsize=11)
ax_sc.set_ylabel("ESS(x) per second  [higher=better]", fontsize=11)
ax_sc.set_title("Efficiency vs. Convergence Trade-off", fontsize=11)
ax_sc.legend(fontsize=8)
ax_sc.grid(alpha=0.3)

# Bar: mixing transitions per 1000 samples
ax_mx = axes6[1]
for i, (sid, c) in enumerate(zip(strategies, bar_colors)):
    r    = results[sid]
    rate = r["inter_well_transitions"] / (r["n_samples"] / 1000.0)
    b    = ax_mx.bar(short_labels[i], rate, color=c, edgecolor='k', linewidth=0.8)
    ax_mx.text(i, rate * 1.03, f"{rate:.2f}", ha='center', fontsize=9)
ax_mx.set_ylabel("Inter-well transitions / 1000 samples", fontsize=11)
ax_mx.set_title("Bimodal Mixing Quality", fontsize=11)
ax_mx.grid(axis='y', alpha=0.3)

plt.tight_layout()
out6 = os.path.join(BASE, "plot6_benchmark_summary.png")
fig6.savefig(out6, dpi=150, bbox_inches='tight')
print(f"  Saved: {out6}")

print("\nAll plots saved successfully.")

# ─────────────────────────────────────────────────────────────────────────────
# Print numerical benchmark table for the report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("BENCHMARK METRICS TABLE")
print("="*72)
header = f"{'Metric':<38} {'S1 (ULA)':>10} {'S2 (MALA)':>10} {'S3 (HMC)':>10}"
print(header)
print("-"*72)

rows = [
    ("Verification pass",
     [str(True)]*3),
    ("R̂(x)  [lower=better]",
     [f"{results[s]['gelman_rubin_x']:.4f}" for s in strategies]),
    ("R̂(y)  [lower=better]",
     [f"{results[s]['gelman_rubin_y']:.4f}" for s in strategies]),
    ("ESS(x)/sec  [higher=better]",
     [f"{results[s]['total_ess_x']/results[s]['simulation_time_s']:,.0f}" for s in strategies]),
    ("Transitions/1000 samples  [higher]",
     [f"{results[s]['inter_well_transitions']/(results[s]['n_samples']/1000):.2f}" for s in strategies]),
    ("|mean(x)|  [lower=better]",
     [f"{abs(results[s]['empirical_mean'][0]):.4f}" for s in strategies]),
    ("|Var(y) - theory|  [lower=better]",
     [f"{abs(results[s]['empirical_covariance'][1][1] - THEORY_VAR_Y):.5f}" for s in strategies]),
    ("Runtime (s)  [lower=better]",
     [f"{results[s]['simulation_time_s']:.2f}" for s in strategies]),
    ("n_samples",
     [f"{results[s]['n_samples']:,}" for s in strategies]),
    ("Acceptance rate",
     [results[s].get('acceptance_rate', 'N/A') if isinstance(results[s].get('acceptance_rate'), str)
      else (f"{results[s]['acceptance_rate']:.3f}" if 'acceptance_rate' in results[s] else 'N/A')
      for s in strategies]),
]

for name, vals in rows:
    print(f"{name:<38} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

print("="*72)
