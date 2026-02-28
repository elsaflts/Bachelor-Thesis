"""
Reproducibility and Statistical Reliability Test

Evaluates the stability, reproducibility, and statistical reliability of the
thermally averaged charge density estimator by running the full solver pipeline
multiple times with independent randomness.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time

# Import solver components
from solver import (
    WaterMolecule,
    compute_potential_energy,
    compute_charge_density,
    create_density_grid,
    SurrogateModel,
    LangevinSampler,
    select_by_uncertainty,
    compute_importance_weights,
    compute_ess_ratio,
    StrategyMonitor,
    KB
)


@dataclass
class RunResult:
    """Results from a single solver run."""
    run_id: int
    seed: int
    rho_bar: np.ndarray
    total_charge: float
    expected_charge: int
    ess: float
    ess_ratio: float
    n_samples: int
    surrogate_rmse: float
    surrogate_rmse_kT: float
    weight_max: float
    weight_min: float
    weight_std: float
    runtime_seconds: float
    converged: bool
    error_message: Optional[str] = None


def run_solver_with_seed(seed: int, run_id: int, verbose: bool = False) -> RunResult:
    """
    Run the complete solver pipeline with a specific random seed.

    Args:
        seed: Random seed for reproducibility
        run_id: Identifier for this run
        verbose: Whether to print progress

    Returns:
        RunResult containing all metrics and the density estimate
    """
    start_time = time.time()

    # Set random seed
    np.random.seed(seed)

    # System parameters
    molecule = WaterMolecule()
    temperature = 300.0
    beta = 1.0 / (KB * temperature)
    kT = KB * temperature

    # Algorithm parameters
    N_init = 20
    N_iter = 10
    M_candidates = 50
    K_select = 5
    langevin_steps = 500
    langevin_thin = 10
    n_grid_points = 15
    grid_padding = 2.5

    # Initialize monitor
    monitor = StrategyMonitor(kT)

    if verbose:
        print(f"  Run {run_id} (seed={seed}): Starting...")

    try:
        # Phase 1: Generate initial configurations
        R_eq = molecule.get_equilibrium_geometry()
        configurations = []
        true_energies = []

        for i in range(N_init):
            displacement = np.random.randn(3, 3) * 0.05
            R = R_eq + displacement
            U = compute_potential_energy(R, molecule)
            configurations.append(R)
            true_energies.append(U)

        true_energies = np.array(true_energies)

        # Check for numerical issues
        if np.any(np.isnan(true_energies)) or np.any(np.isinf(true_energies)):
            return RunResult(
                run_id=run_id, seed=seed, rho_bar=np.array([]),
                total_charge=np.nan, expected_charge=molecule.n_electrons,
                ess=0, ess_ratio=0, n_samples=0,
                surrogate_rmse=np.nan, surrogate_rmse_kT=np.nan,
                weight_max=np.nan, weight_min=np.nan, weight_std=np.nan,
                runtime_seconds=time.time() - start_time,
                converged=False, error_message="NaN/Inf in initial energies"
            )

        # Phase 2: Train initial surrogate
        surrogate = SurrogateModel(molecule.n_atoms)
        surrogate.fit(configurations, true_energies)

        # Phase 3: Adaptive sampling loop
        sampler = LangevinSampler(molecule, temperature, dt=0.0005, friction=1.0)
        all_samples = list(configurations)
        all_true_energies = list(true_energies)
        pred_energies = surrogate.predict_batch(configurations)
        all_surrogate_energies = list(pred_energies)

        for iteration in range(N_iter):
            R_start = all_samples[-1].copy()

            def surrogate_potential(R):
                return surrogate.predict(R)

            candidates, _ = sampler.run(
                R_start, surrogate_potential,
                n_steps=langevin_steps, thin=langevin_thin
            )

            selected_idx, _ = select_by_uncertainty(
                candidates, surrogate, min(K_select, len(candidates))
            )

            selected_configs = [candidates[i] for i in selected_idx]
            selected_true_energies = np.array([
                compute_potential_energy(R, molecule) for R in selected_configs
            ])

            # Check numerical stability
            if np.any(np.isnan(selected_true_energies)) or np.any(np.isinf(selected_true_energies)):
                return RunResult(
                    run_id=run_id, seed=seed, rho_bar=np.array([]),
                    total_charge=np.nan, expected_charge=molecule.n_electrons,
                    ess=0, ess_ratio=0, n_samples=len(all_samples),
                    surrogate_rmse=np.nan, surrogate_rmse_kT=np.nan,
                    weight_max=np.nan, weight_min=np.nan, weight_std=np.nan,
                    runtime_seconds=time.time() - start_time,
                    converged=False, error_message=f"NaN/Inf at iteration {iteration+1}"
                )

            surrogate.update(selected_configs, selected_true_energies)

            all_samples.extend(selected_configs)
            all_true_energies.extend(selected_true_energies)
            selected_surrogate_energies = surrogate.predict_batch(selected_configs)
            all_surrogate_energies.extend(selected_surrogate_energies)

        # Phase 4: Importance weights
        all_true_energies = np.array(all_true_energies)
        all_surrogate_energies = np.array(all_surrogate_energies)

        weights, ess = compute_importance_weights(
            all_samples, all_true_energies, all_surrogate_energies, beta
        )
        ess_ratio = compute_ess_ratio(weights)

        # Phase 5: Compute density
        grid_points, dV, grid_shape = create_density_grid(R_eq, n_grid_points, grid_padding)
        rho_bar = np.zeros(len(grid_points))

        for R, w in zip(all_samples, weights):
            rho = compute_charge_density(R, grid_points, molecule)
            rho_bar += w * rho

        total_charge = np.sum(rho_bar) * dV

        # Compute surrogate RMSE on last batch
        val_pred = surrogate.predict_batch(all_samples[-K_select:])
        val_rmse = np.sqrt(np.mean((val_pred - all_true_energies[-K_select:])**2))

        runtime = time.time() - start_time

        if verbose:
            print(f"  Run {run_id}: Completed in {runtime:.1f}s, Q={total_charge:.4f}, ESS={ess_ratio*100:.1f}%")

        return RunResult(
            run_id=run_id,
            seed=seed,
            rho_bar=rho_bar,
            total_charge=total_charge,
            expected_charge=molecule.n_electrons,
            ess=ess,
            ess_ratio=ess_ratio,
            n_samples=len(all_samples),
            surrogate_rmse=val_rmse,
            surrogate_rmse_kT=val_rmse / kT,
            weight_max=weights.max(),
            weight_min=weights.min(),
            weight_std=weights.std(),
            runtime_seconds=runtime,
            converged=True
        )

    except Exception as e:
        return RunResult(
            run_id=run_id, seed=seed, rho_bar=np.array([]),
            total_charge=np.nan, expected_charge=molecule.n_electrons,
            ess=0, ess_ratio=0, n_samples=0,
            surrogate_rmse=np.nan, surrogate_rmse_kT=np.nan,
            weight_max=np.nan, weight_min=np.nan, weight_std=np.nan,
            runtime_seconds=time.time() - start_time,
            converged=False, error_message=str(e)
        )


def compute_density_at_nuclei(rho_bar: np.ndarray, grid_points: np.ndarray,
                               nuclear_positions: np.ndarray) -> np.ndarray:
    """Interpolate density at nuclear positions."""
    from scipy.interpolate import NearestNDInterpolator
    interp = NearestNDInterpolator(grid_points, rho_bar)
    return interp(nuclear_positions)


def run_reproducibility_test(K: int = 5, base_seed: int = 1000, verbose: bool = True):
    """
    Run the complete reproducibility test.

    Args:
        K: Number of independent runs
        base_seed: Base seed (each run uses base_seed + k)
        verbose: Whether to print progress

    Returns:
        Dictionary containing all results and statistics
    """
    print("=" * 70)
    print("REPRODUCIBILITY AND STATISTICAL RELIABILITY TEST")
    print("=" * 70)
    print(f"\nRunning {K} independent solver executions...")
    print(f"Each run uses independent randomness (seeds {base_seed} to {base_seed + K - 1})")
    print()

    # Run all experiments
    results: List[RunResult] = []

    for k in range(K):
        seed = base_seed + k
        result = run_solver_with_seed(seed, run_id=k+1, verbose=verbose)
        results.append(result)

    # Separate converged and failed runs
    converged_results = [r for r in results if r.converged]
    failed_results = [r for r in results if not r.converged]

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # =========================================================================
    # Per-run metrics table
    # =========================================================================

    print("\n1. PER-RUN METRICS")
    print("-" * 70)
    print(f"{'Run':<5} {'Seed':<8} {'Q (e⁻)':<10} {'ESS %':<10} {'RMSE (kT)':<12} "
          f"{'w_max':<10} {'Time (s)':<10} {'Status':<8}")
    print("-" * 70)

    for r in results:
        status = "OK" if r.converged else "FAILED"
        if r.converged:
            print(f"{r.run_id:<5} {r.seed:<8} {r.total_charge:<10.4f} "
                  f"{r.ess_ratio*100:<10.1f} {r.surrogate_rmse_kT:<12.4f} "
                  f"{r.weight_max:<10.4f} {r.runtime_seconds:<10.1f} {status:<8}")
        else:
            print(f"{r.run_id:<5} {r.seed:<8} {'N/A':<10} {'N/A':<10} {'N/A':<12} "
                  f"{'N/A':<10} {r.runtime_seconds:<10.1f} {status:<8}")
            print(f"      Error: {r.error_message}")

    print("-" * 70)

    if len(converged_results) == 0:
        print("\nERROR: All runs failed. Cannot compute statistics.")
        return {"results": results, "converged": False}

    # =========================================================================
    # Aggregate statistics
    # =========================================================================

    print(f"\n2. AGGREGATE STATISTICS (from {len(converged_results)}/{K} converged runs)")
    print("-" * 70)

    # Total charge statistics
    charges = np.array([r.total_charge for r in converged_results])
    expected_charge = converged_results[0].expected_charge

    Q_mean = np.mean(charges)
    Q_std = np.std(charges, ddof=1) if len(charges) > 1 else 0
    Q_min = np.min(charges)
    Q_max = np.max(charges)

    print(f"\nTotal Charge Q:")
    print(f"  Expected:     {expected_charge} electrons")
    print(f"  Mean ± Std:   {Q_mean:.4f} ± {Q_std:.4f} electrons")
    print(f"  Range:        [{Q_min:.4f}, {Q_max:.4f}]")
    print(f"  Rel. Error:   {abs(Q_mean - expected_charge)/expected_charge * 100:.3f}%")

    # ESS statistics
    ess_ratios = np.array([r.ess_ratio for r in converged_results])
    print(f"\nEffective Sample Size (ESS ratio):")
    print(f"  Mean ± Std:   {np.mean(ess_ratios)*100:.1f}% ± {np.std(ess_ratios, ddof=1)*100:.1f}%")
    print(f"  Range:        [{np.min(ess_ratios)*100:.1f}%, {np.max(ess_ratios)*100:.1f}%]")

    # Surrogate RMSE statistics
    rmses = np.array([r.surrogate_rmse_kT for r in converged_results])
    print(f"\nSurrogate RMSE (in k_B T units):")
    print(f"  Mean ± Std:   {np.mean(rmses):.4f} ± {np.std(rmses, ddof=1):.4f}")
    print(f"  Range:        [{np.min(rmses):.4f}, {np.max(rmses):.4f}]")

    # Weight statistics
    w_maxs = np.array([r.weight_max for r in converged_results])
    w_stds = np.array([r.weight_std for r in converged_results])
    print(f"\nImportance Weight Statistics:")
    print(f"  Max weight:   {np.mean(w_maxs):.4f} ± {np.std(w_maxs, ddof=1):.4f}")
    print(f"  Weight std:   {np.mean(w_stds):.6f} ± {np.std(w_stds, ddof=1):.6f}")

    # Runtime
    runtimes = np.array([r.runtime_seconds for r in converged_results])
    print(f"\nRuntime:")
    print(f"  Mean ± Std:   {np.mean(runtimes):.1f} ± {np.std(runtimes, ddof=1):.1f} seconds")

    # =========================================================================
    # Density field statistics
    # =========================================================================

    print(f"\n3. DENSITY FIELD STATISTICS")
    print("-" * 70)

    # Stack all density fields
    rho_bars = np.array([r.rho_bar for r in converged_results])

    # Mean density
    rho_mean = np.mean(rho_bars, axis=0)

    # Pointwise variance and std
    rho_var = np.var(rho_bars, axis=0, ddof=1) if len(rho_bars) > 1 else np.zeros_like(rho_mean)
    rho_std = np.sqrt(rho_var)

    # Coefficient of variation (relative variability)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rho_cv = np.where(rho_mean > 1e-10, rho_std / rho_mean, 0)

    print(f"\nPointwise Density Statistics:")
    print(f"  Mean density range:     [{rho_mean.min():.6f}, {rho_mean.max():.6f}]")
    print(f"  Std dev range:          [{rho_std.min():.6f}, {rho_std.max():.6f}]")
    print(f"  Coef. of variation:     max = {rho_cv.max():.4f} (at low-density regions)")
    print(f"  Mean abs. std:          {np.mean(rho_std):.6f}")

    # Relative variability in high-density regions
    high_density_mask = rho_mean > 0.1 * rho_mean.max()
    if np.any(high_density_mask):
        cv_high = rho_cv[high_density_mask]
        print(f"  CV in high-ρ region:    mean = {np.mean(cv_high):.4f}, max = {np.max(cv_high):.4f}")

    # =========================================================================
    # Density at nuclear positions
    # =========================================================================

    molecule = WaterMolecule()
    R_eq = molecule.get_equilibrium_geometry()
    grid_points, dV, _ = create_density_grid(R_eq, 15, 2.5)

    print(f"\nDensity at Nuclear Positions:")

    nuclear_densities = []
    for r in converged_results:
        rho_nuc = compute_density_at_nuclei(r.rho_bar, grid_points, R_eq)
        nuclear_densities.append(rho_nuc)

    nuclear_densities = np.array(nuclear_densities)

    for i, atom in enumerate(molecule.atom_types):
        rho_at_nuc = nuclear_densities[:, i]
        print(f"  {atom} (atom {i+1}):  {np.mean(rho_at_nuc):.4f} ± {np.std(rho_at_nuc, ddof=1):.4f}")

    # =========================================================================
    # Consistency checks
    # =========================================================================

    print(f"\n4. CONSISTENCY CHECKS")
    print("-" * 70)

    checks_passed = 0
    checks_total = 4

    # Check 1: Charge conservation
    charge_error = abs(Q_mean - expected_charge) / expected_charge
    check1_pass = charge_error < 0.01  # < 1% error
    checks_passed += check1_pass
    print(f"\n✓ Charge Conservation:" if check1_pass else f"\n✗ Charge Conservation:")
    print(f"  Mean charge {Q_mean:.4f} vs expected {expected_charge}")
    print(f"  Relative error: {charge_error*100:.3f}% {'< 1% ✓' if check1_pass else '>= 1% ✗'}")

    # Check 2: ESS not collapsed
    min_ess = np.min(ess_ratios)
    check2_pass = min_ess > 0.1  # ESS > 10%
    checks_passed += check2_pass
    print(f"\n✓ ESS Stability:" if check2_pass else f"\n✗ ESS Stability:")
    print(f"  Minimum ESS ratio: {min_ess*100:.1f}%")
    print(f"  {'All runs have ESS > 10% ✓' if check2_pass else 'Some runs have ESS < 10% ✗'}")

    # Check 3: Low between-run variability
    # Compare between-run std to typical MC uncertainty (approximated)
    relative_between_run_var = Q_std / Q_mean if Q_mean > 0 else float('inf')
    check3_pass = relative_between_run_var < 0.01  # < 1% coefficient of variation
    checks_passed += check3_pass
    print(f"\n✓ Between-Run Variability:" if check3_pass else f"\n✗ Between-Run Variability:")
    print(f"  Charge CV: {relative_between_run_var*100:.4f}%")
    print(f"  {'Low variability < 1% ✓' if check3_pass else 'High variability >= 1% ✗'}")

    # Check 4: No pathological runs
    n_pathological = len(failed_results)
    # Also check for extreme ESS or weights in converged runs
    n_extreme = sum(1 for r in converged_results if r.ess_ratio < 0.5 or r.weight_max > 0.5)
    check4_pass = n_pathological == 0 and n_extreme == 0
    checks_passed += check4_pass
    print(f"\n✓ No Pathological Behavior:" if check4_pass else f"\n✗ Pathological Behavior Detected:")
    print(f"  Failed runs: {n_pathological}")
    print(f"  Runs with extreme ESS/weights: {n_extreme}")
    print(f"  {'All runs healthy ✓' if check4_pass else 'Some runs show issues ✗'}")

    # =========================================================================
    # Overall assessment
    # =========================================================================

    print(f"\n" + "=" * 70)
    print("5. OVERALL ASSESSMENT")
    print("=" * 70)

    print(f"\nChecks passed: {checks_passed}/{checks_total}")

    # Qualitative assessment
    if checks_passed == checks_total and len(failed_results) == 0:
        stability = "EXCELLENT"
        assessment = (
            "The estimator demonstrates excellent stability and reproducibility.\n"
            "  • All runs converged successfully\n"
            "  • Charge conservation is satisfied within numerical precision\n"
            "  • Between-run variability is negligible\n"
            "  • No pathological behavior detected\n"
            "  • Results are statistically reliable and consistent"
        )
    elif checks_passed >= 3 and len(failed_results) <= 1:
        stability = "GOOD"
        assessment = (
            "The estimator demonstrates good stability with minor issues.\n"
            "  • Most runs converged successfully\n"
            "  • Charge conservation is approximately satisfied\n"
            "  • Between-run variability is acceptable\n"
            "  • Results are generally reliable"
        )
    elif checks_passed >= 2:
        stability = "MODERATE"
        assessment = (
            "The estimator shows moderate stability with some concerns.\n"
            "  • Some runs may have failed or shown pathological behavior\n"
            "  • Between-run variability may be significant\n"
            "  • Results should be interpreted with caution\n"
            "  • Consider increasing sample size or adjusting parameters"
        )
    else:
        stability = "POOR"
        assessment = (
            "The estimator shows poor stability and reliability.\n"
            "  • Multiple runs failed or showed pathological behavior\n"
            "  • High between-run variability\n"
            "  • Results are not reliable\n"
            "  • Algorithm parameters or strategy should be reconsidered"
        )

    print(f"\nStability Rating: {stability}")
    print(f"\nAssessment:\n  {assessment}")

    # =========================================================================
    # Return comprehensive results
    # =========================================================================

    return {
        "results": results,
        "converged_count": len(converged_results),
        "failed_count": len(failed_results),
        "rho_mean": rho_mean,
        "rho_std": rho_std,
        "rho_var": rho_var,
        "grid_points": grid_points,
        "dV": dV,
        "statistics": {
            "total_charge": {"mean": Q_mean, "std": Q_std, "min": Q_min, "max": Q_max},
            "ess_ratio": {"mean": np.mean(ess_ratios), "std": np.std(ess_ratios, ddof=1),
                         "min": np.min(ess_ratios), "max": np.max(ess_ratios)},
            "surrogate_rmse_kT": {"mean": np.mean(rmses), "std": np.std(rmses, ddof=1)},
            "runtime": {"mean": np.mean(runtimes), "std": np.std(runtimes, ddof=1)}
        },
        "checks": {
            "charge_conservation": check1_pass,
            "ess_stability": check2_pass,
            "low_variability": check3_pass,
            "no_pathology": check4_pass,
            "passed": checks_passed,
            "total": checks_total
        },
        "stability_rating": stability,
        "converged": True
    }


if __name__ == "__main__":
    # Run reproducibility test with K=7 independent runs
    K = 7

    print(f"\nStarting reproducibility test with K={K} runs...\n")

    test_results = run_reproducibility_test(K=K, base_seed=2024, verbose=True)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
