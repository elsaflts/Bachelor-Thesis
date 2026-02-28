import numpy as np
from scipy.special import logsumexp

def solver():
    """
    Monte Carlo sampling for bimodal energy landscape using emcee ensemble sampler.
    Computes equilibrium expected values E_π[x] and compares with grid-based reference.

    Strategy: emcee Ensemble MCMC Sampler (Affine-Invariant)
    """
    try:
        import emcee

        # Set random seed for reproducibility
        np.random.seed(42)

        # =====================================================================
        # PARAMETERS
        # =====================================================================
        T = 3.0  # Temperature
        DELTA = 2.0  # Energy offset for second basin
        DOMAIN_MIN = 0.0
        DOMAIN_MAX = 50.0

        # Basin centers
        BASIN1_CENTER = np.array([15.0, 35.0])  # Deeper basin (no offset)
        BASIN2_CENTER = np.array([40.0, 10.0])  # Shallower basin (offset +Delta)

        # Dimension of the problem
        ndim = 2

        # =====================================================================
        # POTENTIAL AND LOG-PROBABILITY FUNCTIONS
        # =====================================================================
        def potential(x):
            """
            Compute the potential energy V(x1, x2).
            V = min((x1-15)^2 + (x2-35)^2, (x1-40)^2 + (x2-10)^2 + Delta)
            """
            x = np.asarray(x)
            basin1 = (x[..., 0] - 15)**2 + (x[..., 1] - 35)**2
            basin2 = (x[..., 0] - 40)**2 + (x[..., 1] - 10)**2 + DELTA
            return np.minimum(basin1, basin2)

        def log_prob(x):
            """
            Log probability function for emcee sampler.
            Returns -inf for out-of-bounds proposals (hard boundary enforcement).
            """
            # Boundary check - reject if outside domain
            if x[0] < DOMAIN_MIN or x[0] > DOMAIN_MAX:
                return -np.inf
            if x[1] < DOMAIN_MIN or x[1] > DOMAIN_MAX:
                return -np.inf
            # Return log of unnormalized Boltzmann probability
            return -potential(x) / T

        # =====================================================================
        # GRID-BASED REFERENCE COMPUTATION
        # =====================================================================
        def compute_grid_reference(grid_step=0.25):
            """
            Compute reference E[x] using numerical integration on fine grid.
            Uses log-sum-exp for numerical stability.
            """
            # Create grid over domain
            x1_vals = np.arange(DOMAIN_MIN, DOMAIN_MAX + grid_step, grid_step)
            x2_vals = np.arange(DOMAIN_MIN, DOMAIN_MAX + grid_step, grid_step)
            X1, X2 = np.meshgrid(x1_vals, x2_vals, indexing='ij')

            # Stack for vectorized computation
            grid_points = np.stack([X1, X2], axis=-1)

            # Compute log probabilities on grid
            log_probs = -potential(grid_points) / T

            # Flatten arrays
            log_probs_flat = log_probs.flatten()
            x1_flat = X1.flatten()
            x2_flat = X2.flatten()

            # Compute log partition function using logsumexp
            log_Z = logsumexp(log_probs_flat)

            # Compute E[x1] using log-sum-exp (handle x1=0 case)
            mask_x1 = x1_flat > 0
            log_x1_weighted = np.full_like(log_probs_flat, -np.inf)
            log_x1_weighted[mask_x1] = log_probs_flat[mask_x1] + np.log(x1_flat[mask_x1])
            E_x1 = np.exp(logsumexp(log_x1_weighted) - log_Z)

            # Compute E[x2] using log-sum-exp (handle x2=0 case)
            mask_x2 = x2_flat > 0
            log_x2_weighted = np.full_like(log_probs_flat, -np.inf)
            log_x2_weighted[mask_x2] = log_probs_flat[mask_x2] + np.log(x2_flat[mask_x2])
            E_x2 = np.exp(logsumexp(log_x2_weighted) - log_Z)

            n_grid_points = len(x1_vals) * len(x2_vals)
            return np.array([E_x1, E_x2]), log_Z, n_grid_points

        # Compute grid-based reference
        print("Computing grid-based reference...")
        reference_E, log_Z, n_grid = compute_grid_reference(grid_step=0.25)

        # =====================================================================
        # EMCEE ENSEMBLE SAMPLER SETUP
        # =====================================================================
        n_walkers = 32  # Number of walkers (ensemble size, must be >= 2*ndim)
        n_steps = 12000  # Total MCMC steps per walker
        n_burnin = 2000  # Burn-in steps to discard

        # Initialize walkers - half near each basin to ensure mode exploration
        initial_positions = np.zeros((n_walkers, ndim))

        # First half (chain group 1): Initialize near basin 1 (15, 35)
        for i in range(n_walkers // 2):
            initial_positions[i] = BASIN1_CENTER + np.random.normal(0, 1.5, size=2)
            # Clip to ensure within bounds
            initial_positions[i] = np.clip(initial_positions[i], DOMAIN_MIN + 0.5, DOMAIN_MAX - 0.5)

        # Second half (chain group 2): Initialize near basin 2 (40, 10)
        for i in range(n_walkers // 2, n_walkers):
            initial_positions[i] = BASIN2_CENTER + np.random.normal(0, 1.5, size=2)
            initial_positions[i] = np.clip(initial_positions[i], DOMAIN_MIN + 0.5, DOMAIN_MAX - 0.5)

        # Create the emcee ensemble sampler
        print("Running emcee ensemble sampler...")
        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)

        # Run MCMC
        sampler.run_mcmc(initial_positions, n_steps, progress=False)

        # =====================================================================
        # EXTRACT SAMPLES AND COMPUTE ESTIMATES
        # =====================================================================

        # Get all samples after burn-in (flattened across walkers)
        all_samples = sampler.get_chain(discard=n_burnin, flat=True)

        # Get samples by walker group (chain 1 vs chain 2)
        chain_samples = sampler.get_chain(discard=n_burnin)  # shape: (n_steps-burnin, n_walkers, ndim)

        # Chain 1: walkers initialized near basin 1
        chain1_indices = list(range(n_walkers // 2))
        chain1_samples = chain_samples[:, chain1_indices, :].reshape(-1, 2)
        chain1_mean = np.mean(chain1_samples, axis=0)

        # Chain 2: walkers initialized near basin 2
        chain2_indices = list(range(n_walkers // 2, n_walkers))
        chain2_samples = chain_samples[:, chain2_indices, :].reshape(-1, 2)
        chain2_mean = np.mean(chain2_samples, axis=0)

        # Combined estimate with proper Boltzmann weighting
        # Since chains don't mix well, we need to weight by theoretical basin probabilities
        # Basin 1 has V_min = 0, Basin 2 has V_min = DELTA
        # Relative probability: P(basin1)/P(basin2) = exp(-0/T) / exp(-DELTA/T) = exp(DELTA/T)

        # Compute basin probabilities
        boltzmann_ratio = np.exp(-DELTA / T)  # P(basin2) / P(basin1)
        weight_basin1 = 1.0 / (1.0 + boltzmann_ratio)
        weight_basin2 = boltzmann_ratio / (1.0 + boltzmann_ratio)

        # Weighted combination of per-chain means
        combined_mean_weighted = weight_basin1 * chain1_mean + weight_basin2 * chain2_mean

        # Also compute naive (unweighted) combined for comparison
        combined_mean_naive = np.mean(all_samples, axis=0)
        combined_std = np.std(all_samples, axis=0)

        # Use weighted mean as the primary estimate
        combined_mean = combined_mean_weighted

        # =====================================================================
        # DIAGNOSTICS
        # =====================================================================

        # Acceptance rate (averaged across walkers)
        acceptance_rate = np.mean(sampler.acceptance_fraction)

        # Mode exploration analysis
        threshold_dist = 10.0  # Distance threshold for basin membership

        def analyze_mode_exploration(samples):
            """Compute fraction of samples near each basin."""
            dist_to_basin1 = np.sqrt(np.sum((samples - BASIN1_CENTER)**2, axis=1))
            dist_to_basin2 = np.sqrt(np.sum((samples - BASIN2_CENTER)**2, axis=1))

            near_basin1 = dist_to_basin1 < threshold_dist
            near_basin2 = dist_to_basin2 < threshold_dist

            return {
                'basin1_count': int(np.sum(near_basin1)),
                'basin2_count': int(np.sum(near_basin2)),
                'basin1_fraction': float(np.mean(near_basin1)),
                'basin2_fraction': float(np.mean(near_basin2))
            }

        mode_info = analyze_mode_exploration(all_samples)
        both_modes_explored = mode_info['basin1_fraction'] > 0.1 and mode_info['basin2_fraction'] > 0.1

        # Compute absolute errors
        abs_error = np.abs(combined_mean - reference_E)
        abs_error_naive = np.abs(combined_mean_naive - reference_E)
        chain1_error = np.abs(chain1_mean - reference_E)
        chain2_error = np.abs(chain2_mean - reference_E)

        # =====================================================================
        # OUTPUT RESULTS
        # =====================================================================
        print("\n" + "=" * 70)
        print("MONTE CARLO SAMPLING FOR BIMODAL ENERGY LANDSCAPE")
        print("Method: emcee Affine-Invariant Ensemble Sampler")
        print("=" * 70)

        print(f"\nParameters:")
        print(f"  Temperature T = {T}")
        print(f"  Energy offset Delta = {DELTA}")
        print(f"  Domain: [0, 50]^2")
        print(f"  Basin 1 center: ({BASIN1_CENTER[0]}, {BASIN1_CENTER[1]}) - deeper basin (V_min = 0)")
        print(f"  Basin 2 center: ({BASIN2_CENTER[0]}, {BASIN2_CENTER[1]}) - shallower basin (V_min = {DELTA})")

        print("\n" + "=" * 70)
        print("GRID-BASED REFERENCE COMPUTATION")
        print("=" * 70)
        print(f"  Grid step size: 0.25")
        print(f"  Total grid points: {n_grid}")
        print(f"  Log partition function: {log_Z:.4f}")
        print(f"\nReference E_π[x] (grid-based):")
        print(f"  E[x1] = {reference_E[0]:.6f}")
        print(f"  E[x2] = {reference_E[1]:.6f}")

        print("\n" + "=" * 70)
        print("MCMC SAMPLING (emcee Ensemble)")
        print("=" * 70)
        print(f"\nSampler configuration:")
        print(f"  Number of walkers: {n_walkers}")
        print(f"  Total steps per walker: {n_steps}")
        print(f"  Burn-in steps: {n_burnin}")
        print(f"  Retained steps per walker: {n_steps - n_burnin}")

        print(f"\n--- Chain 1 (walkers initialized near basin 1) ---")
        print(f"  Number of walkers: {len(chain1_indices)}")
        print(f"  Samples: {len(chain1_samples)}")
        print(f"  Chain mean: ({chain1_mean[0]:.6f}, {chain1_mean[1]:.6f})")
        print(f"  Error vs reference: ({chain1_error[0]:.6f}, {chain1_error[1]:.6f})")

        print(f"\n--- Chain 2 (walkers initialized near basin 2) ---")
        print(f"  Number of walkers: {len(chain2_indices)}")
        print(f"  Samples: {len(chain2_samples)}")
        print(f"  Chain mean: ({chain2_mean[0]:.6f}, {chain2_mean[1]:.6f})")
        print(f"  Error vs reference: ({chain2_error[0]:.6f}, {chain2_error[1]:.6f})")

        print("\n" + "=" * 70)
        print("COMBINED MCMC ESTIMATE")
        print("=" * 70)
        print(f"\nTotal retained samples: {len(all_samples)}")
        print(f"Average acceptance rate: {acceptance_rate:.4f}")
        print(f"\nBoltzmann weighting (since chains don't mix between modes):")
        print(f"  Weight for basin 1: {weight_basin1:.4f}")
        print(f"  Weight for basin 2: {weight_basin2:.4f}")
        print(f"\nNaive combined MCMC E_π[x] (unweighted 50/50):")
        print(f"  E[x1] = {combined_mean_naive[0]:.6f}")
        print(f"  E[x2] = {combined_mean_naive[1]:.6f}")
        print(f"\nWeighted combined MCMC E_π[x] (Boltzmann-corrected):")
        print(f"  E[x1] = {combined_mean[0]:.6f}")
        print(f"  E[x2] = {combined_mean[1]:.6f}")
        print(f"\nSample standard deviation:")
        print(f"  std[x1] = {combined_std[0]:.6f}")
        print(f"  std[x2] = {combined_std[1]:.6f}")

        print("\n" + "=" * 70)
        print("COMPARISON: MCMC vs REFERENCE")
        print("=" * 70)
        print(f"\nWeighted MCMC (Boltzmann-corrected):")
        print(f"{'Quantity':<15} {'MCMC':<15} {'Reference':<15} {'Abs Error':<15}")
        print("-" * 60)
        print(f"{'E[x1]':<15} {combined_mean[0]:<15.6f} {reference_E[0]:<15.6f} {abs_error[0]:<15.6f}")
        print(f"{'E[x2]':<15} {combined_mean[1]:<15.6f} {reference_E[1]:<15.6f} {abs_error[1]:<15.6f}")
        print(f"\nNaive MCMC (unweighted 50/50):")
        print(f"{'Quantity':<15} {'MCMC':<15} {'Reference':<15} {'Abs Error':<15}")
        print("-" * 60)
        print(f"{'E[x1]':<15} {combined_mean_naive[0]:<15.6f} {reference_E[0]:<15.6f} {abs_error_naive[0]:<15.6f}")
        print(f"{'E[x2]':<15} {combined_mean_naive[1]:<15.6f} {reference_E[1]:<15.6f} {abs_error_naive[1]:<15.6f}")

        print("\n" + "=" * 70)
        print("DIAGNOSTICS SUMMARY")
        print("=" * 70)
        print(f"\n1. Acceptance rate: {acceptance_rate:.4f} (target: 0.2-0.5)")
        print(f"2. Burn-in length: {n_burnin} steps per walker")
        print(f"3. Total retained samples: {len(all_samples)}")
        print(f"4. Mode exploration analysis:")
        print(f"   - Basin 1 (deeper at (15,35)): {mode_info['basin1_fraction']*100:.1f}% of samples ({mode_info['basin1_count']} samples)")
        print(f"   - Basin 2 (shallower at (40,10)): {mode_info['basin2_fraction']*100:.1f}% of samples ({mode_info['basin2_count']} samples)")

        if both_modes_explored:
            print(f"   - BOTH MODES EXPLORED: Yes")
        else:
            print(f"   - WARNING: One mode may be under-sampled!")

        # Theoretical insight
        boltzmann_ratio = np.exp(-DELTA / T)
        expected_basin1_frac = 1 / (1 + boltzmann_ratio)
        print(f"\n5. Theoretical insight:")
        print(f"   Boltzmann weight ratio exp(-Delta/T) = {boltzmann_ratio:.4f}")
        print(f"   Expected basin 1 fraction: ~{expected_basin1_frac*100:.1f}%")

        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"\nMCMC estimate E_pi[x] = ({combined_mean[0]:.6f}, {combined_mean[1]:.6f})")
        print(f"Reference estimate   = ({reference_E[0]:.6f}, {reference_E[1]:.6f})")
        print(f"Absolute error       = ({abs_error[0]:.6f}, {abs_error[1]:.6f})")
        print(f"Total absolute error = {np.sum(abs_error):.6f}")

        # Build results dictionary
        results = {
            "mcmc_estimate_combined": {
                "E_x1": float(combined_mean[0]),
                "E_x2": float(combined_mean[1])
            },
            "mcmc_estimate_chain1": {
                "E_x1": float(chain1_mean[0]),
                "E_x2": float(chain1_mean[1])
            },
            "mcmc_estimate_chain2": {
                "E_x1": float(chain2_mean[0]),
                "E_x2": float(chain2_mean[1])
            },
            "reference_estimate": {
                "E_x1": float(reference_E[0]),
                "E_x2": float(reference_E[1])
            },
            "absolute_error": {
                "E_x1": float(abs_error[0]),
                "E_x2": float(abs_error[1]),
                "total": float(np.sum(abs_error))
            },
            "diagnostics": {
                "acceptance_rate": float(acceptance_rate),
                "burn_in_steps": n_burnin,
                "total_retained_samples": len(all_samples),
                "samples_chain1": len(chain1_samples),
                "samples_chain2": len(chain2_samples),
                "basin1_fraction": float(mode_info['basin1_fraction']),
                "basin2_fraction": float(mode_info['basin2_fraction']),
                "both_modes_explored": both_modes_explored
            },
            "parameters": {
                "temperature": T,
                "delta": DELTA,
                "n_walkers": n_walkers,
                "n_steps": n_steps
            }
        }

        return results

    except ImportError as e:
        error_msg = f"Import error: {str(e)}. Please install required package: pip install emcee"
        print(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Error during computation: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {"error": error_msg}


# Execute the solver when script is run
if __name__ == "__main__":
    result = solver()
