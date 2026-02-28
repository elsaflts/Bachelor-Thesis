"""
Level 5 Boltzmann Distribution Sampling
10D Bimodal Molecular Conformational Landscape

Strategy: Parallel Tempering (Replica Exchange MCMC)
Potential: V(x) = min(||x-a||^2, ||x-b||^2 + Delta)
Distribution: pi(x) proportional to exp(-V(x)/T)
Goal: Estimate E_pi[x]
"""

import numpy as np
from typing import Dict, List, Tuple, Any


def solver() -> Dict[str, Any]:
    """
    MCMC solver using Parallel Tempering to estimate E_pi[x] for a 10D bimodal
    Boltzmann distribution.

    Returns:
        Dictionary containing:
        - per_chain_estimates: E[x] estimate from each chain
        - combined_estimate: Pooled E[x] estimate
        - diagnostics: Acceptance rates, basin occupancy, mixing info
    """

    # =========================================================================
    # PROBLEM PARAMETERS
    # =========================================================================
    DIM = 10
    BOUNDS = (0.0, 50.0)
    T = 1.0  # Temperature
    DELTA = 1.0  # Basin energy offset

    # Basin centers
    a = np.array([15, 35, 15, 35, 15, 35, 15, 35, 15, 35], dtype=np.float64)
    b = np.array([40, 10, 40, 10, 40, 10, 40, 10, 40, 10], dtype=np.float64)

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================

    def potential(x: np.ndarray) -> float:
        """Compute V(x) = min(||x-a||^2, ||x-b||^2 + Delta)"""
        dist_a_sq = np.sum((x - a) ** 2)
        dist_b_sq = np.sum((x - b) ** 2) + DELTA
        return min(dist_a_sq, dist_b_sq)

    def get_basin(x: np.ndarray) -> str:
        """Determine which basin x belongs to based on lower energy"""
        dist_a_sq = np.sum((x - a) ** 2)
        dist_b_sq = np.sum((x - b) ** 2) + DELTA
        return 'A' if dist_a_sq <= dist_b_sq else 'B'

    def reflect_bounds(x: np.ndarray) -> np.ndarray:
        """Reflect coordinates back into [0, 50] using reflection boundary"""
        x_new = x.copy()
        for i in range(DIM):
            # Reflect until within bounds
            while x_new[i] < BOUNDS[0] or x_new[i] > BOUNDS[1]:
                if x_new[i] < BOUNDS[0]:
                    x_new[i] = 2 * BOUNDS[0] - x_new[i]
                elif x_new[i] > BOUNDS[1]:
                    x_new[i] = 2 * BOUNDS[1] - x_new[i]
        return x_new

    # =========================================================================
    # PARALLEL TEMPERING MCMC
    # =========================================================================

    def run_parallel_tempering(
        init_x: np.ndarray,
        n_samples: int,
        burn_in: int,
        n_temps: int,
        temp_max: float,
        base_step_size: float,
        swap_interval: int,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, float, List[str], int, Dict]:
        """
        Run Parallel Tempering MCMC chain with adaptive step sizes.

        Args:
            init_x: Initial position
            n_samples: Number of post-burn-in samples
            burn_in: Number of burn-in steps
            n_temps: Number of temperature replicas
            temp_max: Maximum temperature in ladder
            base_step_size: Base proposal step size at T=1
            swap_interval: Steps between swap attempts
            rng: Random number generator

        Returns:
            samples: Array of samples from target temperature
            acceptance_rate: Acceptance rate at target temperature
            basin_history: Basin identity for each sample
            n_swaps_accepted: Number of accepted replica swaps
            extra_info: Additional diagnostics
        """
        # Temperature ladder (geometric spacing)
        # High enough max temp to allow free diffusion across barrier
        temps = T * np.geomspace(1.0, temp_max, n_temps)
        betas = 1.0 / temps  # Inverse temperatures

        # Adaptive step sizes per replica (tuned during burn-in)
        step_sizes = np.array([base_step_size * np.sqrt(temps[i] / T) for i in range(n_temps)])

        # Initialize all replicas at same position
        replicas = [init_x.copy() for _ in range(n_temps)]
        energies = [potential(r) for r in replicas]

        # Storage
        samples = []
        basin_history = []

        # Counters
        accepts = [0] * n_temps
        attempts = [0] * n_temps
        swap_accepts = [0] * (n_temps - 1)
        swap_attempts = [0] * (n_temps - 1)

        # Adaptation windows
        adapt_window = 500
        target_accept = 0.234  # Optimal for high-dim Gaussian

        total_steps = n_samples + burn_in

        for step in range(total_steps):
            # -----------------------------------------------------------------
            # Local Metropolis-Hastings moves for each replica
            # -----------------------------------------------------------------
            for i in range(n_temps):
                # Propose new position with temperature-scaled step
                proposal = replicas[i] + rng.normal(0, step_sizes[i], DIM)
                proposal = reflect_bounds(proposal)

                # Compute acceptance probability
                energy_new = potential(proposal)
                log_alpha = -betas[i] * (energy_new - energies[i])

                attempts[i] += 1
                if np.log(rng.random()) < log_alpha:
                    replicas[i] = proposal
                    energies[i] = energy_new
                    accepts[i] += 1

            # -----------------------------------------------------------------
            # Adapt step sizes during burn-in
            # -----------------------------------------------------------------
            if step < burn_in and step > 0 and step % adapt_window == 0:
                for i in range(n_temps):
                    recent_accept = accepts[i] / attempts[i] if attempts[i] > 0 else 0.5
                    # Adjust step size to approach target acceptance
                    if recent_accept < target_accept - 0.05:
                        step_sizes[i] *= 0.9
                    elif recent_accept > target_accept + 0.05:
                        step_sizes[i] *= 1.1
                    # Clamp step sizes
                    step_sizes[i] = np.clip(step_sizes[i], 0.01, 20.0 * np.sqrt(temps[i]))

            # -----------------------------------------------------------------
            # Replica exchange (swap) moves - try multiple times per interval
            # -----------------------------------------------------------------
            if step % swap_interval == 0 and step > 0:
                # Randomly choose swap direction to ensure detailed balance
                if rng.random() < 0.5:
                    pairs = range(n_temps - 1)
                else:
                    pairs = range(n_temps - 2, -1, -1)

                for i in pairs:
                    j = i + 1

                    # Metropolis criterion for swap
                    # log(accept) = (beta_j - beta_i) * (E_i - E_j)
                    delta_beta = betas[j] - betas[i]
                    delta_energy = energies[i] - energies[j]
                    log_swap_prob = delta_beta * delta_energy

                    swap_attempts[i] += 1
                    if np.log(rng.random()) < log_swap_prob:
                        # Swap replicas (configurations, not temperatures)
                        replicas[i], replicas[j] = replicas[j], replicas[i]
                        energies[i], energies[j] = energies[j], energies[i]
                        swap_accepts[i] += 1

            # -----------------------------------------------------------------
            # Store samples from target temperature (replica 0)
            # -----------------------------------------------------------------
            if step >= burn_in:
                samples.append(replicas[0].copy())
                basin_history.append(get_basin(replicas[0]))

        samples = np.array(samples)
        acceptance_rate = accepts[0] / attempts[0] if attempts[0] > 0 else 0.0

        # Compute swap acceptance rates
        swap_rates = [swap_accepts[i] / swap_attempts[i] if swap_attempts[i] > 0 else 0
                      for i in range(n_temps - 1)]

        extra_info = {
            'final_step_sizes': step_sizes.tolist(),
            'acceptance_rates_all_temps': [accepts[i]/attempts[i] if attempts[i] > 0 else 0
                                           for i in range(n_temps)],
            'swap_acceptance_rates': swap_rates,
            'temperatures': temps.tolist()
        }

        return samples, acceptance_rate, basin_history, sum(swap_accepts), extra_info

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    try:
        # Sampling parameters
        # The barrier height is ~||a-b||^2/4 ≈ 1560, so T_max must be high enough
        # to allow free diffusion at the hottest replica
        N_SAMPLES = 80000      # Post-burn-in samples per chain
        BURN_IN = 30000        # Burn-in period (longer for adaptation)
        N_TEMPS = 20           # Number of temperature levels (more for large barrier)
        TEMP_MAX = 800.0       # Maximum temperature (barrier/T_max ~ 2, allows crossing)
        STEP_SIZE = 0.5        # Base proposal step size (will be adapted)
        SWAP_INTERVAL = 5      # Steps between swap attempts (more frequent)
        N_CHAINS = 4           # Number of independent chains

        # Initialize chains: 2 near basin A, 2 near basin B
        rng_init = np.random.default_rng(42)
        init_positions = [
            a + rng_init.normal(0, 1.0, DIM),  # Chain 0: near A
            a + rng_init.normal(0, 1.0, DIM),  # Chain 1: near A
            b + rng_init.normal(0, 1.0, DIM),  # Chain 2: near B
            b + rng_init.normal(0, 1.0, DIM),  # Chain 3: near B
        ]
        # Ensure within bounds
        init_positions = [reflect_bounds(p) for p in init_positions]
        init_basins = [get_basin(p) for p in init_positions]

        # Run all chains
        chain_results = []
        for chain_id in range(N_CHAINS):
            print(f"Running chain {chain_id} (init near basin {init_basins[chain_id]})...", flush=True)
            rng = np.random.default_rng(42 + chain_id * 1000)

            samples, acc_rate, basin_hist, n_swaps, extra_info = run_parallel_tempering(
                init_x=init_positions[chain_id],
                n_samples=N_SAMPLES,
                burn_in=BURN_IN,
                n_temps=N_TEMPS,
                temp_max=TEMP_MAX,
                base_step_size=STEP_SIZE,
                swap_interval=SWAP_INTERVAL,
                rng=rng
            )

            # Compute per-chain statistics
            mean_estimate = np.mean(samples, axis=0)
            basin_A_frac = sum(1 for b in basin_hist if b == 'A') / len(basin_hist)
            basin_B_frac = 1.0 - basin_A_frac

            # Count basin transitions
            transitions = sum(1 for i in range(1, len(basin_hist))
                            if basin_hist[i] != basin_hist[i-1])

            chain_results.append({
                'chain_id': chain_id,
                'init_basin': init_basins[chain_id],
                'samples': samples,
                'acceptance_rate': acc_rate,
                'basin_A_fraction': basin_A_frac,
                'basin_B_fraction': basin_B_frac,
                'transitions': transitions,
                'mean_estimate': mean_estimate,
                'extra_info': extra_info
            })
            print(f"  Done. Accept rate: {acc_rate:.4f}, Transitions: {transitions}", flush=True)

        # =====================================================================
        # COMPUTE COMBINED ESTIMATES
        # =====================================================================

        all_samples = np.vstack([r['samples'] for r in chain_results])
        combined_mean = np.mean(all_samples, axis=0)
        combined_std_error = np.std(all_samples, axis=0) / np.sqrt(len(all_samples))

        # Overall basin occupancy
        total_A = sum(r['basin_A_fraction'] * len(r['samples']) for r in chain_results)
        total_samples = sum(len(r['samples']) for r in chain_results)
        overall_basin_A = total_A / total_samples
        overall_basin_B = 1.0 - overall_basin_A

        # Mixing assessment
        total_transitions = sum(r['transitions'] for r in chain_results)
        chains_visiting_both = sum(
            1 for r in chain_results
            if r['basin_A_fraction'] > 0.01 and r['basin_B_fraction'] > 0.01
        )

        # Theoretical basin probabilities (approximate)
        V_a = potential(a)  # = 0
        V_b = potential(b)  # = Delta = 1
        w_a = np.exp(-V_a / T)
        w_b = np.exp(-V_b / T)
        theoretical_p_A = w_a / (w_a + w_b)
        theoretical_p_B = w_b / (w_a + w_b)

        # =====================================================================
        # BUILD OUTPUT
        # =====================================================================

        # Per-chain estimates
        per_chain_estimates = {}
        for r in chain_results:
            per_chain_estimates[f"chain_{r['chain_id']}"] = {
                'E_x': r['mean_estimate'].tolist(),
                'init_basin': r['init_basin'],
                'acceptance_rate': round(r['acceptance_rate'], 4),
                'basin_A_fraction': round(r['basin_A_fraction'], 4),
                'basin_B_fraction': round(r['basin_B_fraction'], 4),
                'transitions': r['transitions']
            }

        # Diagnostics summary
        diagnostics = {
            'sampling_parameters': {
                'method': 'Parallel Tempering (Replica Exchange MCMC)',
                'n_chains': N_CHAINS,
                'n_temperature_replicas': N_TEMPS,
                'temperature_range': f'[{T}, {T * TEMP_MAX}]',
                'step_size': STEP_SIZE,
                'swap_interval': SWAP_INTERVAL,
                'burn_in': BURN_IN,
                'retained_samples_per_chain': N_SAMPLES,
                'total_retained_samples': total_samples
            },
            'acceptance_rates': {
                f'chain_{r["chain_id"]}': round(r['acceptance_rate'], 4)
                for r in chain_results
            },
            'basin_occupancy': {
                'per_chain': {
                    f'chain_{r["chain_id"]}': {
                        'A': round(r['basin_A_fraction'], 4),
                        'B': round(r['basin_B_fraction'], 4)
                    } for r in chain_results
                },
                'overall': {
                    'A': round(overall_basin_A, 4),
                    'B': round(overall_basin_B, 4)
                },
                'theoretical_approximate': {
                    'A': round(theoretical_p_A, 4),
                    'B': round(theoretical_p_B, 4)
                }
            },
            'mixing_assessment': {
                'total_basin_transitions': total_transitions,
                'chains_visiting_both_basins': f'{chains_visiting_both}/{N_CHAINS}',
                'status': 'GOOD' if chains_visiting_both >= 3 else
                         ('MODERATE' if chains_visiting_both >= 1 else 'POOR')
            }
        }

        result = {
            'combined_estimate': {
                'E_pi_x': combined_mean.tolist(),
                'std_error': combined_std_error.tolist()
            },
            'per_chain_estimates': per_chain_estimates,
            'diagnostics': diagnostics
        }

        # =====================================================================
        # PRINT RESULTS
        # =====================================================================

        print("=" * 75)
        print("MCMC SAMPLING: 10D BIMODAL BOLTZMANN DISTRIBUTION")
        print("Strategy: Parallel Tempering (Replica Exchange MCMC)")
        print("=" * 75)

        print(f"\nProblem Parameters:")
        print(f"  Dimension: {DIM}")
        print(f"  Temperature T: {T}")
        print(f"  Basin offset Delta: {DELTA}")
        print(f"  Bounds: [{BOUNDS[0]}, {BOUNDS[1]}]^{DIM}")
        print(f"  Basin A center: {a.tolist()}")
        print(f"  Basin B center: {b.tolist()}")
        print(f"  Inter-basin distance ||a-b||: {np.linalg.norm(a - b):.2f}")

        print(f"\nSampling Parameters:")
        print(f"  Number of chains: {N_CHAINS}")
        print(f"  Temperature replicas: {N_TEMPS}")
        print(f"  Temperature range: [{T}, {T * TEMP_MAX}]")
        print(f"  Step size: {STEP_SIZE}")
        print(f"  Burn-in: {BURN_IN}")
        print(f"  Retained samples per chain: {N_SAMPLES}")
        print(f"  Total retained samples: {total_samples}")

        print("\n" + "=" * 75)
        print("PER-CHAIN DIAGNOSTICS")
        print("=" * 75)
        print(f"\n{'Chain':<8}{'Init':<8}{'Accept':<12}{'Basin A':<12}{'Basin B':<12}{'Transitions':<12}")
        print("-" * 64)
        for r in chain_results:
            print(f"{r['chain_id']:<8}{r['init_basin']:<8}{r['acceptance_rate']:<12.4f}"
                  f"{r['basin_A_fraction']:<12.4f}{r['basin_B_fraction']:<12.4f}"
                  f"{r['transitions']:<12}")
        print("-" * 64)
        print(f"{'Overall':<8}{'':<8}{'':<12}{overall_basin_A:<12.4f}{overall_basin_B:<12.4f}"
              f"{total_transitions:<12}")

        print("\n" + "=" * 75)
        print("MIXING ASSESSMENT")
        print("=" * 75)
        print(f"\nTotal basin transitions: {total_transitions}")
        print(f"Chains visiting both basins (>1% each): {chains_visiting_both}/{N_CHAINS}")
        print(f"Mixing status: {diagnostics['mixing_assessment']['status']}")

        print("\n" + "=" * 75)
        print("PER-CHAIN ESTIMATES OF E_pi[x]")
        print("=" * 75)
        for r in chain_results:
            est = r['mean_estimate']
            print(f"\nChain {r['chain_id']} (init: {r['init_basin']}):")
            print(f"  [{', '.join([f'{v:.4f}' for v in est])}]")

        print("\n" + "=" * 75)
        print("COMBINED ESTIMATE OF E_pi[x]")
        print("=" * 75)
        print(f"\nE_pi[x] = [{', '.join([f'{v:.4f}' for v in combined_mean])}]")
        print(f"\nStd Err = [{', '.join([f'{v:.5f}' for v in combined_std_error])}]")

        print("\n" + "=" * 75)
        print("THEORETICAL COMPARISON")
        print("=" * 75)
        print(f"\nEnergy at basin centers: V(a) = {V_a:.1f}, V(b) = {V_b:.1f}")
        print(f"Approximate basin probabilities: P(A) ≈ {theoretical_p_A:.4f}, P(B) ≈ {theoretical_p_B:.4f}")
        simple_expected = theoretical_p_A * a + theoretical_p_B * b
        print(f"Simple basin-weighted estimate:")
        print(f"  [{', '.join([f'{v:.4f}' for v in simple_expected])}]")

        print("\n" + "=" * 75)
        print("FINAL ANSWER")
        print("=" * 75)
        print("\nE_pi[x] components:")
        for i in range(DIM):
            print(f"  E[x_{i+1}] = {combined_mean[i]:.4f} ± {combined_std_error[i]:.5f}")

        return result

    except Exception as e:
        error_result = {
            'error': str(e),
            'combined_estimate': None,
            'per_chain_estimates': None,
            'diagnostics': None
        }
        print(f"ERROR: {str(e)}")
        return error_result


# Execute solver when run directly
if __name__ == "__main__":
    result = solver()
