import numpy as np
from typing import Dict, Any, Tuple, List
import warnings

def solver() -> Dict[str, Any]:
    """
    Solver for computing thermally averaged electronic charge density using
    Replica Exchange Langevin Dynamics (Parallel Tempering).

    Strategy 1: RELD with overdamped Langevin dynamics and replica exchange moves.

    OPTIMIZED VERSION: Vectorized charge density computation for performance.
    """

    # ============================================
    # CONFIGURATION PARAMETERS
    # ============================================

    # Molecular system: Diatomic molecule (e.g., H2-like)
    N_atoms = 2  # Number of nuclei
    dim = 3  # Spatial dimensions
    N_dof = N_atoms * dim  # Total degrees of freedom (3N = 6)

    # Physical constants (atomic units)
    k_B = 3.1668114e-6  # Boltzmann constant in Hartree/K

    # Temperature settings
    T_target = 300.0  # Target temperature in Kelvin
    beta_target = 1.0 / (k_B * T_target)

    # Parallel tempering parameters
    # DEBUGGED v2: 6 replicas with very high T_max for optimal exchange rate (20-40%)
    N_replicas = 6  # Moderate number of replicas
    T_max = 10000.0  # Very high max temperature for challenging barrier crossing

    # Langevin dynamics parameters
    gamma = 1.0  # Friction coefficient
    dt = 0.001  # Timestep (atomic units)

    # Sampling parameters
    # FINAL: Optimized for convergence and reliable statistics
    N_steps = 30000  # Extended for better convergence (R-hat < 1.1)
    N_burnin = 5000  # Extended burn-in for equilibration
    exchange_freq = 10  # Attempt replica exchange every N steps
    thin_freq = 10  # Sample every 10 steps

    # Grid for charge density evaluation
    grid_points = 30  # Grid resolution
    # DEBUGGED: Extended grid range to capture more electron density
    grid_range = 4.0  # Increased from 3.0 Bohr for better electron integration

    # ============================================
    # POTENTIAL ENERGY SURFACE
    # ============================================

    def morse_potential(r: float, D_e: float = 0.17, a: float = 1.0, r_e: float = 1.4) -> float:
        """
        Morse potential for diatomic molecule.
        Default parameters approximate H2 molecule.

        Args:
            r: Internuclear distance
            D_e: Dissociation energy (Hartree)
            a: Width parameter
            r_e: Equilibrium bond length (Bohr)
        """
        return D_e * (1 - np.exp(-a * (r - r_e)))**2

    def compute_potential_energy(R: np.ndarray) -> float:
        """
        Compute potential energy U(R) for the molecular configuration.

        Args:
            R: Nuclear coordinates, shape (N_atoms, 3) or flattened (3N,)

        Returns:
            Potential energy in Hartree
        """
        R = R.reshape(N_atoms, dim)

        # Internuclear distance
        r12 = np.linalg.norm(R[0] - R[1])

        # Prevent singularity
        r12 = max(r12, 0.1)

        # Morse potential for bond
        U = morse_potential(r12)

        # Add harmonic center-of-mass restraint to prevent drift
        com = np.mean(R, axis=0)
        U += 0.01 * np.sum(com**2)

        return U

    def compute_gradient(R: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """
        Compute gradient of potential energy using finite differences.

        Args:
            R: Nuclear coordinates, shape (3N,)
            epsilon: Finite difference step size

        Returns:
            Gradient dU/dR, shape (3N,)
        """
        grad = np.zeros_like(R)
        for i in range(len(R)):
            R_plus = R.copy()
            R_minus = R.copy()
            R_plus[i] += epsilon
            R_minus[i] -= epsilon
            grad[i] = (compute_potential_energy(R_plus) - compute_potential_energy(R_minus)) / (2 * epsilon)
        return grad

    # ============================================
    # ELECTRONIC CHARGE DENSITY (VECTORIZED)
    # ============================================

    def compute_charge_density_grid_vectorized(R: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """
        Compute charge density on a 3D grid using VECTORIZED operations.
        This is orders of magnitude faster than nested loops.

        Args:
            R: Nuclear coordinates, shape (N_atoms, 3)
            grid: 1D array of grid points along each axis

        Returns:
            3D array of charge density values
        """
        R = R.reshape(N_atoms, dim)
        n = len(grid)

        # Create 3D meshgrid of spatial coordinates
        X, Y, Z = np.meshgrid(grid, grid, grid, indexing='ij')

        # Stack into shape (n, n, n, 3) for vectorized distance computation
        grid_coords = np.stack([X, Y, Z], axis=-1)  # Shape: (n, n, n, 3)

        # Slater exponent for hydrogen-like orbitals
        zeta = 1.0

        # Initialize charge density
        rho_grid = np.zeros((n, n, n))

        # Sum contributions from each atom (vectorized)
        for i in range(N_atoms):
            # Distance from atom i to all grid points
            diff = grid_coords - R[i]  # Shape: (n, n, n, 3)
            dist = np.linalg.norm(diff, axis=-1)  # Shape: (n, n, n)

            # 1s Slater orbital contribution: |psi|^2 = (zeta^3/pi) * exp(-2*zeta*r)
            rho_grid += (zeta**3 / np.pi) * np.exp(-2 * zeta * dist)

        return rho_grid

    # ============================================
    # TEMPERATURE LADDER
    # ============================================

    def create_temperature_ladder(T_min: float, T_max: float, n_replicas: int) -> np.ndarray:
        """
        Create geometric temperature ladder for parallel tempering.

        Args:
            T_min: Target (lowest) temperature
            T_max: Highest temperature
            n_replicas: Number of replicas

        Returns:
            Array of temperatures
        """
        ratio = (T_max / T_min) ** (1.0 / (n_replicas - 1))
        temperatures = T_min * (ratio ** np.arange(n_replicas))
        return temperatures

    # ============================================
    # LANGEVIN DYNAMICS
    # ============================================

    def langevin_step(R: np.ndarray, beta: float, dt: float, gamma: float) -> np.ndarray:
        """
        Perform one overdamped Langevin dynamics step.

        dR = -gamma * grad(U) * dt + sqrt(2 * gamma / beta) * dW

        Args:
            R: Current configuration, shape (3N,)
            beta: Inverse temperature
            dt: Timestep
            gamma: Friction coefficient

        Returns:
            Updated configuration
        """
        grad_U = compute_gradient(R)
        noise_scale = np.sqrt(2 * gamma * dt / beta)
        noise = np.random.randn(len(R)) * noise_scale

        R_new = R - gamma * grad_U * dt + noise

        return R_new

    # ============================================
    # REPLICA EXCHANGE
    # ============================================

    def attempt_replica_exchange(
        replicas: np.ndarray,
        energies: np.ndarray,
        betas: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Attempt replica exchange moves between adjacent replicas.

        Args:
            replicas: Configurations for all replicas, shape (N_replicas, 3N)
            energies: Potential energies, shape (N_replicas,)
            betas: Inverse temperatures, shape (N_replicas,)

        Returns:
            Updated replicas, energies, and number of accepted exchanges
        """
        n_accepted = 0
        n_replicas = len(betas)

        # Random sweep through adjacent pairs
        for i in range(n_replicas - 1):
            # Metropolis criterion for exchange
            delta = (betas[i] - betas[i + 1]) * (energies[i] - energies[i + 1])

            if delta < 0 or np.random.rand() < np.exp(-delta):
                # Accept exchange
                replicas[i], replicas[i + 1] = replicas[i + 1].copy(), replicas[i].copy()
                energies[i], energies[i + 1] = energies[i + 1], energies[i]
                n_accepted += 1

        return replicas, energies, n_accepted

    # ============================================
    # DIAGNOSTICS
    # ============================================

    def compute_effective_sample_size(samples: np.ndarray) -> float:
        """
        Compute effective sample size using autocorrelation.

        Args:
            samples: 1D array of samples

        Returns:
            Effective sample size
        """
        n = len(samples)
        if n < 10:
            return float(n)

        # Compute autocorrelation using FFT for efficiency
        samples_centered = samples - np.mean(samples)

        # Use FFT-based autocorrelation
        fft_result = np.fft.fft(samples_centered, n=2*n)
        autocorr = np.fft.ifft(fft_result * np.conj(fft_result))[:n].real
        autocorr = autocorr / autocorr[0]

        # Integrate autocorrelation time (Sokal's method)
        tau_int = 1.0
        for k in range(1, min(n // 2, 1000)):
            if autocorr[k] < 0.05:
                break
            tau_int += 2 * autocorr[k]

        ess = n / tau_int
        return max(1.0, ess)

    def gelman_rubin_diagnostic(chains: List[np.ndarray]) -> float:
        """
        Compute Gelman-Rubin R-hat statistic for convergence assessment.

        Args:
            chains: List of sample chains, each shape (n_samples,)

        Returns:
            R-hat statistic (values close to 1.0 indicate convergence)
        """
        m = len(chains)  # Number of chains
        n = min(len(c) for c in chains)  # Length of chains

        if n < 10 or m < 2:
            return float('nan')

        # Truncate chains to same length
        chains = [c[:n] for c in chains]

        # Chain means and overall mean
        chain_means = np.array([np.mean(c) for c in chains])
        overall_mean = np.mean(chain_means)

        # Between-chain variance
        B = n * np.var(chain_means, ddof=1)

        # Within-chain variance
        W = np.mean([np.var(c, ddof=1) for c in chains])

        # Pooled variance estimate
        var_plus = ((n - 1) * W + B) / n

        # R-hat
        if W > 0:
            R_hat = np.sqrt(var_plus / W)
        else:
            R_hat = float('nan')

        return R_hat

    # ============================================
    # MAIN SIMULATION
    # ============================================

    try:
        # Set random seed for reproducibility
        np.random.seed(42)

        # Initialize temperature ladder
        temperatures = create_temperature_ladder(T_target, T_max, N_replicas)
        betas = 1.0 / (k_B * temperatures)

        # Initialize replicas at equilibrium geometry with small perturbations
        R_eq = np.array([
            [-0.7, 0.0, 0.0],  # Atom 1
            [0.7, 0.0, 0.0]    # Atom 2
        ]).flatten()

        replicas = np.zeros((N_replicas, N_dof))
        for k in range(N_replicas):
            perturbation = np.random.randn(N_dof) * 0.1 * np.sqrt(temperatures[k] / T_target)
            replicas[k] = R_eq + perturbation

        # Initialize energies
        energies = np.array([compute_potential_energy(R) for R in replicas])

        # Storage for samples and diagnostics
        samples_target = []  # Samples from target temperature replica
        energy_trace = []  # Energy trace for diagnostics
        exchange_accepts = 0
        exchange_attempts = 0

        # Create spatial grid for charge density
        grid = np.linspace(-grid_range, grid_range, grid_points)

        # Accumulated charge density (for running average)
        rho_accumulated = np.zeros((grid_points, grid_points, grid_points))
        n_density_samples = 0

        # ============================================
        # LANGEVIN DYNAMICS WITH REPLICA EXCHANGE
        # ============================================

        print("Starting Replica Exchange Langevin Dynamics...")
        print(f"Temperature ladder: {temperatures[:3]}... to {temperatures[-1]:.1f} K")
        print(f"Total steps: {N_steps}, Burn-in: {N_burnin}")
        print(f"Grid: {grid_points}³ points (vectorized computation)")
        print()

        for step in range(N_steps):
            # Langevin step for each replica
            for k in range(N_replicas):
                replicas[k] = langevin_step(replicas[k], betas[k], dt, gamma)
                energies[k] = compute_potential_energy(replicas[k])

            # Attempt replica exchange
            if step % exchange_freq == 0:
                replicas, energies, n_accept = attempt_replica_exchange(replicas, energies, betas)
                exchange_accepts += n_accept
                exchange_attempts += N_replicas - 1

            # Collect samples after burn-in
            if step >= N_burnin and step % thin_freq == 0:
                # Store sample from target temperature replica (index 0)
                samples_target.append(replicas[0].copy())
                energy_trace.append(energies[0])

                # Accumulate charge density (using vectorized function)
                rho_sample = compute_charge_density_grid_vectorized(replicas[0], grid)
                rho_accumulated += rho_sample
                n_density_samples += 1

            # Progress report
            if (step + 1) % 2000 == 0:
                current_rate = exchange_accepts / max(1, exchange_attempts)
                print(f"  Step {step + 1}/{N_steps}, "
                      f"E(T1) = {energies[0]:.4f} Ha, "
                      f"Exchange rate: {current_rate:.2%}")

        # ============================================
        # COMPUTE FINAL RESULTS
        # ============================================

        # Thermally averaged charge density
        rho_avg = rho_accumulated / n_density_samples

        # Convert samples to numpy array
        samples_target = np.array(samples_target)
        energy_trace = np.array(energy_trace)

        # Compute bond length distribution
        bond_lengths = np.array([
            np.linalg.norm(s.reshape(N_atoms, dim)[0] - s.reshape(N_atoms, dim)[1])
            for s in samples_target
        ])

        # Diagnostics
        ess_energy = compute_effective_sample_size(energy_trace)
        ess_bond = compute_effective_sample_size(bond_lengths)
        exchange_rate = exchange_accepts / max(1, exchange_attempts)

        # For Gelman-Rubin, split the chain
        mid = len(energy_trace) // 2
        if mid > 10:
            chains = [energy_trace[:mid], energy_trace[mid:]]
            r_hat = gelman_rubin_diagnostic(chains)
        else:
            r_hat = float('nan')

        # Statistical estimates
        mean_energy = np.mean(energy_trace)
        std_energy = np.std(energy_trace)
        mean_bond = np.mean(bond_lengths)
        std_bond = np.std(bond_lengths)

        # Charge density statistics at origin and bond midpoint
        mid_idx = grid_points // 2
        rho_at_origin = rho_avg[mid_idx, mid_idx, mid_idx]

        # Total electron count (integration check)
        dV = (2 * grid_range / grid_points) ** 3
        total_electrons = np.sum(rho_avg) * dV

        # ============================================
        # COMPILE RESULTS
        # ============================================

        results = {
            "status": "success",
            "method": "Replica Exchange Langevin Dynamics (Parallel Tempering)",

            # System parameters
            "system": {
                "N_atoms": N_atoms,
                "N_dof": N_dof,
                "target_temperature_K": T_target,
                "N_replicas": N_replicas,
                "temperature_range_K": [float(temperatures[0]), float(temperatures[-1])]
            },

            # Sampling statistics
            "sampling": {
                "total_steps": N_steps,
                "burnin_steps": N_burnin,
                "n_samples": len(samples_target),
                "n_density_samples": n_density_samples,
                "exchange_acceptance_rate": float(exchange_rate)
            },

            # Convergence diagnostics
            "diagnostics": {
                "effective_sample_size_energy": float(ess_energy),
                "effective_sample_size_bond": float(ess_bond),
                "gelman_rubin_R_hat": float(r_hat) if not np.isnan(r_hat) else "N/A",
                "exchange_rate_target": "20-40%",
                "exchange_rate_achieved": f"{exchange_rate:.1%}"
            },

            # Physical observables
            "observables": {
                "mean_potential_energy_Ha": float(mean_energy),
                "std_potential_energy_Ha": float(std_energy),
                "mean_bond_length_Bohr": float(mean_bond),
                "std_bond_length_Bohr": float(std_bond),
                "equilibrium_bond_length_Bohr": 1.4
            },

            # Charge density results
            "charge_density": {
                "grid_points": grid_points,
                "grid_range_Bohr": grid_range,
                "rho_at_origin": float(rho_at_origin),
                "integrated_electrons": float(total_electrons),
                "expected_electrons": 2.0,
                "rho_shape": list(rho_avg.shape)
            },

            # Reliability assessment
            "reliability": {
                "convergence_achieved": r_hat < 1.1 if not np.isnan(r_hat) else "uncertain",
                "exchange_rate_healthy": 0.15 < exchange_rate < 0.50,
                "sufficient_samples": len(samples_target) > 100,
                "overall_quality": "good" if (
                    (np.isnan(r_hat) or r_hat < 1.1) and
                    exchange_rate > 0.1 and
                    len(samples_target) > 100
                ) else "needs_more_sampling"
            },

            # Raw data (truncated for display)
            "sample_bond_lengths_first_10": bond_lengths[:10].tolist(),
            "energy_trace_first_10": energy_trace[:10].tolist()
        }

        # Print summary
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Method: {results['method']}")
        print(f"\nSystem: {N_atoms} atoms, {N_dof} DOF, T = {T_target} K")
        print(f"Replicas: {N_replicas} (T range: {temperatures[0]:.0f} - {temperatures[-1]:.0f} K)")
        print(f"\nSampling: {len(samples_target)} samples collected")
        print(f"Exchange acceptance rate: {exchange_rate:.1%}")
        print(f"\nConvergence Diagnostics:")
        print(f"  ESS (energy): {ess_energy:.1f}")
        print(f"  ESS (bond length): {ess_bond:.1f}")
        print(f"  Gelman-Rubin R̂: {r_hat:.3f}" if not np.isnan(r_hat) else "  Gelman-Rubin R̂: N/A")
        print(f"\nPhysical Observables:")
        print(f"  <U> = {mean_energy:.4f} ± {std_energy:.4f} Ha")
        print(f"  <r_bond> = {mean_bond:.3f} ± {std_bond:.3f} Bohr")
        print(f"\nCharge Density:")
        print(f"  Grid: {grid_points}³ points, range ±{grid_range} Bohr")
        print(f"  ρ(0,0,0) = {rho_at_origin:.4f}")
        print(f"  Integrated electrons: {total_electrons:.2f} (expected: 2.0)")
        print(f"\nReliability: {results['reliability']['overall_quality']}")
        print("="*60)

        return results

    except Exception as e:
        import traceback
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        print(f"ERROR: {e}")
        traceback.print_exc()
        return error_result


# Execute solver when run directly
if __name__ == "__main__":
    result = solver()
    print("\nFinal result dictionary keys:", list(result.keys()))
