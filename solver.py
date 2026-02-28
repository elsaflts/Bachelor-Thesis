"""
Thermally Averaged Electronic Charge Density Computation

Implements Hybrid Surrogate-Enhanced Adaptive Sampling to compute:
    ρ̄(r) = E_{π(R)}[ρ(r|R)]

where π(R) is the Boltzmann distribution at finite temperature.
"""

import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Suppress sklearn convergence warnings (expected for small noise levels)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


# =============================================================================
# Physical Constants
# =============================================================================

KB = 8.617333262e-5  # Boltzmann constant in eV/K
ANGSTROM_TO_BOHR = 1.8897259886


# =============================================================================
# Molecular System Definition
# =============================================================================

class WaterMolecule:
    """Simple water molecule model with Morse potential for O-H bonds."""

    def __init__(self):
        self.n_atoms = 3
        self.n_electrons = 10
        self.atom_types = ['O', 'H', 'H']
        self.masses = np.array([15.999, 1.008, 1.008])  # amu

        # Equilibrium geometry (Angstrom)
        self.r_eq_OH = 0.9572
        self.theta_eq = np.deg2rad(104.52)

        # Morse potential parameters for O-H bond
        self.D_e = 4.77  # eV (dissociation energy)
        self.a = 2.287   # Angstrom^-1 (width parameter)
        self.r_e = 0.9572  # Angstrom (equilibrium distance)

        # H-O-H bending force constant
        self.k_theta = 0.07  # eV/rad^2

    def get_equilibrium_geometry(self):
        """Return equilibrium geometry as (3, 3) array."""
        R = np.zeros((3, 3))
        # O at origin
        R[0] = [0.0, 0.0, 0.0]
        # H1 along x-axis
        R[1] = [self.r_eq_OH, 0.0, 0.0]
        # H2 at angle theta
        R[2] = [self.r_eq_OH * np.cos(self.theta_eq),
                self.r_eq_OH * np.sin(self.theta_eq), 0.0]
        return R


def compute_potential_energy(R, molecule):
    """
    Compute potential energy U(R) for given configuration.

    Uses Morse potential for O-H bonds and harmonic potential for H-O-H angle.

    Args:
        R: (n_atoms, 3) array of atomic positions in Angstrom
        molecule: WaterMolecule instance

    Returns:
        float: Potential energy in eV
    """
    # O-H distances
    r_OH1 = np.linalg.norm(R[1] - R[0])
    r_OH2 = np.linalg.norm(R[2] - R[0])

    # Morse potential for each O-H bond
    def morse(r):
        return molecule.D_e * (1 - np.exp(-molecule.a * (r - molecule.r_e)))**2

    U_stretch = morse(r_OH1) + morse(r_OH2)

    # H-O-H angle
    v1 = R[1] - R[0]
    v2 = R[2] - R[0]
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Harmonic bending potential
    U_bend = 0.5 * molecule.k_theta * (theta - molecule.theta_eq)**2

    return U_stretch + U_bend


def compute_potential_gradient(R, molecule, eps=1e-5):
    """Compute gradient of potential energy using finite differences."""
    grad = np.zeros_like(R)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R_plus = R.copy()
            R_minus = R.copy()
            R_plus[i, j] += eps
            R_minus[i, j] -= eps
            grad[i, j] = (compute_potential_energy(R_plus, molecule) -
                         compute_potential_energy(R_minus, molecule)) / (2 * eps)
    return grad


# =============================================================================
# Charge Density Model
# =============================================================================

def compute_charge_density(R, grid_points, molecule):
    """
    Compute electronic charge density ρ(r|R) on a grid.

    Uses a simplified Gaussian orbital model where each atom contributes
    a spherically symmetric charge distribution.

    Args:
        R: (n_atoms, 3) array of atomic positions
        grid_points: (n_grid, 3) array of grid points
        molecule: WaterMolecule instance

    Returns:
        (n_grid,) array: Charge density at each grid point
    """
    # Slater exponents (approximate)
    zeta = {'O': 2.275, 'H': 1.24}
    # Core + valence electrons per atom
    n_elec = {'O': 8, 'H': 1}

    rho = np.zeros(len(grid_points))

    for i, atom_type in enumerate(molecule.atom_types):
        # Distance from atom to grid points
        r = np.linalg.norm(grid_points - R[i], axis=1)

        # Slater-type orbital density (simplified as Gaussian for stability)
        alpha = zeta[atom_type]**2 / 2
        normalization = n_elec[atom_type] * (alpha / np.pi)**1.5
        rho += normalization * np.exp(-alpha * r**2)

    return rho


def create_density_grid(R, n_points=20, padding=2.0):
    """Create a 3D grid around the molecule for density evaluation."""
    # Find bounding box
    r_min = R.min(axis=0) - padding
    r_max = R.max(axis=0) + padding

    # Create grid
    x = np.linspace(r_min[0], r_max[0], n_points)
    y = np.linspace(r_min[1], r_max[1], n_points)
    z = np.linspace(r_min[2], r_max[2], n_points)

    # Grid spacing for integration
    dx = (r_max[0] - r_min[0]) / (n_points - 1)
    dy = (r_max[1] - r_min[1]) / (n_points - 1)
    dz = (r_max[2] - r_min[2]) / (n_points - 1)
    dV = dx * dy * dz

    # Meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    return grid_points, dV, (n_points, n_points, n_points)


# =============================================================================
# Surrogate Model (Gaussian Process)
# =============================================================================

class SurrogateModel:
    """Gaussian Process surrogate for potential energy surface."""

    def __init__(self, n_atoms):
        self.n_atoms = n_atoms
        self.n_features = n_atoms * 3

        # GP kernel: RBF with noise
        kernel = ConstantKernel(1.0) * RBF(length_scale=0.1) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3
        )

        self.X_train = None
        self.y_train = None
        self.is_trained = False

    def _flatten_config(self, R):
        """Convert (n_atoms, 3) to (n_features,) representation."""
        # Use internal coordinates relative to center of mass
        com = R.mean(axis=0)
        return (R - com).ravel()

    def fit(self, configurations, energies):
        """
        Train the surrogate model.

        Args:
            configurations: list of (n_atoms, 3) arrays
            energies: (n_samples,) array of potential energies
        """
        X = np.array([self._flatten_config(R) for R in configurations])
        y = np.array(energies)

        self.X_train = X
        self.y_train = y

        self.gp.fit(X, y)
        self.is_trained = True

    def predict(self, R, return_std=False):
        """Predict potential energy for configuration R."""
        if not self.is_trained:
            raise ValueError("Surrogate model not trained yet")

        X = self._flatten_config(R).reshape(1, -1)

        if return_std:
            mean, std = self.gp.predict(X, return_std=True)
            return mean[0], std[0]
        else:
            return self.gp.predict(X)[0]

    def predict_batch(self, configurations, return_std=False):
        """Predict for multiple configurations."""
        X = np.array([self._flatten_config(R) for R in configurations])

        if return_std:
            mean, std = self.gp.predict(X, return_std=True)
            return mean, std
        else:
            return self.gp.predict(X)

    def get_uncertainty(self, R):
        """Get prediction uncertainty for active learning."""
        _, std = self.predict(R, return_std=True)
        return std

    def update(self, new_configurations, new_energies):
        """Incrementally update the surrogate with new data."""
        new_X = np.array([self._flatten_config(R) for R in new_configurations])
        new_y = np.array(new_energies)

        if self.X_train is not None:
            self.X_train = np.vstack([self.X_train, new_X])
            self.y_train = np.concatenate([self.y_train, new_y])
        else:
            self.X_train = new_X
            self.y_train = new_y

        self.gp.fit(self.X_train, self.y_train)


# =============================================================================
# Langevin Dynamics Sampler
# =============================================================================

class LangevinSampler:
    """Overdamped Langevin dynamics sampler."""

    def __init__(self, molecule, temperature, dt=0.001, friction=1.0):
        """
        Args:
            molecule: WaterMolecule instance
            temperature: Temperature in Kelvin
            dt: Time step in ps
            friction: Friction coefficient (dimensionless)
        """
        self.molecule = molecule
        self.temperature = temperature
        self.beta = 1.0 / (KB * temperature)
        self.dt = dt
        self.friction = friction

        # Noise amplitude for overdamped Langevin
        self.noise_amplitude = np.sqrt(2 * KB * temperature * dt / friction)

    def step(self, R, potential_func, gradient_func=None):
        """
        Perform one Langevin dynamics step.

        Args:
            R: Current configuration
            potential_func: Function returning potential energy
            gradient_func: Function returning gradient (uses finite diff if None)

        Returns:
            R_new: Updated configuration
        """
        if gradient_func is None:
            # Use finite differences
            grad = self._finite_diff_gradient(R, potential_func)
        else:
            grad = gradient_func(R)

        # Overdamped Langevin: dR = -grad(U)/friction * dt + sqrt(2*kT*dt/friction) * dW
        noise = np.random.randn(*R.shape) * self.noise_amplitude

        R_new = R - (grad / self.friction) * self.dt + noise

        return R_new

    def _finite_diff_gradient(self, R, potential_func, eps=1e-5):
        """Compute gradient using finite differences."""
        grad = np.zeros_like(R)
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                R_plus = R.copy()
                R_minus = R.copy()
                R_plus[i, j] += eps
                R_minus[i, j] -= eps
                grad[i, j] = (potential_func(R_plus) - potential_func(R_minus)) / (2 * eps)
        return grad

    def run(self, R_init, potential_func, n_steps, thin=10, gradient_func=None):
        """
        Run Langevin dynamics and collect samples.

        Args:
            R_init: Initial configuration
            potential_func: Potential energy function
            n_steps: Total number of steps
            thin: Save every thin-th configuration
            gradient_func: Optional gradient function

        Returns:
            samples: List of configurations
            energies: List of potential energies
        """
        R = R_init.copy()
        samples = []
        energies = []

        for step in range(n_steps):
            R = self.step(R, potential_func, gradient_func)

            if step % thin == 0:
                samples.append(R.copy())
                energies.append(potential_func(R))

        return samples, energies


# =============================================================================
# Active Learning
# =============================================================================

def select_by_uncertainty(candidates, surrogate, n_select):
    """
    Select configurations with highest uncertainty (query-by-committee).

    Args:
        candidates: List of configurations
        surrogate: Trained SurrogateModel
        n_select: Number of configurations to select

    Returns:
        selected_indices: Indices of selected configurations
    """
    _, uncertainties = surrogate.predict_batch(candidates, return_std=True)

    # Select top-k by uncertainty
    selected_indices = np.argsort(uncertainties)[-n_select:]

    return selected_indices, uncertainties[selected_indices]


# =============================================================================
# Importance Reweighting
# =============================================================================

def compute_importance_weights(samples, true_energies, surrogate_energies, beta):
    """
    Compute importance weights w(R) = exp(-β(U(R) - Ũ(R))).

    Args:
        samples: List of configurations
        true_energies: Array of true potential energies
        surrogate_energies: Array of surrogate potential energies
        beta: Inverse temperature

    Returns:
        weights: Normalized importance weights
        ess: Effective sample size
    """
    # Log weights for numerical stability
    log_weights = -beta * (true_energies - surrogate_energies)

    # Normalize in log space
    log_weights_max = np.max(log_weights)
    weights = np.exp(log_weights - log_weights_max)
    weights /= weights.sum()

    # Effective sample size
    ess = 1.0 / np.sum(weights**2)

    return weights, ess


def compute_ess_ratio(weights):
    """Compute ESS as fraction of nominal sample size."""
    n = len(weights)
    ess = 1.0 / np.sum(weights**2)
    return ess / n


# =============================================================================
# Diagnostics
# =============================================================================

def compute_diagnostics(weights, surrogate, validation_configs, validation_energies, temperature):
    """Compute diagnostic metrics for convergence and quality."""
    diagnostics = {}

    # ESS ratio
    diagnostics['ess_ratio'] = compute_ess_ratio(weights)
    diagnostics['ess'] = 1.0 / np.sum(weights**2)
    diagnostics['n_samples'] = len(weights)

    # Weight statistics
    diagnostics['weight_max'] = weights.max()
    diagnostics['weight_min'] = weights.min()
    diagnostics['weight_std'] = weights.std()

    # Surrogate validation error
    if len(validation_configs) > 0:
        pred_energies = surrogate.predict_batch(validation_configs)
        rmse = np.sqrt(np.mean((pred_energies - validation_energies)**2))
        diagnostics['surrogate_rmse'] = rmse
        diagnostics['surrogate_rmse_kT'] = rmse / (KB * temperature)

    return diagnostics


# =============================================================================
# Strategy Switching Logic
# =============================================================================

class StrategyMonitor:
    """Monitor for detecting when to switch strategies."""

    def __init__(self, kT):
        self.kT = kT
        self.surrogate_errors = []
        self.ess_history = []
        self.stall_counter = 0
        self.last_best_error = float('inf')

    def check_surrogate_convergence(self, rmse):
        """Check if surrogate has failed to converge."""
        self.surrogate_errors.append(rmse)

        # Fail if error > kT after 5 retraining cycles
        if len(self.surrogate_errors) >= 5:
            recent_errors = self.surrogate_errors[-5:]
            if all(e > self.kT for e in recent_errors):
                return False, "Surrogate validation error > k_B T for 5 consecutive cycles"
        return True, None

    def check_ess(self, ess_ratio):
        """Check if importance weights have degenerated."""
        self.ess_history.append(ess_ratio)

        if ess_ratio < 0.1:
            return False, f"ESS ratio {ess_ratio:.3f} < 10% of sample size"
        return True, None

    def check_active_learning_progress(self, current_error):
        """Check if active learning has stalled."""
        if current_error < self.last_best_error * 0.95:
            self.stall_counter = 0
            self.last_best_error = current_error
        else:
            self.stall_counter += 1

        if self.stall_counter >= 3:
            return False, "No improvement in surrogate for 3 iterations"
        return True, None

    def check_numerical_stability(self, values):
        """Check for NaN/Inf in values."""
        if np.any(np.isnan(values)) or np.any(np.isinf(values)):
            return False, "NaN or Inf detected in numerical values"
        return True, None


# =============================================================================
# Main Solver
# =============================================================================

def solver():
    """
    Compute thermally averaged electronic charge density using
    Hybrid Surrogate-Enhanced Adaptive Sampling.

    Returns:
        dict: Contains:
            - 'rho_bar': Thermally averaged charge density on grid
            - 'grid_points': (n_grid, 3) array of grid coordinates
            - 'grid_shape': Shape of 3D grid
            - 'dV': Volume element for integration
            - 'diagnostics': Dictionary of convergence metrics
            - 'samples': List of sampled configurations
            - 'weights': Importance weights for each sample
            - 'total_charge': Integrated total charge (should equal n_electrons)

        Or if error:
            - 'error': Error message
            - 'strategy_switch': Information about strategy change if applicable
    """

    # =========================================================================
    # 1. Setup and Parameters
    # =========================================================================

    np.random.seed(42)  # For reproducibility

    # System parameters
    molecule = WaterMolecule()
    temperature = 300.0  # Kelvin
    beta = 1.0 / (KB * temperature)
    kT = KB * temperature

    # Algorithm parameters
    N_init = 20           # Initial training configurations
    N_iter = 10           # Number of adaptive iterations
    M_candidates = 50     # Candidates per iteration from Langevin sampling
    K_select = 5          # Configurations to select per iteration
    langevin_steps = 500  # Steps per Langevin run
    langevin_thin = 10    # Thinning factor

    # Grid parameters for density
    n_grid_points = 15    # Points per dimension
    grid_padding = 2.5    # Angstrom

    # Initialize strategy monitor
    monitor = StrategyMonitor(kT)
    current_strategy = "Hybrid Surrogate-Enhanced Adaptive Sampling"

    print(f"Starting {current_strategy}")
    print(f"Temperature: {temperature} K, β = {beta:.4f} eV^-1")
    print(f"k_B T = {kT:.6f} eV")

    # =========================================================================
    # 2. Initialize: Generate initial configurations and compute true values
    # =========================================================================

    print(f"\nPhase 1: Generating {N_init} initial configurations...")

    R_eq = molecule.get_equilibrium_geometry()

    # Generate initial configurations by small random perturbations
    configurations = []
    true_energies = []

    for i in range(N_init):
        # Random displacement (thermal scale)
        displacement = np.random.randn(3, 3) * 0.05  # ~0.05 Angstrom std
        R = R_eq + displacement

        U = compute_potential_energy(R, molecule)

        configurations.append(R)
        true_energies.append(U)

    true_energies = np.array(true_energies)
    print(f"Initial energy range: [{true_energies.min():.4f}, {true_energies.max():.4f}] eV")

    # Check numerical stability
    ok, msg = monitor.check_numerical_stability(true_energies)
    if not ok:
        return {'error': msg, 'strategy_switch': f"{current_strategy} failed: {msg}"}

    # =========================================================================
    # 3. Train initial surrogate model
    # =========================================================================

    print("\nPhase 2: Training initial surrogate model...")

    surrogate = SurrogateModel(molecule.n_atoms)
    surrogate.fit(configurations, true_energies)

    # Validate surrogate on training data
    pred_energies = surrogate.predict_batch(configurations)
    train_rmse = np.sqrt(np.mean((pred_energies - true_energies)**2))
    print(f"Initial surrogate training RMSE: {train_rmse:.6f} eV ({train_rmse/kT:.2f} k_B T)")

    # =========================================================================
    # 4. Adaptive Sampling Loop
    # =========================================================================

    print(f"\nPhase 3: Adaptive sampling ({N_iter} iterations)...")

    sampler = LangevinSampler(molecule, temperature, dt=0.0005, friction=1.0)
    all_samples = list(configurations)
    all_true_energies = list(true_energies)
    all_surrogate_energies = list(pred_energies)

    for iteration in range(N_iter):
        print(f"\n  Iteration {iteration + 1}/{N_iter}")

        # 4a. Run Langevin dynamics with surrogate to generate candidates
        R_start = all_samples[-1].copy()

        def surrogate_potential(R):
            return surrogate.predict(R)

        candidates, candidate_surrogate_energies = sampler.run(
            R_start,
            surrogate_potential,
            n_steps=langevin_steps,
            thin=langevin_thin
        )

        print(f"    Generated {len(candidates)} candidate samples")

        # 4b. Select high-uncertainty configurations via active learning
        selected_idx, uncertainties = select_by_uncertainty(
            candidates, surrogate, min(K_select, len(candidates))
        )

        selected_configs = [candidates[i] for i in selected_idx]
        print(f"    Selected {len(selected_configs)} high-uncertainty configurations")
        print(f"    Uncertainty range: [{uncertainties.min():.4f}, {uncertainties.max():.4f}]")

        # 4c. Compute true U(R) for selected configurations
        selected_true_energies = []
        for R in selected_configs:
            U = compute_potential_energy(R, molecule)
            selected_true_energies.append(U)
        selected_true_energies = np.array(selected_true_energies)

        # Check numerical stability
        ok, msg = monitor.check_numerical_stability(selected_true_energies)
        if not ok:
            print(f"    ⚠️ STRATEGY CHANGE: {current_strategy} → PT-Langevin")
            print(f"    Reason: {msg}")
            return {'error': msg, 'strategy_switch': f"Switch to PT-Langevin: {msg}"}

        # 4d. Update surrogate model
        surrogate.update(selected_configs, selected_true_energies)

        # Compute validation error
        val_pred = surrogate.predict_batch(selected_configs)
        val_rmse = np.sqrt(np.mean((val_pred - selected_true_energies)**2))
        print(f"    Validation RMSE: {val_rmse:.6f} eV ({val_rmse/kT:.2f} k_B T)")

        # Check surrogate convergence
        ok, msg = monitor.check_surrogate_convergence(val_rmse)
        if not ok:
            print(f"    ⚠️ STRATEGY CHANGE: {current_strategy} → PT-Langevin")
            print(f"    Reason: {msg}")
            return {'error': msg, 'strategy_switch': f"Switch to PT-Langevin: {msg}"}

        # Check active learning progress
        ok, msg = monitor.check_active_learning_progress(val_rmse)
        if not ok:
            print(f"    ⚠️ Warning: {msg}")
            # Don't switch immediately, just warn

        # Add selected samples to collection
        all_samples.extend(selected_configs)
        all_true_energies.extend(selected_true_energies)

        # Get surrogate predictions for all new samples
        selected_surrogate_energies = surrogate.predict_batch(selected_configs)
        all_surrogate_energies.extend(selected_surrogate_energies)

    # =========================================================================
    # 5. Compute importance weights
    # =========================================================================

    print("\nPhase 4: Computing importance weights...")

    all_true_energies = np.array(all_true_energies)
    all_surrogate_energies = np.array(all_surrogate_energies)

    weights, ess = compute_importance_weights(
        all_samples, all_true_energies, all_surrogate_energies, beta
    )

    ess_ratio = compute_ess_ratio(weights)
    print(f"Effective sample size: {ess:.1f} ({ess_ratio*100:.1f}% of {len(weights)} samples)")

    # Check ESS
    ok, msg = monitor.check_ess(ess_ratio)
    if not ok:
        print(f"⚠️ STRATEGY CHANGE: {current_strategy} → PT-Langevin")
        print(f"Reason: {msg}")
        return {'error': msg, 'strategy_switch': f"Switch to PT-Langevin: {msg}"}

    # =========================================================================
    # 6. Compute thermally averaged charge density
    # =========================================================================

    print("\nPhase 5: Computing thermally averaged charge density...")

    # Create density grid based on equilibrium geometry
    grid_points, dV, grid_shape = create_density_grid(R_eq, n_grid_points, grid_padding)
    print(f"Grid: {grid_shape[0]}x{grid_shape[1]}x{grid_shape[2]} = {len(grid_points)} points")

    # Compute weighted average of charge densities
    rho_bar = np.zeros(len(grid_points))

    for i, (R, w) in enumerate(zip(all_samples, weights)):
        rho = compute_charge_density(R, grid_points, molecule)
        rho_bar += w * rho

    # Verify total charge
    total_charge = np.sum(rho_bar) * dV
    print(f"Integrated total charge: {total_charge:.4f} (expected: {molecule.n_electrons})")

    # =========================================================================
    # 7. Compute final diagnostics
    # =========================================================================

    print("\nPhase 6: Computing diagnostics...")

    diagnostics = compute_diagnostics(
        weights, surrogate,
        all_samples[-K_select:], all_true_energies[-K_select:],
        temperature
    )

    diagnostics['total_charge'] = total_charge
    diagnostics['expected_charge'] = molecule.n_electrons
    diagnostics['charge_error'] = abs(total_charge - molecule.n_electrons)
    diagnostics['temperature'] = temperature
    diagnostics['n_iterations'] = N_iter
    diagnostics['strategy'] = current_strategy

    print("\nDiagnostics Summary:")
    print(f"  ESS ratio: {diagnostics['ess_ratio']*100:.1f}%")
    print(f"  Surrogate RMSE: {diagnostics.get('surrogate_rmse', 'N/A'):.6f} eV")
    print(f"  Weight range: [{diagnostics['weight_min']:.2e}, {diagnostics['weight_max']:.2e}]")
    print(f"  Charge conservation error: {diagnostics['charge_error']:.4f} electrons")

    # =========================================================================
    # 8. Return results
    # =========================================================================

    print("\n" + "="*60)
    print("Computation completed successfully!")
    print("="*60)

    return {
        'rho_bar': rho_bar,
        'grid_points': grid_points,
        'grid_shape': grid_shape,
        'dV': dV,
        'diagnostics': diagnostics,
        'samples': all_samples,
        'weights': weights,
        'true_energies': all_true_energies,
        'surrogate_energies': all_surrogate_energies,
        'total_charge': total_charge,
        'molecule': molecule
    }


# =============================================================================
# Verification Functions
# =============================================================================

def verify_harmonic_potential():
    """
    Verify algorithm on harmonic potential where analytical solution is known.

    For harmonic potential U(x) = 0.5 * k * x^2, the thermal average is:
    <x^2> = k_B T / k
    """
    print("\n" + "="*60)
    print("VERIFICATION: Harmonic potential test")
    print("="*60)

    np.random.seed(123)  # Reproducibility for verification

    # 1D harmonic oscillator
    k = 1.0  # eV/Angstrom^2
    temperature = 300.0
    kT = KB * temperature

    # Analytical result
    x2_analytical = kT / k
    print(f"Analytical <x^2> = k_B T / k = {x2_analytical:.6f} Angstrom^2")

    # Numerical estimate using Langevin dynamics (increased samples for reliability)
    dt = 0.001
    n_steps = 200000
    burn_in = 10000
    thin = 100

    x = 0.0
    noise_amp = np.sqrt(2 * kT * dt)

    samples = []
    for step in range(n_steps):
        # Overdamped Langevin
        force = -k * x
        x = x + force * dt + noise_amp * np.random.randn()

        if step > burn_in and step % thin == 0:
            samples.append(x)

    x2_numerical = np.mean(np.array(samples)**2)
    error = abs(x2_numerical - x2_analytical) / x2_analytical * 100

    print(f"Numerical <x^2> = {x2_numerical:.6f} Angstrom^2")
    print(f"Relative error: {error:.2f}%")

    return error < 15.0  # Pass if error < 15%


def verify_charge_conservation():
    """Verify that charge density integrates to correct number of electrons."""
    print("\n" + "="*60)
    print("VERIFICATION: Charge conservation test")
    print("="*60)

    molecule = WaterMolecule()
    R = molecule.get_equilibrium_geometry()

    grid_points, dV, _ = create_density_grid(R, n_points=30, padding=3.0)
    rho = compute_charge_density(R, grid_points, molecule)

    total_charge = np.sum(rho) * dV
    expected = molecule.n_electrons
    error = abs(total_charge - expected) / expected * 100

    print(f"Integrated charge: {total_charge:.4f} electrons")
    print(f"Expected charge: {expected} electrons")
    print(f"Relative error: {error:.2f}%")

    return error < 5.0  # Pass if error < 5%


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Suppress sklearn warnings during execution
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    # Run verification tests first
    print("Running verification tests...\n")

    test1_passed = verify_harmonic_potential()
    test2_passed = verify_charge_conservation()

    print("\n" + "="*60)
    print("Verification Summary:")
    print(f"  Harmonic potential test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"  Charge conservation test: {'PASSED' if test2_passed else 'FAILED'}")
    print("="*60)

    if test1_passed and test2_passed:
        print("\nProceeding with main computation...\n")
        result = solver()

        if 'error' in result:
            print(f"\nError: {result['error']}")
            if 'strategy_switch' in result:
                print(f"Strategy switch: {result['strategy_switch']}")
        else:
            print(f"\nFinal result shape: {result['rho_bar'].shape}")
            print(f"Grid shape: {result['grid_shape']}")
    else:
        print("\nVerification failed. Please check the implementation.")
