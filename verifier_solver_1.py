def solver():
    """
    Verification function to check if the thermally averaged charge density
    computation results adhere to the problem constraints.

    Returns:
        dict: {"evaluation": "correct"} if all constraints are satisfied,
              {"evaluation": "comma-separated list of failed constraints"} otherwise.
    """
    import numpy as np
    import warnings

    # Import necessary components
    from scipy.optimize import minimize
    from scipy.spatial.distance import cdist

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    # Physical Constants
    KB = 8.617333262e-5  # Boltzmann constant in eV/K

    # =========================================================================
    # Define all necessary classes and functions from the original solver
    # =========================================================================

    class WaterMolecule:
        def __init__(self):
            self.n_atoms = 3
            self.n_electrons = 10
            self.atom_types = ['O', 'H', 'H']
            self.masses = np.array([15.999, 1.008, 1.008])
            self.r_eq_OH = 0.9572
            self.theta_eq = np.deg2rad(104.52)
            self.D_e = 4.77
            self.a = 2.287
            self.r_e = 0.9572
            self.k_theta = 0.07

        def get_equilibrium_geometry(self):
            R = np.zeros((3, 3))
            R[0] = [0.0, 0.0, 0.0]
            R[1] = [self.r_eq_OH, 0.0, 0.0]
            R[2] = [self.r_eq_OH * np.cos(self.theta_eq),
                    self.r_eq_OH * np.sin(self.theta_eq), 0.0]
            return R

    def compute_potential_energy(R, molecule):
        r_OH1 = np.linalg.norm(R[1] - R[0])
        r_OH2 = np.linalg.norm(R[2] - R[0])
        def morse(r):
            return molecule.D_e * (1 - np.exp(-molecule.a * (r - molecule.r_e)))**2
        U_stretch = morse(r_OH1) + morse(r_OH2)
        v1 = R[1] - R[0]
        v2 = R[2] - R[0]
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        U_bend = 0.5 * molecule.k_theta * (theta - molecule.theta_eq)**2
        return U_stretch + U_bend

    def compute_charge_density(R, grid_points, molecule):
        zeta = {'O': 2.275, 'H': 1.24}
        n_elec = {'O': 8, 'H': 1}
        rho = np.zeros(len(grid_points))
        for i, atom_type in enumerate(molecule.atom_types):
            r = np.linalg.norm(grid_points - R[i], axis=1)
            alpha = zeta[atom_type]**2 / 2
            normalization = n_elec[atom_type] * (alpha / np.pi)**1.5
            rho += normalization * np.exp(-alpha * r**2)
        return rho

    def create_density_grid(R, n_points=20, padding=2.0):
        r_min = R.min(axis=0) - padding
        r_max = R.max(axis=0) + padding
        x = np.linspace(r_min[0], r_max[0], n_points)
        y = np.linspace(r_min[1], r_max[1], n_points)
        z = np.linspace(r_min[2], r_max[2], n_points)
        dx = (r_max[0] - r_min[0]) / (n_points - 1)
        dy = (r_max[1] - r_min[1]) / (n_points - 1)
        dz = (r_max[2] - r_min[2]) / (n_points - 1)
        dV = dx * dy * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        return grid_points, dV, (n_points, n_points, n_points)

    class SurrogateModel:
        def __init__(self, n_atoms):
            self.n_atoms = n_atoms
            self.n_features = n_atoms * 3
            kernel = ConstantKernel(1.0) * RBF(length_scale=0.1) + WhiteKernel(noise_level=1e-5)
            self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=3)
            self.X_train = None
            self.y_train = None
            self.is_trained = False

        def _flatten_config(self, R):
            com = R.mean(axis=0)
            return (R - com).ravel()

        def fit(self, configurations, energies):
            X = np.array([self._flatten_config(R) for R in configurations])
            y = np.array(energies)
            self.X_train = X
            self.y_train = y
            self.gp.fit(X, y)
            self.is_trained = True

        def predict(self, R, return_std=False):
            if not self.is_trained:
                raise ValueError("Surrogate model not trained yet")
            X = self._flatten_config(R).reshape(1, -1)
            if return_std:
                mean, std = self.gp.predict(X, return_std=True)
                return mean[0], std[0]
            else:
                return self.gp.predict(X)[0]

        def predict_batch(self, configurations, return_std=False):
            X = np.array([self._flatten_config(R) for R in configurations])
            if return_std:
                mean, std = self.gp.predict(X, return_std=True)
                return mean, std
            else:
                return self.gp.predict(X)

        def update(self, new_configurations, new_energies):
            new_X = np.array([self._flatten_config(R) for R in new_configurations])
            new_y = np.array(new_energies)
            if self.X_train is not None:
                self.X_train = np.vstack([self.X_train, new_X])
                self.y_train = np.concatenate([self.y_train, new_y])
            else:
                self.X_train = new_X
                self.y_train = new_y
            self.gp.fit(self.X_train, self.y_train)

    class LangevinSampler:
        def __init__(self, molecule, temperature, dt=0.001, friction=1.0):
            self.molecule = molecule
            self.temperature = temperature
            self.beta = 1.0 / (KB * temperature)
            self.dt = dt
            self.friction = friction
            self.noise_amplitude = np.sqrt(2 * KB * temperature * dt / friction)

        def step(self, R, potential_func, gradient_func=None):
            if gradient_func is None:
                grad = self._finite_diff_gradient(R, potential_func)
            else:
                grad = gradient_func(R)
            noise = np.random.randn(*R.shape) * self.noise_amplitude
            R_new = R - (grad / self.friction) * self.dt + noise
            return R_new

        def _finite_diff_gradient(self, R, potential_func, eps=1e-5):
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
            R = R_init.copy()
            samples = []
            energies = []
            for step in range(n_steps):
                R = self.step(R, potential_func, gradient_func)
                if step % thin == 0:
                    samples.append(R.copy())
                    energies.append(potential_func(R))
            return samples, energies

    def select_by_uncertainty(candidates, surrogate, n_select):
        _, uncertainties = surrogate.predict_batch(candidates, return_std=True)
        selected_indices = np.argsort(uncertainties)[-n_select:]
        return selected_indices, uncertainties[selected_indices]

    def compute_importance_weights(samples, true_energies, surrogate_energies, beta):
        log_weights = -beta * (true_energies - surrogate_energies)
        log_weights_max = np.max(log_weights)
        weights = np.exp(log_weights - log_weights_max)
        weights /= weights.sum()
        ess = 1.0 / np.sum(weights**2)
        return weights, ess

    # =========================================================================
    # Run the solver
    # =========================================================================

    np.random.seed(42)
    molecule = WaterMolecule()
    temperature = 300.0
    beta = 1.0 / (KB * temperature)
    kT = KB * temperature

    N_init = 20
    N_iter = 10
    K_select = 5
    langevin_steps = 500
    langevin_thin = 10
    n_grid_points = 15
    grid_padding = 2.5

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

    surrogate = SurrogateModel(molecule.n_atoms)
    surrogate.fit(configurations, true_energies)
    pred_energies = surrogate.predict_batch(configurations)

    sampler = LangevinSampler(molecule, temperature, dt=0.0005, friction=1.0)
    all_samples = list(configurations)
    all_true_energies = list(true_energies)
    all_surrogate_energies = list(pred_energies)

    for iteration in range(N_iter):
        R_start = all_samples[-1].copy()
        def surrogate_potential(R):
            return surrogate.predict(R)
        candidates, _ = sampler.run(R_start, surrogate_potential, n_steps=langevin_steps, thin=langevin_thin)
        selected_idx, _ = select_by_uncertainty(candidates, surrogate, min(K_select, len(candidates)))
        selected_configs = [candidates[i] for i in selected_idx]
        selected_true_energies = []
        for R in selected_configs:
            U = compute_potential_energy(R, molecule)
            selected_true_energies.append(U)
        selected_true_energies = np.array(selected_true_energies)
        surrogate.update(selected_configs, selected_true_energies)
        all_samples.extend(selected_configs)
        all_true_energies.extend(selected_true_energies)
        selected_surrogate_energies = surrogate.predict_batch(selected_configs)
        all_surrogate_energies.extend(selected_surrogate_energies)

    all_true_energies = np.array(all_true_energies)
    all_surrogate_energies = np.array(all_surrogate_energies)

    weights, ess = compute_importance_weights(all_samples, all_true_energies, all_surrogate_energies, beta)
    ess_ratio = ess / len(weights)

    grid_points, dV, grid_shape = create_density_grid(R_eq, n_grid_points, grid_padding)

    rho_bar = np.zeros(len(grid_points))
    for i, (R, w) in enumerate(zip(all_samples, weights)):
        rho = compute_charge_density(R, grid_points, molecule)
        rho_bar += w * rho

    total_charge = np.sum(rho_bar) * dV

    # Compute surrogate RMSE on last batch
    val_pred = surrogate.predict_batch(all_samples[-K_select:])
    val_true = all_true_energies[-K_select:]
    surrogate_rmse = np.sqrt(np.mean((val_pred - val_true)**2))

    # =========================================================================
    # Verification of constraints
    # =========================================================================

    failed_constraints = []

    # Constraint 1: Numerical stability - no NaN or Inf in rho_bar
    if np.any(np.isnan(rho_bar)) or np.any(np.isinf(rho_bar)):
        failed_constraints.append("rho_bar_numerical_stability")

    # Constraint 2: Charge conservation (within 10% tolerance)
    charge_error_ratio = abs(total_charge - molecule.n_electrons) / molecule.n_electrons
    if charge_error_ratio > 0.10:
        failed_constraints.append("charge_conservation")

    # Constraint 3: Weights normalization (sum to 1)
    weight_sum = np.sum(weights)
    if abs(weight_sum - 1.0) > 1e-6:
        failed_constraints.append("weights_normalization")

    # Constraint 4: Weights positivity
    if np.any(weights < 0):
        failed_constraints.append("weights_positivity")

    # Constraint 5: ESS ratio above threshold (>10%)
    if ess_ratio < 0.10:
        failed_constraints.append("ess_ratio")

    # Constraint 6: Surrogate accuracy (RMSE < 3*kT is acceptable for thermal averaging)
    if surrogate_rmse > 3 * kT:
        failed_constraints.append("surrogate_accuracy")

    # Constraint 7: Temperature is positive
    if temperature <= 0:
        failed_constraints.append("temperature_positive")

    # Constraint 8: rho_bar is non-negative (density cannot be negative)
    if np.any(rho_bar < -1e-10):  # Small tolerance for numerical errors
        failed_constraints.append("density_non_negative")

    # Constraint 9: Grid points properly defined
    if grid_points.shape[1] != 3 or len(grid_points) != np.prod(grid_shape):
        failed_constraints.append("grid_definition")

    # Constraint 10: Samples have valid structure
    for sample in all_samples:
        if sample.shape != (molecule.n_atoms, 3):
            failed_constraints.append("sample_structure")
            break
        if np.any(np.isnan(sample)) or np.any(np.isinf(sample)):
            failed_constraints.append("sample_numerical_stability")
            break

    # Constraint 11: True energies are finite and reasonable
    if np.any(np.isnan(all_true_energies)) or np.any(np.isinf(all_true_energies)):
        failed_constraints.append("energy_numerical_stability")

    # Constraint 12: Beta (inverse temperature) is positive
    if beta <= 0:
        failed_constraints.append("beta_positive")

    # Return evaluation result
    if len(failed_constraints) == 0:
        return {"evaluation": "correct"}
    else:
        return {"evaluation": ",".join(failed_constraints)}


# Run the verification
if __name__ == "__main__":
    result = solver()
    print(result)
