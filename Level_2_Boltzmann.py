import numpy as np

def solver():
    """
    Solve the Boltzmann expectation computation problem using direct enumeration.

    Computes E_π[x] = Σ_x x · π(x) where π(x) ∝ exp(-V(x)/T)
    for a discrete 2D molecular conformational space.

    Energy function: V(x1, x2) = min((x1-15)² + (x2-35)², (x1-40)² + (x2-10)² + Δ)

    Returns:
        Dictionary containing expected values and analysis results
    """
    try:
        # Define parameters
        delta = 10      # Energy offset between basins (makes basin 1 more favorable)
        T = 1.0         # Temperature
        grid_size = 51  # States from 0 to 50 inclusive

        # Validate parameters
        if T <= 0:
            return {"error": "Temperature must be positive"}
        if delta < 0:
            return {"error": "Delta must be non-negative"}

        # Define basin centers
        basin1_center = (15, 35)  # Lower energy basin (V = 0 at center)
        basin2_center = (40, 10)  # Higher energy basin (V = delta at center)

        # Step 1: Create meshgrid for all states (vectorized grid construction)
        x1_coords = np.arange(grid_size)
        x2_coords = np.arange(grid_size)
        X1, X2 = np.meshgrid(x1_coords, x2_coords, indexing='ij')

        # Step 2: Compute energy for each basin (vectorized)
        # Basin 1: E1 = (x1 - 15)² + (x2 - 35)²
        E1 = (X1 - basin1_center[0])**2 + (X2 - basin1_center[1])**2

        # Basin 2: E2 = (x1 - 40)² + (x2 - 10)² + Δ
        E2 = (X1 - basin2_center[0])**2 + (X2 - basin2_center[1])**2 + delta

        # Energy is the minimum of two basins (element-wise)
        V = np.minimum(E1, E2)

        # Step 3: Compute Boltzmann weights using log-sum-exp trick for numerical stability
        # This prevents underflow when V/T is large
        log_weights = -V / T
        max_log_weight = np.max(log_weights)

        # Shift by max for numerical stability: exp(log_w - max) instead of exp(log_w)
        weights = np.exp(log_weights - max_log_weight)

        # Step 4: Normalize to get probability distribution π(x)
        Z = np.sum(weights)  # Partition function (shifted)
        probabilities = weights / Z

        # Validate that probabilities sum to 1
        prob_sum = np.sum(probabilities)
        if not np.isclose(prob_sum, 1.0, rtol=1e-10):
            return {"error": f"Probabilities do not sum to 1: {prob_sum}"}

        # Step 5: Compute expected values E[x1] and E[x2]
        # E[x1] = Σ x1 · π(x1, x2)
        E_x1 = np.sum(X1 * probabilities)

        # E[x2] = Σ x2 · π(x1, x2)
        E_x2 = np.sum(X2 * probabilities)

        # Additional analysis: Calculate probability mass in each basin
        basin1_mask = E1 <= E2  # States belonging to basin 1
        basin1_prob = np.sum(probabilities[basin1_mask])
        basin2_prob = np.sum(probabilities[~basin1_mask])

        # Find the state with maximum probability
        max_idx = np.unravel_index(np.argmax(probabilities), probabilities.shape)
        max_prob_state = (int(max_idx[0]), int(max_idx[1]))
        max_prob_value = probabilities[max_idx]

        # Compute actual partition function (for reference)
        log_Z = np.log(Z) + max_log_weight

        # Prepare comprehensive result dictionary
        result = {
            "E[x1]": float(E_x1),
            "E[x2]": float(E_x2),
            "E[x]": (float(E_x1), float(E_x2)),
            "temperature": T,
            "delta": delta,
            "basin1_probability": float(basin1_prob),
            "basin2_probability": float(basin2_prob),
            "max_probability_state": max_prob_state,
            "max_probability_value": float(max_prob_value),
            "log_partition_function": float(log_Z),
            "total_states": grid_size * grid_size,
            "objective_value": (float(E_x1), float(E_x2))
        }

        # Print formatted results
        print("=" * 60)
        print("Boltzmann Equilibrium Expected Value Computation")
        print("=" * 60)
        print(f"\nProblem Setup:")
        print(f"  Energy: V(x1,x2) = min((x1-15)² + (x2-35)²,")
        print(f"                         (x1-40)² + (x2-10)² + {delta})")
        print(f"  State space: {{0,...,50}} × {{0,...,50}} ({grid_size * grid_size} states)")
        print(f"\nParameters:")
        print(f"  Temperature T = {T}")
        print(f"  Energy offset Δ = {delta}")
        print(f"\nBasin Information:")
        print(f"  Basin 1: Center at {basin1_center}, V_min = 0")
        print(f"  Basin 2: Center at {basin2_center}, V_min = {delta}")
        print(f"\n" + "=" * 60)
        print(f"RESULTS for T = {T}")
        print("=" * 60)
        print(f"\n  E[x1] = {E_x1:.6f}")
        print(f"  E[x2] = {E_x2:.6f}")
        print(f"\n  >>> E_π[x] = ({E_x1:.6f}, {E_x2:.6f}) <<<")
        print(f"\nBasin Probability Analysis:")
        print(f"  P(state in Basin 1) = {basin1_prob:.10f}")
        print(f"  P(state in Basin 2) = {basin2_prob:.10f}")
        print(f"\nMost Probable State:")
        print(f"  State: {max_prob_state}")
        print(f"  Probability: {max_prob_value:.10f}")
        print(f"  Energy: {V[max_prob_state]:.1f}")
        print("=" * 60)

        return result

    except Exception as e:
        return {"error": str(e)}


# Execute the solver when script is run
if __name__ == "__main__":
    solution = solver()
    print("\nFinal Solution Dictionary:")
    for key, value in solution.items():
        print(f"  {key}: {value}")
