import numpy as np

def solver():
    """
    Verify that the Boltzmann equilibrium solution follows all problem constraints.

    Constraints to verify:
    1. Expected values E[x1], E[x2] must be in valid range [0, 50]
    2. Probability distribution must be properly normalized (sum to 1)
    3. Energy function must be correctly computed
    4. Boltzmann weights must follow π(x) ∝ exp(-V(x)/T)
    5. Expected values must match independent recomputation
    """
    try:
        # Given solution to verify
        solution_E_x1 = 15.00113494671756
        solution_E_x2 = 34.99886505328245

        # Problem parameters
        delta = 10
        T = 1.0
        grid_size = 51

        # Basin centers
        basin1_center = (15, 35)
        basin2_center = (40, 10)

        violations = []

        # Constraint 1: Expected values must be within valid state space [0, 50]
        if not (0 <= solution_E_x1 <= 50):
            violations.append("E[x1]_range")
        if not (0 <= solution_E_x2 <= 50):
            violations.append("E[x2]_range")

        # Constraint 2 & 3 & 4: Verify by independent recomputation
        # Create meshgrid for all states
        x1_coords = np.arange(grid_size)
        x2_coords = np.arange(grid_size)
        X1, X2 = np.meshgrid(x1_coords, x2_coords, indexing='ij')

        # Compute energy for each basin
        E1 = (X1 - basin1_center[0])**2 + (X2 - basin1_center[1])**2
        E2 = (X1 - basin2_center[0])**2 + (X2 - basin2_center[1])**2 + delta

        # Energy is minimum of two basins
        V = np.minimum(E1, E2)

        # Verify energy function constraints
        # Energy at basin 1 center should be 0
        if V[basin1_center[0], basin1_center[1]] != 0:
            violations.append("energy_basin1_min")

        # Energy at basin 2 center should be delta
        if V[basin2_center[0], basin2_center[1]] != delta:
            violations.append("energy_basin2_min")

        # Compute Boltzmann weights with log-sum-exp trick
        log_weights = -V / T
        max_log_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_log_weight)

        # Normalize probabilities
        Z = np.sum(weights)
        probabilities = weights / Z

        # Constraint: Probabilities must sum to 1 (normalization)
        prob_sum = np.sum(probabilities)
        if not np.isclose(prob_sum, 1.0, rtol=1e-10):
            violations.append("probability_normalization")

        # Constraint: All probabilities must be non-negative
        if np.any(probabilities < 0):
            violations.append("probability_non_negative")

        # Constraint: Probabilities must follow Boltzmann distribution
        # Check that higher energy states have lower probability
        # Compare basin centers
        prob_basin1 = probabilities[basin1_center[0], basin1_center[1]]
        prob_basin2 = probabilities[basin2_center[0], basin2_center[1]]

        # Basin 1 (lower energy) should have higher probability
        if not (prob_basin1 > prob_basin2):
            violations.append("boltzmann_ordering")

        # Recompute expected values
        recomputed_E_x1 = np.sum(X1 * probabilities)
        recomputed_E_x2 = np.sum(X2 * probabilities)

        # Constraint 5: Solution must match recomputation within tolerance
        tolerance = 1e-6
        if not np.isclose(solution_E_x1, recomputed_E_x1, rtol=tolerance):
            violations.append("E[x1]_accuracy")
        if not np.isclose(solution_E_x2, recomputed_E_x2, rtol=tolerance):
            violations.append("E[x2]_accuracy")

        # Additional physical constraints
        # At low temperature (T=1.0) with delta=10, expected value should be close to basin 1
        distance_to_basin1 = np.sqrt((solution_E_x1 - basin1_center[0])**2 +
                                      (solution_E_x2 - basin1_center[1])**2)
        distance_to_basin2 = np.sqrt((solution_E_x1 - basin2_center[0])**2 +
                                      (solution_E_x2 - basin2_center[1])**2)

        # Expected value should be much closer to basin 1 (lower energy)
        if not (distance_to_basin1 < distance_to_basin2):
            violations.append("physical_expectation")

        # Format output
        if len(violations) == 0:
            result = {"evaluation": "correct"}
        else:
            result = {"evaluation": ",".join(violations)}

        # Print verification results
        print("=" * 60)
        print("Constraint Verification Results")
        print("=" * 60)
        print(f"\nSolution being verified:")
        print(f"  E[x1] = {solution_E_x1}")
        print(f"  E[x2] = {solution_E_x2}")
        print(f"\nRecomputed values:")
        print(f"  E[x1] = {recomputed_E_x1}")
        print(f"  E[x2] = {recomputed_E_x2}")
        print(f"\nConstraint checks:")
        print(f"  E[x1] in [0,50]: {0 <= solution_E_x1 <= 50}")
        print(f"  E[x2] in [0,50]: {0 <= solution_E_x2 <= 50}")
        print(f"  Probabilities sum to 1: {np.isclose(prob_sum, 1.0)}")
        print(f"  All probabilities >= 0: {not np.any(probabilities < 0)}")
        print(f"  Boltzmann ordering correct: {prob_basin1 > prob_basin2}")
        print(f"  E[x1] matches recomputation: {np.isclose(solution_E_x1, recomputed_E_x1, rtol=tolerance)}")
        print(f"  E[x2] matches recomputation: {np.isclose(solution_E_x2, recomputed_E_x2, rtol=tolerance)}")
        print(f"  Closer to lower energy basin: {distance_to_basin1 < distance_to_basin2}")
        print(f"\n" + "=" * 60)
        print(f"EVALUATION: {result['evaluation']}")
        print("=" * 60)

        return result

    except Exception as e:
        return {"evaluation": f"error: {str(e)}"}


# Execute the verification
if __name__ == "__main__":
    verification_result = solver()
    print("\nFinal Verification Result:")
    print(verification_result)
