def solver():
    """
    Verify if the solution to the Boltzmann distribution problem satisfies all constraints.

    Returns:
        Dictionary with "evaluation" key mapping to "correct" or list of violated constraints
    """
    import math

    # Define the result from the original solver
    # Running the original solver to get the result
    def original_solver(temperature=1.0):
        try:
            if temperature <= 0:
                return {"error": "Temperature must be positive (T > 0)"}

            states = []
            potentials = []
            unnormalized_probs = []

            for x1 in range(11):
                for x2 in range(11):
                    states.append((x1, x2))
                    V = (x1 - 3)**2 + (x2 - 7)**2
                    potentials.append(V)
                    prob = math.exp(-V / temperature)
                    unnormalized_probs.append(prob)

            Z = sum(unnormalized_probs)

            if Z == 0 or not math.isfinite(Z):
                return {"error": "Partition function is invalid (zero or infinite)"}

            normalized_probs = [p / Z for p in unnormalized_probs]
            prob_sum = sum(normalized_probs)

            if not (0.9999 <= prob_sum <= 1.0001):
                return {"error": f"Distribution normalization failed. Sum = {prob_sum}"}

            E_x1 = sum(states[i][0] * normalized_probs[i] for i in range(len(states)))
            E_x2 = sum(states[i][1] * normalized_probs[i] for i in range(len(states)))

            Var_x1 = sum((states[i][0] - E_x1)**2 * normalized_probs[i] for i in range(len(states)))
            Var_x2 = sum((states[i][1] - E_x2)**2 * normalized_probs[i] for i in range(len(states)))

            Std_x1 = math.sqrt(Var_x1)
            Std_x2 = math.sqrt(Var_x2)

            max_prob_idx = normalized_probs.index(max(normalized_probs))
            mode_x1, mode_x2 = states[max_prob_idx]
            mode_probability = normalized_probs[max_prob_idx]

            min_potential_idx = potentials.index(min(potentials))
            optimal_x1, optimal_x2 = states[min_potential_idx]
            min_potential = potentials[min_potential_idx]

            results = {
                "expected_x1": E_x1,
                "expected_x2": E_x2,
                "expected_value": [E_x1, E_x2],
                "variance_x1": Var_x1,
                "variance_x2": Var_x2,
                "std_x1": Std_x1,
                "std_x2": Std_x2,
                "mode_x1": mode_x1,
                "mode_x2": mode_x2,
                "mode": [mode_x1, mode_x2],
                "mode_probability": mode_probability,
                "optimal_state": [optimal_x1, optimal_x2],
                "minimum_potential": min_potential,
                "partition_function": Z,
                "temperature": temperature,
                "total_states": len(states),
                "solution_method": "Exact Enumeration (Strategy 1)",
                "distribution_sum": prob_sum
            }

            return results

        except Exception as e:
            return {"error": str(e)}

    # Get the solution
    solution = original_solver(temperature=1.0)

    # Check if solution has error
    if "error" in solution:
        return {"evaluation": "error: " + solution["error"]}

    # List to track constraint violations
    violations = []

    # Constraint 1: Temperature must be positive (T > 0)
    if solution["temperature"] <= 0:
        violations.append("temperature")

    # Constraint 2: Distribution must sum to 1 (normalization constraint)
    if not (0.9999 <= solution["distribution_sum"] <= 1.0001):
        violations.append("distribution_sum")

    # Constraint 3: Expected values must be within valid range [0, 10]
    if not (0 <= solution["expected_x1"] <= 10):
        violations.append("expected_x1")
    if not (0 <= solution["expected_x2"] <= 10):
        violations.append("expected_x2")

    # Constraint 4: Mode values must be integers in {0, 1, ..., 10}
    if not (isinstance(solution["mode_x1"], int) and 0 <= solution["mode_x1"] <= 10):
        violations.append("mode_x1")
    if not (isinstance(solution["mode_x2"], int) and 0 <= solution["mode_x2"] <= 10):
        violations.append("mode_x2")

    # Constraint 5: Optimal state values must be integers in {0, 1, ..., 10}
    if not (isinstance(solution["optimal_state"][0], int) and 0 <= solution["optimal_state"][0] <= 10):
        violations.append("optimal_state_x1")
    if not (isinstance(solution["optimal_state"][1], int) and 0 <= solution["optimal_state"][1] <= 10):
        violations.append("optimal_state_x2")

    # Constraint 6: Optimal state should minimize potential V(x1, x2) = (x1-3)² + (x2-7)²
    # The minimum is at (3, 7) with V = 0
    optimal_x1, optimal_x2 = solution["optimal_state"]
    expected_optimal = (3, 7)
    if (optimal_x1, optimal_x2) != expected_optimal:
        violations.append("optimal_state_correctness")

    # Constraint 7: Minimum potential should be 0
    if abs(solution["minimum_potential"] - 0.0) > 0.001:
        violations.append("minimum_potential")

    # Constraint 8: Partition function must be positive and finite
    if not (solution["partition_function"] > 0 and math.isfinite(solution["partition_function"])):
        violations.append("partition_function")

    # Constraint 9: Mode probability must be between 0 and 1
    if not (0 <= solution["mode_probability"] <= 1):
        violations.append("mode_probability")

    # Constraint 10: Variances must be non-negative
    if solution["variance_x1"] < 0:
        violations.append("variance_x1")
    if solution["variance_x2"] < 0:
        violations.append("variance_x2")

    # Constraint 11: Standard deviations must be non-negative
    if solution["std_x1"] < 0:
        violations.append("std_x1")
    if solution["std_x2"] < 0:
        violations.append("std_x2")

    # Constraint 12: Total states should be 121 (11 x 11)
    if solution["total_states"] != 121:
        violations.append("total_states")

    # Format result
    if len(violations) == 0:
        result = {"evaluation": "correct"}
    else:
        result = {"evaluation": ", ".join(violations)}

    # Print result
    print(result)

    return result

if __name__ == "__main__":
    verification_result = solver()
