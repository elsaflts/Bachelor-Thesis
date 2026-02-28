import numpy as np

def solver():
    """
    Solve the Boltzmann expectation computation problem for multiple temperatures.

    Uses Strategy 1: Vectorized Direct Enumeration with Temperature Loop

    Computes E_π[x] = Σ_x x · π(x; T) where π(x; T) ∝ exp(-V(x)/T)
    for a discrete 2D molecular conformational space at multiple temperatures.

    Energy function: V(x1, x2) = min((x1-15)² + (x2-35)², (x1-40)² + (x2-10)² + Δ)

    Returns:
        Dictionary containing expected values and analysis for each temperature
    """
    try:
        # ============================================================
        # PARAMETERS
        # ============================================================
        delta = 10                              # Energy offset (Basin 2 penalty)
        grid_size = 51                          # States from 0 to 50 inclusive
        temperatures = [1.0, 5.0, 10.0, 50.0, 100.0]  # Temperature values to evaluate

        # Basin centers (for reference and validation)
        basin1_center = (15, 35)  # Lower energy basin (V_min = 0)
        basin2_center = (40, 10)  # Higher energy basin (V_min = delta)

        # Validate parameters
        if delta < 0:
            return {"error": "Delta must be non-negative"}
        for T in temperatures:
            if T <= 0:
                return {"error": f"Temperature must be positive, got T={T}"}

        # ============================================================
        # STEP 1: Create meshgrid for all states (done once)
        # ============================================================
        x1_coords = np.arange(grid_size)
        x2_coords = np.arange(grid_size)
        X1, X2 = np.meshgrid(x1_coords, x2_coords, indexing='ij')

        # ============================================================
        # STEP 2: Pre-compute energy landscape (temperature-independent)
        # ============================================================
        # Basin 1: E1 = (x1 - 15)² + (x2 - 35)²
        E1 = (X1 - basin1_center[0])**2 + (X2 - basin1_center[1])**2

        # Basin 2: E2 = (x1 - 40)² + (x2 - 10)² + Δ
        E2 = (X1 - basin2_center[0])**2 + (X2 - basin2_center[1])**2 + delta

        # Energy is minimum of two basins (element-wise)
        V = np.minimum(E1, E2)

        # Pre-compute basin membership mask
        basin1_mask = E1 <= E2

        # ============================================================
        # STEP 3: Loop over temperatures and compute expectations
        # ============================================================
        results = {}
        all_results = []

        for T in temperatures:
            # Compute Boltzmann weights using log-sum-exp trick
            log_weights = -V / T
            max_log_weight = np.max(log_weights)

            # Shift by max for numerical stability
            weights = np.exp(log_weights - max_log_weight)

            # Normalize to get probability distribution π(x; T)
            Z = np.sum(weights)
            probabilities = weights / Z

            # Validate normalization
            prob_sum = np.sum(probabilities)
            if not np.isclose(prob_sum, 1.0, rtol=1e-10):
                return {"error": f"Probabilities do not sum to 1 at T={T}: {prob_sum}"}

            # Compute expected values
            E_x1 = float(np.sum(X1 * probabilities))
            E_x2 = float(np.sum(X2 * probabilities))

            # Compute basin probabilities
            basin1_prob = float(np.sum(probabilities[basin1_mask]))
            basin2_prob = float(np.sum(probabilities[~basin1_mask]))

            # Find maximum probability state
            max_idx = np.unravel_index(np.argmax(probabilities), probabilities.shape)
            max_prob_state = (int(max_idx[0]), int(max_idx[1]))
            max_prob_value = float(probabilities[max_idx])

            # Compute log partition function
            log_Z = float(np.log(Z) + max_log_weight)

            # Store results for this temperature
            temp_result = {
                "E[x1]": E_x1,
                "E[x2]": E_x2,
                "E[x]": (E_x1, E_x2),
                "basin1_probability": basin1_prob,
                "basin2_probability": basin2_prob,
                "max_probability_state": max_prob_state,
                "max_probability_value": max_prob_value,
                "log_partition_function": log_Z
            }

            results[f"T={T}"] = temp_result
            all_results.append({
                "temperature": T,
                **temp_result
            })

        # ============================================================
        # STEP 4: Compile final output
        # ============================================================
        final_result = {
            "parameters": {
                "delta": delta,
                "grid_size": grid_size,
                "total_states": grid_size * grid_size,
                "basin1_center": basin1_center,
                "basin2_center": basin2_center,
                "temperatures_evaluated": temperatures
            },
            "results_by_temperature": results,
            "summary": all_results
        }

        # ============================================================
        # PRINT FORMATTED OUTPUT
        # ============================================================
        print("=" * 80)
        print("Boltzmann Equilibrium Expected Values - Multi-Temperature Analysis")
        print("=" * 80)
        print(f"\nProblem Setup:")
        print(f"  Energy: V(x1,x2) = min((x1-15)² + (x2-35)², (x1-40)² + (x2-10)² + {delta})")
        print(f"  State space: {{0,...,50}} × {{0,...,50}} ({grid_size * grid_size} states)")
        print(f"  Basin 1: Center at {basin1_center}, V_min = 0")
        print(f"  Basin 2: Center at {basin2_center}, V_min = {delta}")

        print(f"\n" + "=" * 80)
        print("RESULTS BY TEMPERATURE")
        print("=" * 80)
        print(f"\n{'T':>8} | {'E[x1]':>10} | {'E[x2]':>10} | {'P(Basin1)':>12} | {'P(Basin2)':>12}")
        print("-" * 80)

        for r in all_results:
            print(f"{r['temperature']:>8.1f} | {r['E[x1]']:>10.6f} | {r['E[x2]']:>10.6f} | "
                  f"{r['basin1_probability']:>12.8f} | {r['basin2_probability']:>12.8f}")

        print("-" * 80)

        print(f"\n" + "=" * 80)
        print("EXPECTED VALUES E_π[x] = (E[x1], E[x2])")
        print("=" * 80)
        for r in all_results:
            print(f"  T = {r['temperature']:>6.1f}:  E_π[x] = ({r['E[x1]']:.6f}, {r['E[x2]']:.6f})")

        print("=" * 80)

        return final_result

    except Exception as e:
        return {"error": str(e)}


# Execute the solver when script is run
if __name__ == "__main__":
    solution = solver()

    print("\n" + "=" * 80)
    print("FINAL SOLUTION DICTIONARY")
    print("=" * 80)

    if "error" in solution:
        print(f"Error: {solution['error']}")
    else:
        print("\nParameters:")
        for key, value in solution["parameters"].items():
            print(f"  {key}: {value}")

        print("\nResults Summary:")
        for r in solution["summary"]:
            print(f"  T={r['temperature']}: E[x]=({r['E[x1]']:.6f}, {r['E[x2]']:.6f}), "
                  f"P(B1)={r['basin1_probability']:.6f}, P(B2)={r['basin2_probability']:.6f}")
