import numpy as np

def solver():
    """
    Verify that the multi-temperature Boltzmann equilibrium solutions follow all constraints.

    Constraints to verify for each temperature:
    1. Expected values E[x1], E[x2] must be in valid range [0, 50]
    2. Probability distribution must be properly normalized (sum to 1)
    3. Energy function must be correctly computed
    4. Boltzmann weights must follow π(x) ∝ exp(-V(x)/T)
    5. Expected values must match independent recomputation
    6. Physical behavior: E[x] should transition from Basin 1 to geometric center as T increases
    """
    try:
        # ============================================================
        # SOLUTIONS TO VERIFY (from solver output)
        # ============================================================
        solutions = {
            1.0: {"E[x1]": 15.001135, "E[x2]": 34.998865, "P_B1": 0.99995460, "P_B2": 0.00004540},
            5.0: {"E[x1]": 17.980073, "E[x2]": 32.019927, "P_B1": 0.88079708, "P_B2": 0.11920292},
            10.0: {"E[x1]": 21.723521, "E[x2]": 28.276479, "P_B1": 0.73105901, "P_B2": 0.26894099},
            50.0: {"E[x1]": 25.955457, "E[x2]": 24.044543, "P_B1": 0.55822329, "P_B2": 0.44177671},
            100.0: {"E[x1]": 25.851597, "E[x2]": 24.148403, "P_B1": 0.55389707, "P_B2": 0.44610293}
        }

        # Problem parameters
        delta = 10
        grid_size = 51
        basin1_center = (15, 35)
        basin2_center = (40, 10)
        geometric_center = (25.0, 25.0)

        # ============================================================
        # PRE-COMPUTE ENERGY LANDSCAPE
        # ============================================================
        x1_coords = np.arange(grid_size)
        x2_coords = np.arange(grid_size)
        X1, X2 = np.meshgrid(x1_coords, x2_coords, indexing='ij')

        E1 = (X1 - basin1_center[0])**2 + (X2 - basin1_center[1])**2
        E2 = (X1 - basin2_center[0])**2 + (X2 - basin2_center[1])**2 + delta
        V = np.minimum(E1, E2)
        basin1_mask = E1 <= E2

        # ============================================================
        # VERIFY EACH TEMPERATURE
        # ============================================================
        all_violations = []
        verification_details = {}

        print("=" * 80)
        print("Multi-Temperature Constraint Verification")
        print("=" * 80)

        for T, sol in solutions.items():
            violations = []
            details = {}

            # Recompute for verification
            log_weights = -V / T
            max_log_weight = np.max(log_weights)
            weights = np.exp(log_weights - max_log_weight)
            Z = np.sum(weights)
            probabilities = weights / Z

            recomputed_E_x1 = np.sum(X1 * probabilities)
            recomputed_E_x2 = np.sum(X2 * probabilities)
            recomputed_P_B1 = np.sum(probabilities[basin1_mask])
            recomputed_P_B2 = np.sum(probabilities[~basin1_mask])

            # Constraint 1: Expected values in valid range [0, 50]
            if not (0 <= sol["E[x1]"] <= 50):
                violations.append(f"T={T}_E[x1]_range")
            if not (0 <= sol["E[x2]"] <= 50):
                violations.append(f"T={T}_E[x2]_range")

            # Constraint 2: Probabilities sum to 1
            prob_sum = np.sum(probabilities)
            if not np.isclose(prob_sum, 1.0, rtol=1e-10):
                violations.append(f"T={T}_normalization")

            # Constraint 3: All probabilities non-negative
            if np.any(probabilities < 0):
                violations.append(f"T={T}_non_negative")

            # Constraint 4: Boltzmann ordering (lower energy = higher probability)
            prob_basin1_center = probabilities[basin1_center[0], basin1_center[1]]
            prob_basin2_center = probabilities[basin2_center[0], basin2_center[1]]
            if not (prob_basin1_center > prob_basin2_center):
                violations.append(f"T={T}_boltzmann_ordering")

            # Constraint 5: Solution matches recomputation
            tolerance = 1e-5
            if not np.isclose(sol["E[x1]"], recomputed_E_x1, rtol=tolerance):
                violations.append(f"T={T}_E[x1]_accuracy")
            if not np.isclose(sol["E[x2]"], recomputed_E_x2, rtol=tolerance):
                violations.append(f"T={T}_E[x2]_accuracy")
            if not np.isclose(sol["P_B1"], recomputed_P_B1, rtol=tolerance):
                violations.append(f"T={T}_P_B1_accuracy")
            if not np.isclose(sol["P_B2"], recomputed_P_B2, rtol=tolerance):
                violations.append(f"T={T}_P_B2_accuracy")

            # Store details
            details = {
                "E[x1]_in_range": 0 <= sol["E[x1]"] <= 50,
                "E[x2]_in_range": 0 <= sol["E[x2]"] <= 50,
                "normalized": np.isclose(prob_sum, 1.0),
                "non_negative": not np.any(probabilities < 0),
                "boltzmann_ordering": prob_basin1_center > prob_basin2_center,
                "E[x1]_match": np.isclose(sol["E[x1]"], recomputed_E_x1, rtol=tolerance),
                "E[x2]_match": np.isclose(sol["E[x2]"], recomputed_E_x2, rtol=tolerance),
                "recomputed_E[x1]": float(recomputed_E_x1),
                "recomputed_E[x2]": float(recomputed_E_x2)
            }
            verification_details[T] = details
            all_violations.extend(violations)

            # Print results for this temperature
            print(f"\nT = {T}:")
            print(f"  Solution:    E[x] = ({sol['E[x1]']:.6f}, {sol['E[x2]']:.6f})")
            print(f"  Recomputed:  E[x] = ({recomputed_E_x1:.6f}, {recomputed_E_x2:.6f})")
            print(f"  Constraints: ", end="")
            if len(violations) == 0:
                print("ALL PASSED ✓")
            else:
                print(f"VIOLATIONS: {violations}")

        # ============================================================
        # VERIFY PHYSICAL BEHAVIOR ACROSS TEMPERATURES
        # ============================================================
        print("\n" + "-" * 80)
        print("Physical Behavior Verification (across temperatures)")
        print("-" * 80)

        temps = sorted(solutions.keys())

        # Check 1: E[x] should move toward geometric center as T increases
        dist_to_basin1 = []
        dist_to_geom_center = []
        for T in temps:
            d1 = np.sqrt((solutions[T]["E[x1]"] - basin1_center[0])**2 +
                         (solutions[T]["E[x2]"] - basin1_center[1])**2)
            dg = np.sqrt((solutions[T]["E[x1]"] - geometric_center[0])**2 +
                         (solutions[T]["E[x2]"] - geometric_center[1])**2)
            dist_to_basin1.append(d1)
            dist_to_geom_center.append(dg)

        # At low T, should be close to Basin 1
        low_T_close_to_basin1 = dist_to_basin1[0] < 1.0
        print(f"  Low T close to Basin 1: {low_T_close_to_basin1} (distance = {dist_to_basin1[0]:.4f})")

        # At high T, should be closer to geometric center than at low T
        high_T_closer_to_center = dist_to_geom_center[-1] < dist_to_geom_center[0]
        print(f"  High T closer to center: {high_T_closer_to_center}")

        # Basin 1 probability should decrease with temperature
        p_b1_decreasing = all(solutions[temps[i]]["P_B1"] >= solutions[temps[i+1]]["P_B1"]
                              for i in range(len(temps)-1))
        print(f"  P(Basin1) decreases with T: {p_b1_decreasing}")

        if not low_T_close_to_basin1:
            all_violations.append("physical_low_T_localization")
        if not high_T_closer_to_center:
            all_violations.append("physical_high_T_delocalization")
        if not p_b1_decreasing:
            all_violations.append("physical_basin_probability_trend")

        # ============================================================
        # FINAL EVALUATION
        # ============================================================
        print("\n" + "=" * 80)
        if len(all_violations) == 0:
            result = {"evaluation": "correct"}
            print("FINAL EVALUATION: correct")
            print("All constraints satisfied for all temperatures.")
        else:
            result = {"evaluation": ",".join(all_violations)}
            print(f"FINAL EVALUATION: {result['evaluation']}")
        print("=" * 80)

        return result

    except Exception as e:
        return {"evaluation": f"error: {str(e)}"}


# Execute the verification
if __name__ == "__main__":
    verification_result = solver()
    print("\nFinal Verification Result:")
    print(verification_result)
