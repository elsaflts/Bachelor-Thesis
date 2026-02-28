import numpy as np

def solver():
    """
    Verify that the solution produced by the Boltzmann solver satisfies all constraints
    and is correct.
    """
    try:
        # The result produced by the solver code
        E_x1 = 5.4997215164306965
        E_x2 = 4.500278483569304
        T = 1.0
        num_states = 121

        violations = []

        # ---- Constraint 1: Grid size is correct (11 x 11 = 121 states) ----
        grid_size = 11
        if num_states != grid_size * grid_size:
            violations.append("Number_of_states")

        # ---- Constraint 2: Temperature must be strictly positive ----
        if T <= 0:
            violations.append("Temperature")

        # ---- Constraint 3: E[x1] and E[x2] must lie within the domain [0, 10] ----
        if not (0 <= E_x1 <= 10):
            violations.append("E[x1]_domain")
        if not (0 <= E_x2 <= 10):
            violations.append("E[x2]_domain")

        # ---- Constraint 4: Verify the result by independent exact enumeration ----
        xs = np.arange(grid_size)
        x1, x2 = np.meshgrid(xs, xs, indexing='ij')

        basin1 = (x1 - 3)**2 + (x2 - 7)**2
        basin2 = (x1 - 8)**2 + (x2 - 2)**2
        V = np.minimum(basin1, basin2)

        log_weights = -V / T
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        Z = weights.sum()

        expected_x1 = (weights * x1).sum() / Z
        expected_x2 = (weights * x2).sum() / Z

        tol = 1e-6
        if abs(E_x1 - expected_x1) > tol:
            violations.append("E[x1]_accuracy")
        if abs(E_x2 - expected_x2) > tol:
            violations.append("E[x2]_accuracy")

        # ---- Constraint 5: Symmetry check ----
        if abs((E_x1 + E_x2) - (expected_x1 + expected_x2)) > tol:
            violations.append("symmetry")

        # Build evaluation result
        if len(violations) == 0:
            result = {"evaluation": "correct"}
        else:
            result = {"evaluation": ",".join(violations)}

        print("Verification result:", result)
        print(f"  Reported E[x1] = {E_x1}, independently computed = {expected_x1:.10f}")
        print(f"  Reported E[x2] = {E_x2}, independently computed = {expected_x2:.10f}")
        return result

    except Exception as e:
        error_result = {"evaluation": f"verification_error: {str(e)}"}
        print(error_result)
        return error_result


# Execute the verifier
result = solver()
print("\nFinal verification:", result)
