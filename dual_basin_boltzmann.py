import numpy as np

def solver():
    """
    Compute the expected value E_π[x] under the Boltzmann distribution
    π(x) ∝ exp(−V(x)/T) over the discrete state space {0,...,10}².

    V(x1, x2) = min((x1−3)² + (x2−7)², (x1−8)² + (x2−2)²)

    Uses exact enumeration since the state space has only 121 points.
    """
    try:
        # Define the grid: x1, x2 ∈ {0, 1, ..., 10}
        grid_size = 11
        xs = np.arange(grid_size)
        x1, x2 = np.meshgrid(xs, xs, indexing='ij')  # shape (11, 11)

        # Compute the potential V(x1, x2) = min of two quadratic basins
        basin1 = (x1 - 3)**2 + (x2 - 7)**2  # minimum at (3, 7)
        basin2 = (x1 - 8)**2 + (x2 - 2)**2  # minimum at (8, 2)
        V = np.minimum(basin1, basin2)

        # Temperature parameter
        T = 1.0

        if T <= 0:
            return {"error": "Temperature T must be strictly positive."}

        # Compute unnormalized log-weights with log-sum-exp trick
        log_weights = -V / T
        log_weights -= log_weights.max()  # shift for numerical stability

        weights = np.exp(log_weights)
        Z = weights.sum()

        if Z == 0:
            return {"error": "Partition function is zero; all weights collapsed."}

        # Compute expected values
        E_x1 = (weights * x1).sum() / Z
        E_x2 = (weights * x2).sum() / Z

        result = {
            "E[x1]": float(E_x1),
            "E[x2]": float(E_x2),
            "E[x]": [float(E_x1), float(E_x2)],
            "Temperature": T,
            "Number_of_states": grid_size * grid_size
        }

        # Print the solution
        print("=" * 50)
        print("Expected Value under Boltzmann Distribution")
        print("=" * 50)
        print(f"Temperature T = {T}")
        print(f"State space size = {grid_size} x {grid_size} = {grid_size**2}")
        print(f"E_pi[x1] = {E_x1:.6f}")
        print(f"E_pi[x2] = {E_x2:.6f}")
        print(f"E_pi[x]  = ({E_x1:.6f}, {E_x2:.6f})")
        print("=" * 50)

        # Sensitivity to temperature
        print("\nSensitivity to temperature:")
        print(f"{'T':>8s}  {'E[x1]':>10s}  {'E[x2]':>10s}")
        print("-" * 32)
        for t in [0.5, 1.0, 2.0, 5.0, 10.0]:
            lw = -V / t
            lw -= lw.max()
            w = np.exp(lw)
            z = w.sum()
            ex1 = (w * x1).sum() / z
            ex2 = (w * x2).sum() / z
            print(f"{t:8.1f}  {ex1:10.6f}  {ex2:10.6f}")

        return result

    except Exception as e:
        error_msg = f"Solver error: {str(e)}"
        print(error_msg)
        return {"error": error_msg}


if __name__ == "__main__":
    result = solver()
    print("\nFinal result:", result)
