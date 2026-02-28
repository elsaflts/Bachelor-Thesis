import numpy as np

def solver(temperatures=None):
    """
    Analyze how temperature affects the Boltzmann equilibrium expected value.

    At low T: Distribution concentrates on lowest energy states (Basin 1)
    At high T: Distribution becomes more uniform, approaching geometric center
    """
    if temperatures is None:
        temperatures = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0]

    # Problem parameters
    delta = 10
    grid_size = 51

    # Basin centers
    basin1_center = (15, 35)  # V_min = 0
    basin2_center = (40, 10)  # V_min = delta

    # Geometric center of state space
    geometric_center = (25, 25)

    # Create meshgrid for all states
    x1_coords = np.arange(grid_size)
    x2_coords = np.arange(grid_size)
    X1, X2 = np.meshgrid(x1_coords, x2_coords, indexing='ij')

    # Compute energy for each basin (temperature independent)
    E1 = (X1 - basin1_center[0])**2 + (X2 - basin1_center[1])**2
    E2 = (X1 - basin2_center[0])**2 + (X2 - basin2_center[1])**2 + delta
    V = np.minimum(E1, E2)

    results = []

    print("=" * 80)
    print("Temperature Analysis of Boltzmann Equilibrium Expected Values")
    print("=" * 80)
    print(f"\nEnergy: V(x1,x2) = min((x1-15)² + (x2-35)², (x1-40)² + (x2-10)² + {delta})")
    print(f"Basin 1 center: {basin1_center}, V_min = 0")
    print(f"Basin 2 center: {basin2_center}, V_min = {delta}")
    print(f"Geometric center: {geometric_center}")
    print("\n" + "-" * 80)
    print(f"{'T':>8} | {'E[x1]':>10} | {'E[x2]':>10} | {'P(Basin1)':>12} | {'P(Basin2)':>12} | Behavior")
    print("-" * 80)

    for T in temperatures:
        # Compute Boltzmann weights
        log_weights = -V / T
        max_log_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_log_weight)

        # Normalize
        Z = np.sum(weights)
        probabilities = weights / Z

        # Compute expected values
        E_x1 = np.sum(X1 * probabilities)
        E_x2 = np.sum(X2 * probabilities)

        # Basin probabilities
        basin1_mask = E1 <= E2
        basin1_prob = np.sum(probabilities[basin1_mask])
        basin2_prob = np.sum(probabilities[~basin1_mask])

        # Determine behavior
        if basin1_prob > 0.99:
            behavior = "Strongly localized at Basin 1"
        elif basin1_prob > 0.9:
            behavior = "Mostly Basin 1"
        elif basin1_prob > 0.7:
            behavior = "Basin 1 dominant"
        elif basin1_prob > 0.55:
            behavior = "Slight Basin 1 preference"
        else:
            behavior = "Nearly uniform"

        print(f"{T:>8.1f} | {E_x1:>10.4f} | {E_x2:>10.4f} | {basin1_prob:>12.8f} | {basin2_prob:>12.8f} | {behavior}")

        results.append({
            "temperature": T,
            "E[x1]": E_x1,
            "E[x2]": E_x2,
            "E[x]": (E_x1, E_x2),
            "basin1_probability": basin1_prob,
            "basin2_probability": basin2_prob
        })

    print("-" * 80)

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print("\n1. LOW TEMPERATURE (T → 0):")
    print(f"   - Distribution concentrates at global minimum (Basin 1 center)")
    print(f"   - E[x] → {basin1_center}")
    print(f"   - At T=0.5: E[x] ≈ ({results[0]['E[x1]']:.4f}, {results[0]['E[x2]']:.4f})")

    print("\n2. MODERATE TEMPERATURE (T ~ 10-50):")
    print(f"   - Both basins contribute, but Basin 1 still favored due to lower energy")
    print(f"   - Expected value shifts toward geometric center")

    print("\n3. HIGH TEMPERATURE (T → ∞):")
    print(f"   - Boltzmann factor exp(-V/T) → 1 for all states")
    print(f"   - Distribution becomes uniform over all states")
    print(f"   - E[x] → geometric center = {geometric_center}")
    print(f"   - At T=1000: E[x] ≈ ({results[-1]['E[x1]']:.4f}, {results[-1]['E[x2]']:.4f})")

    # Compute theoretical high-T limit (uniform distribution)
    uniform_E_x1 = np.mean(x1_coords)
    uniform_E_x2 = np.mean(x2_coords)
    print(f"   - Theoretical limit (uniform): ({uniform_E_x1}, {uniform_E_x2})")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    results = solver()

    print("\nKey Results Summary:")
    print("-" * 40)
    for r in results:
        print(f"T = {r['temperature']:>6.1f}: E[x] = ({r['E[x1]']:.4f}, {r['E[x2]']:.4f})")
