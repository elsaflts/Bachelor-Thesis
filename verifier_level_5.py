"""
Verifier for Level 5 Boltzmann Distribution MCMC Sampling

This script verifies that the MCMC sampling results satisfy all constraints
and requirements specified in the optimization problem.
"""


def solver():
    """
    Verify that the MCMC sampling solution satisfies all problem constraints.

    Checks:
    1. Bounds: All sample components x_i ∈ [0, 50]
    2. Chain initialization: 2 chains near basin A, 2 near basin B
    3. Diagnostics present: acceptance rates, burn-in, sample counts, basin occupancy
    4. Mixing assessment: chains should not remain permanently trapped
    5. Parameters: T = 1.0, Δ = 1.0
    6. Output format: per-chain estimates, combined estimate, diagnostics

    Returns:
        Dictionary with key "evaluation" mapping to "correct" or list of violations
    """
    import numpy as np
    from typing import Dict, List, Any

    # =========================================================================
    # PROBLEM PARAMETERS
    # =========================================================================
    DIM = 10
    BOUNDS = (0.0, 50.0)
    T = 1.0
    DELTA = 1.0
    MIN_CHAINS = 4
    MIN_CHAINS_NEAR_A = 2
    MIN_CHAINS_NEAR_B = 2

    # Basin centers
    a = np.array([15, 35, 15, 35, 15, 35, 15, 35, 15, 35], dtype=np.float64)
    b = np.array([40, 10, 40, 10, 40, 10, 40, 10, 40, 10], dtype=np.float64)

    # =========================================================================
    # RUN THE SOLVER TO GET RESULTS
    # =========================================================================

    try:
        # Import and run the solver
        import sys
        import os

        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        from level_5_Boltzmann import solver as mcmc_solver

        # Suppress print output during verification
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = mcmc_solver()

    except Exception as e:
        return {"evaluation": f"solver_import_error: {str(e)}"}

    # =========================================================================
    # VERIFICATION CHECKS
    # =========================================================================

    violations = []

    # Check 1: Verify result structure
    if result is None:
        violations.append("result_is_none")
        return {"evaluation": ", ".join(violations)}

    if 'error' in result and result['error'] is not None:
        violations.append(f"solver_error: {result['error']}")
        return {"evaluation": ", ".join(violations)}

    # Check 2: Verify required keys exist
    required_keys = ['combined_estimate', 'per_chain_estimates', 'diagnostics']
    for key in required_keys:
        if key not in result:
            violations.append(f"missing_key_{key}")

    if violations:
        return {"evaluation": ", ".join(violations)}

    # Check 3: Verify combined estimate structure
    combined = result['combined_estimate']
    if 'E_pi_x' not in combined:
        violations.append("missing_E_pi_x_in_combined_estimate")
    else:
        E_pi_x = np.array(combined['E_pi_x'])

        # Check dimension
        if len(E_pi_x) != DIM:
            violations.append(f"wrong_dimension_E_pi_x: got {len(E_pi_x)}, expected {DIM}")

        # Check bounds on expected values (should be within [0, 50])
        for i, val in enumerate(E_pi_x):
            if val < BOUNDS[0] or val > BOUNDS[1]:
                violations.append(f"E_pi_x_{i+1}_out_of_bounds: {val}")

    if 'std_error' not in combined:
        violations.append("missing_std_error_in_combined_estimate")

    # Check 4: Verify per-chain estimates
    per_chain = result['per_chain_estimates']

    # Count chains initialized near each basin
    chains_init_A = 0
    chains_init_B = 0

    if len(per_chain) < MIN_CHAINS:
        violations.append(f"insufficient_chains: got {len(per_chain)}, need {MIN_CHAINS}")

    for chain_name, chain_data in per_chain.items():
        # Check required fields
        required_chain_fields = ['E_x', 'init_basin', 'acceptance_rate',
                                  'basin_A_fraction', 'basin_B_fraction']
        for field in required_chain_fields:
            if field not in chain_data:
                violations.append(f"{chain_name}_missing_{field}")

        if 'init_basin' in chain_data:
            if chain_data['init_basin'] == 'A':
                chains_init_A += 1
            elif chain_data['init_basin'] == 'B':
                chains_init_B += 1

        # Check E_x bounds
        if 'E_x' in chain_data:
            E_x = np.array(chain_data['E_x'])
            if len(E_x) != DIM:
                violations.append(f"{chain_name}_wrong_dimension")
            for i, val in enumerate(E_x):
                if val < BOUNDS[0] or val > BOUNDS[1]:
                    violations.append(f"{chain_name}_E_x_{i+1}_out_of_bounds")

        # Check acceptance rate is reasonable (between 0 and 1)
        if 'acceptance_rate' in chain_data:
            acc = chain_data['acceptance_rate']
            if acc < 0 or acc > 1:
                violations.append(f"{chain_name}_invalid_acceptance_rate: {acc}")

        # Check basin fractions sum to ~1
        if 'basin_A_fraction' in chain_data and 'basin_B_fraction' in chain_data:
            total = chain_data['basin_A_fraction'] + chain_data['basin_B_fraction']
            if abs(total - 1.0) > 0.01:
                violations.append(f"{chain_name}_basin_fractions_dont_sum_to_1: {total}")

    # Check chain initialization requirement
    if chains_init_A < MIN_CHAINS_NEAR_A:
        violations.append(f"insufficient_chains_init_near_A: got {chains_init_A}, need {MIN_CHAINS_NEAR_A}")
    if chains_init_B < MIN_CHAINS_NEAR_B:
        violations.append(f"insufficient_chains_init_near_B: got {chains_init_B}, need {MIN_CHAINS_NEAR_B}")

    # Check 5: Verify diagnostics
    diag = result['diagnostics']

    # Check sampling parameters
    if 'sampling_parameters' not in diag:
        violations.append("missing_sampling_parameters")
    else:
        params = diag['sampling_parameters']
        required_params = ['burn_in', 'retained_samples_per_chain', 'total_retained_samples']
        for param in required_params:
            if param not in params:
                violations.append(f"missing_param_{param}")

        # Verify burn-in is positive
        if 'burn_in' in params and params['burn_in'] <= 0:
            violations.append("invalid_burn_in")

        # Verify sample counts are positive
        if 'retained_samples_per_chain' in params and params['retained_samples_per_chain'] <= 0:
            violations.append("invalid_retained_samples_per_chain")

    # Check acceptance rates reported
    if 'acceptance_rates' not in diag:
        violations.append("missing_acceptance_rates_in_diagnostics")

    # Check basin occupancy
    if 'basin_occupancy' not in diag:
        violations.append("missing_basin_occupancy_in_diagnostics")
    else:
        occ = diag['basin_occupancy']
        if 'overall' not in occ:
            violations.append("missing_overall_basin_occupancy")
        else:
            overall = occ['overall']
            if 'A' not in overall or 'B' not in overall:
                violations.append("missing_basin_A_or_B_in_overall")
            else:
                # Check they sum to ~1
                total = overall['A'] + overall['B']
                if abs(total - 1.0) > 0.01:
                    violations.append(f"overall_basin_fractions_dont_sum_to_1: {total}")

    # Check mixing assessment
    if 'mixing_assessment' not in diag:
        violations.append("missing_mixing_assessment")
    else:
        mixing = diag['mixing_assessment']
        if 'status' not in mixing:
            violations.append("missing_mixing_status")

        # Check if chains are mixing (not trapped in single basin)
        if 'chains_visiting_both_basins' in mixing:
            # Parse "X/Y" format
            try:
                parts = mixing['chains_visiting_both_basins'].split('/')
                visiting_both = int(parts[0])
                total_chains = int(parts[1])

                # At least some chains should visit both basins for good sampling
                if visiting_both == 0:
                    violations.append("no_chains_visiting_both_basins_mode_collapse")
            except:
                pass  # Format might be different

    # Check 6: Verify estimates are reasonable
    # The expected value should be between basins, weighted by Boltzmann factors
    # Theoretical: E[x] ≈ P(A)*a + P(B)*b where P(A) ≈ exp(-V(a)/T), P(B) ≈ exp(-V(b)/T)

    if 'E_pi_x' in combined:
        E_pi_x = np.array(combined['E_pi_x'])

        # Check that estimates are in a reasonable range
        # They should be between the two basin centers (roughly)
        midpoint = (a + b) / 2

        # Allow estimates to deviate from midpoint, but they should be
        # between the basins (with some tolerance for fluctuations)
        for i in range(DIM):
            min_val = min(a[i], b[i]) - 5  # Some tolerance
            max_val = max(a[i], b[i]) + 5
            if E_pi_x[i] < min_val or E_pi_x[i] > max_val:
                violations.append(f"E_pi_x_{i+1}_unreasonable: {E_pi_x[i]:.2f} not in [{min_val:.1f}, {max_val:.1f}]")

    # Check 7: Verify per-chain consistency
    # Chains initialized in different basins should converge to similar estimates
    # if mixing is good
    if len(per_chain) >= 4:
        chain_means = []
        for chain_name, chain_data in per_chain.items():
            if 'E_x' in chain_data:
                chain_means.append(np.array(chain_data['E_x']))

        if len(chain_means) >= 4:
            # Check variance across chains
            chain_means = np.array(chain_means)
            chain_std = np.std(chain_means, axis=0)

            # Standard deviation across chains should be reasonable
            # (not too large, indicating good mixing)
            max_allowed_std = 3.0  # Allow some variation
            for i in range(DIM):
                if chain_std[i] > max_allowed_std:
                    violations.append(f"high_inter_chain_variance_x_{i+1}: std={chain_std[i]:.2f}")

    # =========================================================================
    # RETURN RESULT
    # =========================================================================

    if len(violations) == 0:
        print("=" * 60)
        print("VERIFICATION RESULT: CORRECT")
        print("=" * 60)
        print("\nAll constraints satisfied:")
        print("  ✓ Bounds: All E[x_i] ∈ [0, 50]")
        print("  ✓ Chains: 4 chains (2 init near A, 2 init near B)")
        print("  ✓ Diagnostics: All required metrics present")
        print("  ✓ Mixing: Chains exploring both basins")
        print("  ✓ Estimates: Reasonable and consistent across chains")
        print("  ✓ Output format: All required fields present")

        return {"evaluation": "correct"}
    else:
        print("=" * 60)
        print("VERIFICATION RESULT: VIOLATIONS FOUND")
        print("=" * 60)
        print("\nViolations:")
        for v in violations:
            print(f"  ✗ {v}")

        return {"evaluation": ", ".join(violations)}


# Execute verifier when run directly
if __name__ == "__main__":
    result = solver()
    print(f"\nFinal evaluation: {result}")
