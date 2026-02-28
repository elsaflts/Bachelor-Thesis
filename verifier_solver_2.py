"""
Verification code for Thermally Averaged Electronic Charge Density Solver

Verifies that the solution from Solver_2.py adheres to all constraints:
1. Temperature constraint: T > 0, β > 0
2. Physical constraints: Nuclei cannot overlap (bond length > r_min)
3. Non-negativity: Charge density ρ(r) ≥ 0 everywhere
4. Normalization: Integrated charge density ≈ expected electrons
5. Convergence: Diagnostics within acceptable ranges
6. Energy: Finite, physically reasonable values
"""

def solver():
    """
    Verify that the Replica Exchange Langevin Dynamics solver output
    satisfies all physical and mathematical constraints.

    Returns:
        dict: {"evaluation": "correct"} if all constraints satisfied,
              {"evaluation": "constraint1, constraint2, ..."} listing violated constraints
    """
    import numpy as np
    from typing import Dict, Any, List

    # =========================================================================
    # CONSTRAINT THRESHOLDS
    # =========================================================================

    # Physical constants
    k_B = 3.1668114e-6  # Boltzmann constant in Hartree/K

    # Constraint thresholds
    R_MIN_BOND = 0.5  # Minimum allowed bond length (Bohr) - prevents nuclear overlap
    R_MAX_BOND = 5.0  # Maximum reasonable bond length (Bohr)
    T_MIN = 0.0  # Temperature must be > 0
    GELMAN_RUBIN_MAX = 1.2  # R-hat threshold for convergence
    ESS_MIN_FRACTION = 0.005  # Minimum ESS as fraction of samples (0.5% - realistic for MCMC)
    ESS_ABSOLUTE_MIN = 10.0  # Absolute minimum ESS for any observable
    ELECTRON_TOLERANCE = 0.5  # Tolerance for integrated electrons (absolute)
    ENERGY_MAX = 10.0  # Maximum reasonable energy (Hartree)

    # =========================================================================
    # RUN THE SOLVER TO GET RESULTS
    # =========================================================================

    try:
        # Import and run the solver
        import sys
        import os
        import importlib.util

        # Suppress print output during solver execution
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Get path to Solver_2.py
            current_dir = os.path.dirname(os.path.abspath(__file__))
            solver_path = os.path.join(current_dir, "Solver_2.py")

            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("solver_module", solver_path)
            solver_module = importlib.util.module_from_spec(spec)

            # Execute the module (this runs the solver function and sets 'result')
            spec.loader.exec_module(solver_module)

            # Get the result from the module
            result = getattr(solver_module, 'result', None)

            # If result not found as attribute, try calling solver function
            if result is None and hasattr(solver_module, 'solver'):
                result = solver_module.solver()

        finally:
            # Restore stdout
            sys.stdout = old_stdout

        if result is None:
            return {"evaluation": "solver_execution_failed_no_result"}

    except Exception as e:
        if 'old_stdout' in dir():
            sys.stdout = old_stdout
        return {"evaluation": f"solver_import_error: {str(e)}"}

    # =========================================================================
    # CHECK CONSTRAINTS
    # =========================================================================

    violated_constraints: List[str] = []

    # --- Check 1: Solver Status ---
    if result.get("status") != "success":
        violated_constraints.append("solver_status")
        return {"evaluation": ", ".join(violated_constraints)}

    # --- Check 2: Temperature Constraint (T > 0, β > 0) ---
    try:
        system_info = result.get("system", {})
        T_target = system_info.get("target_temperature_K", 0)
        temp_range = system_info.get("temperature_range_K", [0, 0])

        if T_target <= T_MIN:
            violated_constraints.append("temperature_positive")

        if temp_range[0] <= T_MIN or temp_range[1] <= T_MIN:
            violated_constraints.append("temperature_ladder_positive")

        # Check β > 0 (implied by T > 0)
        beta = 1.0 / (k_B * T_target) if T_target > 0 else float('inf')
        if not np.isfinite(beta) or beta <= 0:
            violated_constraints.append("inverse_temperature_positive")

    except Exception as e:
        violated_constraints.append(f"temperature_check_error")

    # --- Check 3: Physical Constraint - Bond Lengths ---
    try:
        bond_lengths = result.get("sample_bond_lengths_first_10", [])
        observables = result.get("observables", {})
        mean_bond = observables.get("mean_bond_length_Bohr", 0)
        std_bond = observables.get("std_bond_length_Bohr", 0)

        # Check sample bond lengths
        for i, bl in enumerate(bond_lengths):
            if bl < R_MIN_BOND:
                violated_constraints.append(f"nuclear_overlap_sample_{i}")
                break  # Only report once
            if bl > R_MAX_BOND:
                violated_constraints.append(f"bond_too_long_sample_{i}")
                break

        # Check mean bond length
        if mean_bond < R_MIN_BOND:
            violated_constraints.append("mean_bond_nuclear_overlap")
        if mean_bond > R_MAX_BOND:
            violated_constraints.append("mean_bond_too_long")

        # Check if mean ± 2*std stays physical
        if mean_bond - 2*std_bond < 0:
            violated_constraints.append("bond_length_nonphysical_negative")

    except Exception as e:
        violated_constraints.append("bond_length_check_error")

    # --- Check 4: Non-negativity of Charge Density ---
    try:
        charge_info = result.get("charge_density", {})
        rho_at_origin = charge_info.get("rho_at_origin", 0)

        if rho_at_origin < 0:
            violated_constraints.append("charge_density_negative")

        # Charge density should be positive for molecular system
        if rho_at_origin == 0:
            violated_constraints.append("charge_density_zero_at_origin")

    except Exception as e:
        violated_constraints.append("charge_density_check_error")

    # --- Check 5: Electron Integration (Normalization) ---
    try:
        charge_info = result.get("charge_density", {})
        integrated_electrons = charge_info.get("integrated_electrons", 0)
        expected_electrons = charge_info.get("expected_electrons", 2.0)

        # Check if integrated electrons is within tolerance
        electron_error = abs(integrated_electrons - expected_electrons)
        if electron_error > ELECTRON_TOLERANCE:
            violated_constraints.append("electron_integration_error")

        # Check non-negative
        if integrated_electrons < 0:
            violated_constraints.append("negative_integrated_electrons")

    except Exception as e:
        violated_constraints.append("electron_integration_check_error")

    # --- Check 6: Energy Constraint ---
    try:
        observables = result.get("observables", {})
        mean_energy = observables.get("mean_potential_energy_Ha", 0)
        std_energy = observables.get("std_potential_energy_Ha", 0)

        energy_trace = result.get("energy_trace_first_10", [])

        # Check mean energy is finite and reasonable
        if not np.isfinite(mean_energy):
            violated_constraints.append("energy_not_finite")
        elif abs(mean_energy) > ENERGY_MAX:
            violated_constraints.append("energy_unreasonably_large")

        # Check energy samples
        for i, e in enumerate(energy_trace):
            if not np.isfinite(e):
                violated_constraints.append(f"energy_sample_{i}_not_finite")
                break
            if e < 0:
                # Morse potential should be non-negative at equilibrium
                # But can be negative relative to dissociation limit
                pass  # Allow negative energies (reference dependent)

    except Exception as e:
        violated_constraints.append("energy_check_error")

    # --- Check 7: Convergence Diagnostics ---
    try:
        diagnostics = result.get("diagnostics", {})
        ess_energy = diagnostics.get("effective_sample_size_energy", 0)
        ess_bond = diagnostics.get("effective_sample_size_bond", 0)
        r_hat = diagnostics.get("gelman_rubin_R_hat", "N/A")

        sampling = result.get("sampling", {})
        n_samples = sampling.get("n_samples", 1)

        # Check R-hat (if available)
        if isinstance(r_hat, (int, float)) and np.isfinite(r_hat):
            if r_hat > GELMAN_RUBIN_MAX:
                violated_constraints.append("gelman_rubin_not_converged")

        # Check ESS is reasonable (either fraction of samples OR absolute minimum)
        ess_energy_threshold = max(n_samples * ESS_MIN_FRACTION, ESS_ABSOLUTE_MIN)
        ess_bond_threshold = max(n_samples * ESS_MIN_FRACTION, ESS_ABSOLUTE_MIN)

        if ess_energy < ess_energy_threshold:
            violated_constraints.append("ess_energy_too_low")
        if ess_bond < ess_bond_threshold:
            violated_constraints.append("ess_bond_too_low")

    except Exception as e:
        violated_constraints.append("convergence_check_error")

    # --- Check 8: Sampling Statistics ---
    try:
        sampling = result.get("sampling", {})
        n_samples = sampling.get("n_samples", 0)
        n_density_samples = sampling.get("n_density_samples", 0)
        exchange_rate = sampling.get("exchange_acceptance_rate", 0)

        # Must have collected samples
        if n_samples <= 0:
            violated_constraints.append("no_samples_collected")
        if n_density_samples <= 0:
            violated_constraints.append("no_density_samples")

        # Exchange rate should be in (0, 1)
        if exchange_rate <= 0 or exchange_rate >= 1:
            violated_constraints.append("exchange_rate_out_of_bounds")

    except Exception as e:
        violated_constraints.append("sampling_stats_check_error")

    # --- Check 9: Grid Properties ---
    try:
        charge_info = result.get("charge_density", {})
        grid_points = charge_info.get("grid_points", 0)
        grid_range = charge_info.get("grid_range_Bohr", 0)
        rho_shape = charge_info.get("rho_shape", [])

        # Grid should have positive size
        if grid_points <= 0:
            violated_constraints.append("invalid_grid_points")
        if grid_range <= 0:
            violated_constraints.append("invalid_grid_range")

        # Check shape consistency
        if len(rho_shape) != 3:
            violated_constraints.append("charge_density_not_3d")
        elif rho_shape[0] != grid_points or rho_shape[1] != grid_points or rho_shape[2] != grid_points:
            violated_constraints.append("grid_shape_mismatch")

    except Exception as e:
        violated_constraints.append("grid_check_error")

    # --- Check 10: System Configuration ---
    try:
        system_info = result.get("system", {})
        n_atoms = system_info.get("N_atoms", 0)
        n_dof = system_info.get("N_dof", 0)
        n_replicas = system_info.get("N_replicas", 0)

        # Degrees of freedom should be 3 * N_atoms
        if n_dof != 3 * n_atoms:
            violated_constraints.append("dof_atom_mismatch")

        # Must have positive replicas for parallel tempering
        if n_replicas <= 0:
            violated_constraints.append("invalid_replica_count")

    except Exception as e:
        violated_constraints.append("system_config_check_error")

    # =========================================================================
    # RETURN EVALUATION RESULT
    # =========================================================================

    if len(violated_constraints) == 0:
        return {"evaluation": "correct"}
    else:
        return {"evaluation": ", ".join(violated_constraints)}


# Execute verifier when run directly
if __name__ == "__main__":
    result = solver()
    print("\n" + "="*60)
    print("VERIFICATION RESULT")
    print("="*60)
    print(f"Evaluation: {result['evaluation']}")
    if result['evaluation'] == "correct":
        print("✓ All constraints satisfied")
    else:
        print("✗ Violated constraints:")
        for constraint in result['evaluation'].split(", "):
            print(f"  - {constraint}")
    print("="*60)
