"""
Verification code to check if the MCMC solution adheres to all constraints
specified in the optimization problem.
"""

import numpy as np

def verify_solution():
    """
    Verify that the MCMC solution satisfies all problem constraints.
    Returns a dictionary with verification results.
    """

    # =========================================================================
    # EXPECTED RESULTS FROM THE SOLVER (extracted from output)
    # =========================================================================

    results = {
        "mcmc_estimate_combined": {
            "E_x1": 23.483461,
            "E_x2": 26.508340
        },
        "mcmc_estimate_chain1": {
            "E_x1": 14.990345,
            "E_x2": 34.981651
        },
        "mcmc_estimate_chain2": {
            "E_x1": 40.025793,
            "E_x2": 10.004584
        },
        "reference_estimate": {
            "E_x1": 23.481091,
            "E_x2": 26.518909
        },
        "absolute_error": {
            "E_x1": 0.002370,
            "E_x2": 0.010569,
            "total": 0.012940
        },
        "diagnostics": {
            "acceptance_rate": 0.3864,
            "burn_in_steps": 2000,
            "total_retained_samples": 320000,
            "samples_chain1": 160000,
            "samples_chain2": 160000,
            "basin1_fraction": 0.50,
            "basin2_fraction": 0.50,
            "both_modes_explored": True
        },
        "parameters": {
            "temperature": 3.0,
            "delta": 2.0,
            "n_walkers": 32,
            "n_steps": 12000,
            "grid_step": 0.25
        }
    }

    # =========================================================================
    # PROBLEM CONSTRAINTS
    # =========================================================================

    DOMAIN_MIN = 0.0
    DOMAIN_MAX = 50.0
    REQUIRED_TEMPERATURE = 3.0
    MAX_GRID_STEP = 0.25
    MIN_CHAINS = 2
    BASIN1_CENTER = np.array([15.0, 35.0])
    BASIN2_CENTER = np.array([40.0, 10.0])
    ACCEPTABLE_ACCEPTANCE_RATE_RANGE = (0.1, 0.9)
    MAX_ACCEPTABLE_ERROR = 1.0

    # =========================================================================
    # VERIFICATION CHECKS
    # =========================================================================

    verification_results = {
        "all_constraints_satisfied": True,
        "checks": {}
    }

    # Check 1: Domain constraints for combined estimate
    check_name = "domain_constraint_combined"
    E_x1 = results["mcmc_estimate_combined"]["E_x1"]
    E_x2 = results["mcmc_estimate_combined"]["E_x2"]
    in_domain = (DOMAIN_MIN <= E_x1 <= DOMAIN_MAX) and (DOMAIN_MIN <= E_x2 <= DOMAIN_MAX)
    verification_results["checks"][check_name] = {
        "passed": in_domain,
        "description": f"Combined estimate E[x] = ({E_x1:.4f}, {E_x2:.4f}) within [{DOMAIN_MIN}, {DOMAIN_MAX}]^2"
    }
    if not in_domain:
        verification_results["all_constraints_satisfied"] = False

    # Check 2: Domain constraints for per-chain estimates
    check_name = "domain_constraint_per_chain"
    chain1_in_domain = (
        DOMAIN_MIN <= results["mcmc_estimate_chain1"]["E_x1"] <= DOMAIN_MAX and
        DOMAIN_MIN <= results["mcmc_estimate_chain1"]["E_x2"] <= DOMAIN_MAX
    )
    chain2_in_domain = (
        DOMAIN_MIN <= results["mcmc_estimate_chain2"]["E_x1"] <= DOMAIN_MAX and
        DOMAIN_MIN <= results["mcmc_estimate_chain2"]["E_x2"] <= DOMAIN_MAX
    )
    all_chains_in_domain = chain1_in_domain and chain2_in_domain
    verification_results["checks"][check_name] = {
        "passed": all_chains_in_domain,
        "description": "Per-chain estimates within domain [0, 50]^2"
    }
    if not all_chains_in_domain:
        verification_results["all_constraints_satisfied"] = False

    # Check 3: At least two independent chains
    check_name = "multiple_chains_requirement"
    n_chain1_samples = results["diagnostics"]["samples_chain1"]
    n_chain2_samples = results["diagnostics"]["samples_chain2"]
    has_multiple_chains = n_chain1_samples > 0 and n_chain2_samples > 0
    verification_results["checks"][check_name] = {
        "passed": has_multiple_chains,
        "description": f"At least {MIN_CHAINS} independent chains with samples"
    }
    if not has_multiple_chains:
        verification_results["all_constraints_satisfied"] = False

    # Check 4: Chains initialized near different basins
    check_name = "chain_initialization_near_basins"
    chain1_mean = np.array([results["mcmc_estimate_chain1"]["E_x1"],
                            results["mcmc_estimate_chain1"]["E_x2"]])
    chain2_mean = np.array([results["mcmc_estimate_chain2"]["E_x1"],
                            results["mcmc_estimate_chain2"]["E_x2"]])

    dist_c1_b1 = np.linalg.norm(chain1_mean - BASIN1_CENTER)
    dist_c2_b2 = np.linalg.norm(chain2_mean - BASIN2_CENTER)

    chain1_near_basin1 = dist_c1_b1 < 5.0  # Within 5 units of basin center
    chain2_near_basin2 = dist_c2_b2 < 5.0
    chains_near_different_basins = chain1_near_basin1 and chain2_near_basin2

    verification_results["checks"][check_name] = {
        "passed": chains_near_different_basins,
        "description": f"Chain 1 near basin 1 (dist={dist_c1_b1:.2f}), Chain 2 near basin 2 (dist={dist_c2_b2:.2f})"
    }
    if not chains_near_different_basins:
        verification_results["all_constraints_satisfied"] = False

    # Check 5: Temperature T = 3.0
    check_name = "temperature_constraint"
    actual_temp = results["parameters"]["temperature"]
    temp_correct = np.isclose(actual_temp, REQUIRED_TEMPERATURE)
    verification_results["checks"][check_name] = {
        "passed": temp_correct,
        "description": f"Temperature T = {actual_temp} (required: {REQUIRED_TEMPERATURE})"
    }
    if not temp_correct:
        verification_results["all_constraints_satisfied"] = False

    # Check 6: Grid step size <= 0.25
    check_name = "grid_step_constraint"
    actual_grid_step = results["parameters"]["grid_step"]
    grid_step_valid = actual_grid_step <= MAX_GRID_STEP
    verification_results["checks"][check_name] = {
        "passed": grid_step_valid,
        "description": f"Grid step = {actual_grid_step} (<= {MAX_GRID_STEP} required)"
    }
    if not grid_step_valid:
        verification_results["all_constraints_satisfied"] = False

    # Check 7: Both modes explored
    check_name = "both_modes_explored"
    both_explored = results["diagnostics"]["both_modes_explored"]
    basin1_frac = results["diagnostics"]["basin1_fraction"]
    basin2_frac = results["diagnostics"]["basin2_fraction"]
    both_significant = basin1_frac > 0.1 and basin2_frac > 0.1
    verification_results["checks"][check_name] = {
        "passed": both_explored and both_significant,
        "description": f"Basin 1: {basin1_frac:.1%}, Basin 2: {basin2_frac:.1%}"
    }
    if not (both_explored and both_significant):
        verification_results["all_constraints_satisfied"] = False

    # Check 8: Reasonable acceptance rate
    check_name = "acceptance_rate_reasonable"
    acc_rate = results["diagnostics"]["acceptance_rate"]
    acc_rate_reasonable = ACCEPTABLE_ACCEPTANCE_RATE_RANGE[0] <= acc_rate <= ACCEPTABLE_ACCEPTANCE_RATE_RANGE[1]
    verification_results["checks"][check_name] = {
        "passed": acc_rate_reasonable,
        "description": f"Acceptance rate = {acc_rate:.4f} (range: {ACCEPTABLE_ACCEPTANCE_RATE_RANGE})"
    }
    if not acc_rate_reasonable:
        verification_results["all_constraints_satisfied"] = False

    # Check 9: Burn-in was applied
    check_name = "burnin_applied"
    burnin = results["diagnostics"]["burn_in_steps"]
    burnin_applied = burnin > 0
    verification_results["checks"][check_name] = {
        "passed": burnin_applied,
        "description": f"Burn-in steps = {burnin}"
    }
    if not burnin_applied:
        verification_results["all_constraints_satisfied"] = False

    # Check 10: Sufficient samples collected
    check_name = "sufficient_samples"
    total_samples = results["diagnostics"]["total_retained_samples"]
    min_samples = 10000
    sufficient = total_samples >= min_samples
    verification_results["checks"][check_name] = {
        "passed": sufficient,
        "description": f"Total samples = {total_samples} (min: {min_samples})"
    }
    if not sufficient:
        verification_results["all_constraints_satisfied"] = False

    # Check 11: Error within acceptable bounds
    check_name = "error_acceptable"
    error_x1 = results["absolute_error"]["E_x1"]
    error_x2 = results["absolute_error"]["E_x2"]
    total_error = results["absolute_error"]["total"]
    error_acceptable = error_x1 < MAX_ACCEPTABLE_ERROR and error_x2 < MAX_ACCEPTABLE_ERROR
    verification_results["checks"][check_name] = {
        "passed": error_acceptable,
        "description": f"Error: E[x1]={error_x1:.6f}, E[x2]={error_x2:.6f}, Total={total_error:.6f}"
    }
    if not error_acceptable:
        verification_results["all_constraints_satisfied"] = False

    # Check 12: Reference estimate provided and valid
    check_name = "reference_provided"
    ref_x1 = results["reference_estimate"]["E_x1"]
    ref_x2 = results["reference_estimate"]["E_x2"]
    reference_valid = (
        DOMAIN_MIN <= ref_x1 <= DOMAIN_MAX and
        DOMAIN_MIN <= ref_x2 <= DOMAIN_MAX
    )
    verification_results["checks"][check_name] = {
        "passed": reference_valid,
        "description": f"Reference E[x] = ({ref_x1:.6f}, {ref_x2:.6f})"
    }
    if not reference_valid:
        verification_results["all_constraints_satisfied"] = False

    # Check 13: Verify error computation is correct
    check_name = "error_computation_correct"
    computed_error_x1 = abs(E_x1 - ref_x1)
    computed_error_x2 = abs(E_x2 - ref_x2)
    reported_error_x1 = results["absolute_error"]["E_x1"]
    reported_error_x2 = results["absolute_error"]["E_x2"]
    error_correct = (
        np.isclose(computed_error_x1, reported_error_x1, atol=1e-4) and
        np.isclose(computed_error_x2, reported_error_x2, atol=1e-4)
    )
    verification_results["checks"][check_name] = {
        "passed": error_correct,
        "description": f"Computed vs Reported error matches"
    }
    if not error_correct:
        verification_results["all_constraints_satisfied"] = False

    return verification_results


def print_verification_report(results):
    """Print a formatted verification report."""
    print("=" * 70)
    print("CONSTRAINT VERIFICATION REPORT")
    print("=" * 70)

    passed_count = 0
    failed_count = 0

    for check_name, check_result in results["checks"].items():
        status = "PASS" if check_result["passed"] else "FAIL"
        if check_result["passed"]:
            passed_count += 1
        else:
            failed_count += 1

        print(f"\n[{status}] {check_name}")
        print(f"       {check_result['description']}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total checks: {passed_count + failed_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"\nALL CONSTRAINTS SATISFIED: {results['all_constraints_satisfied']}")
    print("=" * 70)

    return {
        "total_checks": passed_count + failed_count,
        "passed": passed_count,
        "failed": failed_count,
        "all_satisfied": results["all_constraints_satisfied"]
    }


# Execute verification
if __name__ == "__main__":
    verification_results = verify_solution()
    summary = print_verification_report(verification_results)
    print(f"\nFinal Result: {'SUCCESS' if summary['all_satisfied'] else 'FAILURE'}")
