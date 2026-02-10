"""
Quick reference for running fair comparison experiments.
"""

# =============================================================================
# QUICK START
# =============================================================================

# Run full comparison (both methods, 50 trials each):
python scripts/compare_methods.py

# Run individual experiments:
python scripts/run_experiments.py --config alternating_optimization_trials.yaml
python scripts/run_experiments.py --config moment_matching_trials.yaml

# Analyze saved results:
python scripts/analyze_results.py results/experiments/alternating_optimization_TIMESTAMP.pkl
python scripts/analyze_results.py results/experiments/moment_matching_TIMESTAMP.pkl


# =============================================================================
# WHAT'S NEW?
# =============================================================================

# 1. Random Chi Tensor Generation
#    - MomentMatchingFitter now uses random chi initialization for each trial
#    - Satisfies physical constraints (traceless, axiality, rhombicity)
#    - Provides fair comparison with AlternatingOptimization's random assignment

# 2. Removed Deterministic Override
#    - moment_matching experiments can now run multiple trials
#    - Each trial uses different random chi initialization

# 3. New Comparison Script
#    - Automatically runs both experiments
#    - Generates comparative plots and statistics
#    - Performs Mann-Whitney U tests


# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Both configs use:
#   - 50 trials (n_trials: 50)
#   - Sequential random seeds (0, 1, 2, ..., 49)
#   - trust-constr optimizer
#   - Tolerance: 1e-8

# Differences:
#   - Alternating: max_iters=50, random assignment initialization
#   - Moment Matching: max_iters=100, random chi tensor initialization


# =============================================================================
# OUTPUT STRUCTURE
# =============================================================================

# results/experiments/
#   ├── alternating_optimization_TIMESTAMP.pkl
#   └── moment_matching_TIMESTAMP.pkl

# results/comparison/
#   ├── comparison_loss_distribution.png       # Box + histogram of final loss
#   ├── comparison_chi_error_distribution.png  # Box + histogram of chi error
#   ├── comparison_iterations.png              # Convergence speed
#   ├── comparison_success_rate.png            # Constraint satisfaction
#   └── comparison_summary.csv                 # Summary statistics table


# =============================================================================
# EXPECTED RUNTIME
# =============================================================================

# Single experiment (50 trials):
#   - Alternating Optimization: ~5-10 minutes (depending on system)
#   - Moment Matching: ~10-20 minutes (more iterations per trial)

# Full comparison: ~15-30 minutes


# =============================================================================
# INTERPRETING RESULTS
# =============================================================================

# Key metrics to examine:
#   1. Best Loss: Minimum achieved across all trials
#   2. Median Loss: Typical performance
#   3. Success Rate: % of trials satisfying constraints
#   4. Convergence Rate: % of trials that converged
#   5. Mean Iterations: Computational cost per trial

# Statistical significance:
#   - Mann-Whitney U test p-value < 0.05 indicates significant difference
#   - Compare distributions, not just best results


# =============================================================================
# TROUBLESHOOTING
# =============================================================================

# If you get import errors:
#   - Ensure you're running from sandbox_experiments directory
#   - Check that src/ directory exists with optimizers.py

# If experiments fail:
#   - Check configs/ directory has the YAML files
#   - Verify data/ directory has required CSV files
#   - Try running with verbose: true in config

# If comparison script fails:
#   - Ensure both experiments completed successfully
#   - Check that results files exist in results/experiments/
#   - Verify scipy is installed (for statistical tests)
