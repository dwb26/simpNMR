# Sandbox Experiments

Experimental code for paramagnetic NMR assignment and susceptibility fitting using alternating optimization methods.

## Directory Structure

```
sandbox_experiments/
├── configs/          # Configuration files (YAML)
├── data/             # Input data files (CSV, Excel)
├── docs/             # Documentation
├── examples/         # Example scripts and notebooks
├── results/          # Experiment results, figures, and animations
├── scripts/          # Executable scripts for running experiments
├── src/              # Core source code modules
└── tests/            # Unit tests
```

### `configs/`
Configuration files for different experiments:
- `config_default.yaml` - Default experiment configuration
- `config_noise_sensitivity.yaml` - Noise sensitivity experiment
- `config_quick_noise_test.yaml` - Quick test configuration

### `data/`
Input data files:
- `hyperfines_and_shifts_298.00_K.csv` - Hyperfine tensors and observed shifts
- `susceptibility_tensor.csv/xlsx` - Ground truth susceptibility tensor

### `scripts/`
Executable scripts:
- `run_experiments.py` - Main experiment runner
- `run_noise_experiment.py` - Noise sensitivity experiments
- `analyze_results.py` - Results analysis and visualization
- `analyze_noise_results.py` - Noise experiment analysis
- `test_animation.py` - Test animation functionality
- `depreciated_main.py` - Legacy code (deprecated)

### `src/`
Core modules:
- `optimizers.py` - Optimization algorithms (alternating optimization, moment matching)
- `data_handling.py` - Data loading and preprocessing
- `Gaussian_fit.py` - Gaussian fitting utilities

### `results/`
Experiment outputs:
- `figures/` - Generated plots and visualizations
- `noise_figures/` - Noise sensitivity plots
- `experiments/` - Pickled experiment results
- `*.mp4` - Animation videos

## Usage

### Running Experiments

From the `sandbox_experiments` directory:

```bash
# Run default experiment
python scripts/run_experiments.py

# Run noise sensitivity experiment
python scripts/run_noise_experiment.py --config configs/config_noise_sensitivity.yaml

# Analyze results
python scripts/analyze_results.py results/experiments/results_*.pkl
```

### Configuration

Edit YAML files in `configs/` to customize experiments:
- Number of trials
- Optimization parameters (max iterations, tolerance, optimizer)
- Initial conditions
- Random seeds

## Requirements

See `requirements.txt` for Python dependencies.
