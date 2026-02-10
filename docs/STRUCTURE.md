# Project Structure

## Overview

This directory has been reorganized into a standard Python project structure for better maintainability and clarity.

## Migration from Old Structure

### Old Structure
```
sandbox_experiments/
├── *.py (all scripts and modules mixed)
├── experiments/
│   ├── config_*.yaml
│   ├── figures/
│   ├── noise_figures/
│   └── results/
└── input_data/
```

### New Structure
```
sandbox_experiments/
├── configs/          # Configuration files
├── data/             # Input data
├── docs/             # Documentation (this file)
├── examples/         # Example usage
├── results/          # All outputs
├── scripts/          # Executable scripts
├── src/              # Core modules
└── tests/            # Unit tests
```

## Key Changes

1. **Configuration files** moved from `experiments/` to `configs/`
2. **Input data** renamed from `input_data/` to `data/`
3. **Core modules** (`optimizers.py`, `data_handling.py`, `Gaussian_fit.py`) moved to `src/`
4. **Executable scripts** moved to `scripts/`
5. **All results** consolidated in `results/` directory
6. **Import paths** updated in all scripts to use relative imports from `src/`

## Import Updates

All scripts now use relative imports:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from optimizers import AlternatingOptimisationFitter
from data_handling import load_hyperfines_and_susceptibility_tensor
```

Data paths are now relative to module location:

```python
from pathlib import Path
DATA_DIR = Path(__file__).parent.parent / 'data'
```

## Running Scripts

All scripts should be run from the `sandbox_experiments` directory:

```bash
cd sandbox_experiments
python scripts/run_experiments.py
```

Configuration files use relative paths from the project root:

```python
config = load_config("configs/config_default.yaml")
```
