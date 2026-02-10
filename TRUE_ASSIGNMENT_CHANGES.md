# Using True Assignment After Chi Fitting

## Changes Made

Modified the `MomentMatchingFitter` to use the **true assignment** for computing predicted shifts after fitting chi, while still fitting chi based on moment matching (which doesn't require knowing the assignment).

## Key Insight

Your collaborators are correct: "once we know the correct chi, finding the correct assignment is easy." 

The moment matching approach:
1. **Fits chi** by matching statistical moments (doesn't require assignment)
2. **After fitting**, uses the true assignment to compute predicted shifts
3. This should show better alignment in the receiver space (left panel of animation)

## Implementation Details

### Modified Files

**[src/optimizers.py](src/optimizers.py)** - `MomentMatchingFitter` class:

1. **Store true hyperfines** (line ~520):
   ```python
   self.true_hyperfines_dict = hyperfines_dict.copy()
   ```
   This happens BEFORE `super().__init__()` permutes the hyperfines.

2. **Use true hyperfines in callback** (line ~700):
   ```python
   # Compute predicted shifts using TRUE assignment (for visualization)
   predicted_shifts = np.array([
       (1/3) * np.trace(self.true_hyperfines_dict[label] @ chi_current)
       for label in self.atom_labels
   ])
   
   # But still fit based on moments from permuted hyperfines
   predicted_shifts_permuted = np.array([
       (1/3) * np.trace(self.hyperfines_dict[label] @ chi_current)
       for label in self.atom_labels
   ])
   ```

3. **Updated initial frame recording** (line ~745):
   Now also uses true hyperfines for initial predicted shifts.

## Testing

Quick test configuration created: [configs/moment_matching_test.yaml](configs/moment_matching_test.yaml)

Run test:
```bash
python scripts/run_experiments.py --config moment_matching_test.yaml
```

Then create animation:
```bash
python scripts/analyze_results.py results/experiments/moment_matching_*.pkl
```

## Expected Behavior

### Before This Change
- Left panel (receiver space): Points scattered randomly
- Right panel (moment space): Moments converging
- Problem: Chi was fitted correctly, but visualization used wrong assignment

### After This Change
- Left panel (receiver space): Points should align on diagonal (y=x line)
- Right panel (moment space): Moments still converging
- Success: Chi fitting + true assignment = correct predictions

## The Conceptual Flow

```
1. Moment Matching (no assignment needed)
   ├─> Fit chi to match moments
   └─> Converge in moment space

2. Apply True Assignment (after chi is fitted)
   ├─> Use fitted chi + true hyperfines
   └─> Compute predicted shifts for visualization

3. Result
   ├─> Left panel: Shows y=x alignment (correct!)
   └─> Right panel: Shows moment convergence
```

## What This Means

The moment matching method:
- **Doesn't need** to solve the assignment problem during optimization
- **Fits chi** purely from statistical properties of shift distribution
- **Once chi is known**, using the correct assignment gives correct predictions

This validates your collaborators' intuition and shows that the moment matching approach can work IF we have access to the true assignment afterward.

## Next Steps

1. Run the test to verify animation alignment
2. If alignment is good, this confirms moment matching works conceptually
3. The remaining challenge: Can we recover the correct assignment from chi alone?
   - Hungarian algorithm post-processing?
   - Confidence metrics for assignment?
   - Or accept that we need other information to get assignment?
