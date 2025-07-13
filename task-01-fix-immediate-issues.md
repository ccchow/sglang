# Task 1: Fix Immediate Issues in MeZO Trainer

## Status: In Progress

## Description
Fix basic implementation issues in the MeZO trainer to ensure it runs without errors.

## Issues Identified

1. **Missing import statement**
   - Line 132: `math.sqrt(5)` is used but `math` module is not imported
   - Need to add `import math` at the top of the file

2. **Incorrect MeZO implementation**
   - Current implementation uses random perturbations for each parameter independently
   - MeZO paper specifies using a fixed perturbation direction `z` sampled once per step
   - Need to fix the perturbation logic to match the MeZO algorithm

3. **Loss calculation issues**
   - Current implementation only uses the first token's logits for loss calculation
   - Should compute cross-entropy loss over the full sequence

## Implementation Plan

1. Add missing `import math` statement
2. Fix MeZO algorithm implementation:
   - Sample a single fixed direction `z` at the beginning of each step
   - Apply symmetric perturbations `+εz` and `-εz` to all parameters
   - Update gradient estimation logic
3. Improve loss calculation to handle full sequences

## Code Changes

### Fixed mezo_trainer.py
The following changes were made:
- Added `import math` statement
- Implemented correct MeZO algorithm with fixed perturbation direction
- Fixed loss calculation to handle full sequences (placeholder for now)

## Testing
- Unit tests to be added in subsequent task
- Basic functionality tests can be run with example dataset

## Next Steps
- Proceed to Task 2: Refine initialization with proper ServerArgs parsing