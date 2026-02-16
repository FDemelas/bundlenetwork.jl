# Examples

## Complete Workflow Example

This example shows a complete training-to-testing workflow.

### Step 1: Prepare Data
```bash
mkdir -p data/my_problem
# Add your .dat or .json files
```

### Step 2: Train
```bash
julia runs/train_batch.jl \
  --data ./data/my_problem/ \
  --lr 0.001 \
  --mti 100 \
  --mvi 20 \
  --seed 42 \
  --maxItBack 50 \
  --maxEP 100 \
  --batch_size 4 \
  --incremental true
```

### Step 3: Monitor
```bash
tensorboard --logdir resLogs/
```

### Step 4: Test
```bash
julia runs/test.jl \
  --data ./data/my_problem/ \
  --model ./resLogs/BatchVersion_.../ \
  --dataset ./resLogs/BatchVersion_.../
```

### Step 5: Analyze
```julia
using JSON, Statistics, Plots

# Load results
results = JSON.parsefile("res_test2_my_problem.json")

# Plot convergence
gaps = [data["gaps"] for (_, data) in results]
plot(gaps, alpha=0.3, xlabel="Iteration", ylabel="Gap (%)", 
     yscale=:log10, legend=false, title="Test Set Convergence")

# Compute statistics
final_gaps = [g[end] for g in gaps]
println("Mean: $(mean(final_gaps))%")
println("Median: $(median(final_gaps))%")
```

## More Examples

See individual tutorial pages for detailed examples:
- [Batch Training Examples](@ref)
- [Episodic Training Examples](@ref)
- [Testing Examples](@ref)
