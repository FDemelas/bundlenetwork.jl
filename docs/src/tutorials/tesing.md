# Testing & Evaluation Tutorial

Learn how to evaluate trained models on test instances.

## Basic Testing
```bash
julia runs/test.jl \
  --data ./data/MCNDforTest/ \
  --model ./resLogs/BatchVersion_bs_1_seed42_.../ \
  --dataset ./resLogs/BatchVersion_bs_1_seed42_.../
```

**Arguments**:
- `--data`: Path to instance folder
- `--model`: Path to folder containing trained model (nn_bestLV.bson or nn_best.bson)
- `--dataset`: Path to folder containing dataset.json

## What the Test Script Does

1. **Load Model**: Reads trained neural network from BSON file
2. **Load Test Split**: Reads test instance list from dataset.json
3. **Load Instances**: Processes each test instance
4. **Run Bundle Method**: Solves with NN guidance for 100 iterations (default)
5. **Compute Metrics**: Tracks objectives, times, and gaps
6. **Save Results**: Outputs to JSON file

## Test Configuration

The test script uses these default settings:
- **Iterations**: 100 (can be changed via `maxIT` parameter in code)
- **Proximity parameter**: 0.000001
- **Device**: CPU only
- **Exact gradients**: Enabled
- **Instance features**: Enabled

## Understanding Test Results

### Output File Structure

File: `res_test2_<dataset_name>.json`
```json
{
  "instance1.dat": {
    "time": 1.234,
    "objs": [0.0, 100.5, 150.3, ..., 245.8],
    "times": [0.01, 0.02, 0.03, ..., 1.23],
    "gaps": [100.0, 15.2, 5.3, ..., 1.2]
  },
  "instance2.dat": { ... }
}
```

**Fields**:
- `time`: Total solving time (seconds)
- `objs`: Objective value at each iteration
- `times`: Cumulative time at each iteration
- `gaps`: Optimality gap at each iteration (%)

### Analyzing Results
```julia
using JSON
using Statistics

# Load results
results = JSON.parsefile("res_test2_MCNDforTest.json")

# Compute statistics across all instances
final_gaps = [data["gaps"][end] for (inst, data) in results]
final_times = [data["time"] for (inst, data) in results]
final_objs = [data["objs"][end] for (inst, data) in results]

println("Mean final gap: $(mean(final_gaps))%")
println("Median final gap: $(median(final_gaps))%")
println("Mean solving time: $(mean(final_times))s")

# Find best/worst instances
sorted_gaps = sort(collect(results), by=x->x[2]["gaps"][end])
println("Best instance: $(sorted_gaps[1][1])")
println("Worst instance: $(sorted_gaps[end][1])")
```

## Visualization

### Plot Convergence
```julia
using Plots

# Load results
results = JSON.parsefile("res_test2_MCNDforTest.json")

# Plot single instance
instance_name = "instance1.dat"
data = results[instance_name]

plot(data["objs"], 
     xlabel="Iteration", 
     ylabel="Objective Value",
     title="Convergence: $instance_name",
     legend=false)
```

### Plot Gaps Over Time
```julia
plot(data["gaps"],
     xlabel="Iteration",
     ylabel="Optimality Gap (%)",
     title="Gap Convergence",
     yscale=:log10,
     legend=false)
```

### Compare Multiple Instances
```julia
p = plot(xlabel="Iteration", ylabel="Gap (%)", 
         title="Gap Convergence", yscale=:log10)

for (inst, data) in results
    plot!(p, data["gaps"], alpha=0.3, label=inst)
end

plot!(p)
```

## Comparing Models

### Test Multiple Models
```bash
# Test model 1
julia test.jl --data ./data/ --model ./resLogs/model1/ --dataset ./resLogs/model1/

# Test model 2
julia test.jl --data ./data/ --model ./resLogs/model2/ --dataset ./resLogs/model2/
```

### Compare Results
```julia
using JSON, Statistics

# Load both results
res1 = JSON.parsefile("res_test2_model1.json")
res2 = JSON.parsefile("res_test2_model2.json")

# Compare final gaps
gaps1 = [data["gaps"][end] for (_, data) in res1]
gaps2 = [data["gaps"][end] for (_, data) in res2]

println("Model 1 mean gap: $(mean(gaps1))%")
println("Model 2 mean gap: $(mean(gaps2))%")
println("Improvement: $(mean(gaps1) - mean(gaps2))%")

# Statistical test
using HypothesisTests
t_test = OneSampleTTest(gaps1 .- gaps2)
println(t_test)
```

## Advanced: Custom Testing

### Modify Test Parameters

Edit `test.jl` to change:
```julia
# Change number of iterations
maxIT = 200  # Instead of default 100

# Change proximity parameter
t = 0.00001  # Instead of default 0.000001

# Disable instance features
instance_features = false
```

### Test on Custom Dataset
```julia
# Create custom test set
custom_test = ["instance1.dat", "instance5.dat", "instance10.dat"]

# Modify test.jl or create dataset.json:
dataset = Dict("test" => custom_test)
```

## Performance Benchmarking

### Against Baseline

Compare against standard bundle method:
```julia
# Test with NN guidance
julia test.jl --data ./data/ --model ./resLogs/trained_model/ ...

# Test with constant t (baseline)
# Modify test.jl to use constant t_strat instead of nn_t_strategy()
```

### Metrics to Report

- **Final Gap**: Optimality at termination
- **Time to 5% Gap**: Iterations needed
- **Total Time**: Computational cost
- **Success Rate**: % instances below target gap

## Troubleshooting

### Model Not Found
```
ERROR: SystemError: opening file "nn_bestLV.bson": No such file
```

**Solution**: Check model path and ensure `nn_bestLV.bson` exists.

### Gold Solutions Missing

If using .dat format, ensure:
```bash
./golds/<dataset_name>/gold.json
```
exists with format:
```json
{
  "instance1.dat": optimal_value,
  ...
}
```

### Out of Memory

Reduce batch size in test (currently always 1) or test fewer instances.

## Next Steps

- Return to [Batch Training](@ref) or [Episodic Training](@ref)
- See [Troubleshooting](@ref) for common issues
- Explore [API Reference](@ref) for function details
