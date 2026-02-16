# Quick Start Guide

This guide will help you train and test your first model in under 5 minutes.

## Step 1: Prepare Your Data

Ensure you have:
- Problem instances in `./data/MCNDforTest/`
- (Optional) Gold solutions in `./golds/MCNDforTest/gold.json`

## Step 2: Train a Model

### Minimal Training Command
```bash
julia runs/train_batch.jl \
  --data ./data/MCNDforTest/ \
  --lr 0.001 \
  --mti 50 \
  --mvi 10 \
  --seed 42 \
  --maxItBack 30 \
  --maxEP 20
```

**Parameters Explained:**
- `--lr 0.001`: Learning rate
- `--mti 50`: Use 50 training instances
- `--mvi 10`: Use 10 validation instances
- `--seed 42`: Random seed for reproducibility
- `--maxItBack 30`: Unroll 30 iterations for gradient computation
- `--maxEP 20`: Train for 20 epochs

### What Happens During Training

1. **Data Loading**: Instances are loaded and rescaled
2. **Model Initialization**: Neural network is created
3. **Training Loop**: For each epoch:
   - Shuffle training data
   - Process batches
   - Compute gradients via unrolling
   - Update model parameters
   - Validate on validation set
4. **Saving**: Best model and metrics are saved

### Expected Output
```
Epoch 1 Training - lsp: 1234.56  gap: 15.2%
Epoch 1 Validation - lsp: 1250.30  gap: 12.8%
Epoch 2 Training - lsp: 1450.23  gap: 8.5%
Epoch 2 Validation - lsp: 1480.15  gap: 7.2%
...
```

Results are saved to: `resLogs/BatchVersion_bs_1_seed42_.../`

## Step 3: Monitor Training

### Using TensorBoard
```bash
tensorboard --logdir resLogs/
```

Open browser to: `http://localhost:6006`

### Key Metrics to Watch

- **GAP_percentage**: Optimality gap (lower is better)
- **LSP_value**: Objective value (higher is better)
- **Loss_value**: Training loss

## Step 4: Test Your Model
```bash
julia runs/test.jl \
  --data ./data/MCNDforTest/ \
  --model ./resLogs/BatchVersion_bs_1_seed42_.../ \
  --dataset ./resLogs/BatchVersion_bs_1_seed42_.../
```

### Test Results

Results saved to: `res_test2_MCNDforTest.json`
```json
{
  "instance1.dat": {
    "time": 1.234,
    "objs": [0.0, 100.5, 150.3, 180.2, ...],
    "gaps": [100.0, 15.2, 5.3, 2.1, ...]
  }
}
```

## Step 5: Analyze Results
```julia
using JSON

# Load results
results = JSON.parsefile("res_test2_MCNDforTest.json")

# Compute statistics
for (instance, data) in results
    final_gap = data["gaps"][end]
    println("$instance: Final gap = $(round(final_gap, digits=2))%")
end
```

## Next Steps

### Improve Performance

1. **Increase Training Data**: Use more instances (`--mti 200`)
2. **Train Longer**: More epochs (`--maxEP 100`)
3. **Tune Learning Rate**: Try different values (`--lr 0.0001`)
4. **Increase Model Capacity**: Larger hidden size (`--h_representation 128`)

### Advanced Features

- [Batch Training Tutorial](@ref): Learn batch processing
- [Episodic Training Tutorial](@ref): Instance-by-instance learning
- [Hyperparameter Tuning](@ref): Optimize performance

### Troubleshooting

See [Troubleshooting](@ref) if you encounter issues.
