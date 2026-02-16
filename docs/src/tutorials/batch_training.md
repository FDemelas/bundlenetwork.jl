# Batch Training Tutorial

This tutorial covers batch training mode, where multiple instances are processed 
together for more stable gradient estimates.

## When to Use Batch Training

Use batch training when:
- You have sufficient memory (GPU/CPU)
- Instances are similar in size and structure
- You want more stable gradients
- Training time is not critical

## Basic Batch Training

### Minimal Example
```bash
julia runs/train_batch.jl \
  --data ./data/MCNDforTest/ \
  --lr 0.001 \
  --mti 100 \
  --mvi 20 \
  --seed 42 \
  --maxItBack 50 \
  --maxEP 100 \
  --batch_size 4
```

## Understanding Batch Size

**Batch Size 1** (Default):
- Processes one instance at a time
- Lower memory usage
- Higher gradient variance
- Faster iterations

**Batch Size 4-8**:
- Processes multiple instances together
- More stable gradients
- Higher memory usage
- Slower iterations but better convergence

**Choosing Batch Size**:
```julia
# Memory-constrained
--batch_size 1

# Balanced
--batch_size 4

# High-memory system
--batch_size 8
```

## Advanced Configuration

### Curriculum Learning

Gradually increase difficulty by starting with fewer iterations:
```bash
julia runs/train_batch.jl \
  --incremental true \
  --maxIt 100 \
  --maxEP 100 \
  ...
```

**How it works**:
- Epochs 1-50: Linearly increase from 2 iterations to 100
- Epochs 51-100: Use full 100 iterations

**Benefits**:
- Easier optimization early in training
- Helps avoid local minima
- Can improve final performance

### Loss Functions

#### Standard Loss
```bash
--lambda 0.0 --gamma 0.0
```
Loss = -ϕ(x_final)

#### Weighted Loss
```bash
--lambda 0.5
```
Loss = -[0.5 * ϕ(x_final) + 0.5 * ϕ(x_stabilization)]

#### Telescopic Loss
```bash
--gamma 0.1
```
Loss = -Σ γ^i * ϕ(x_i)

**Recommendation**: Start with standard loss, add telescopic if underfitting.

### Architecture Options

#### Hidden Size
```bash
--h_representation 64   # Default, good balance
--h_representation 32   # Faster, less capacity
--h_representation 128  # Slower, more capacity
```

#### Activation Functions
```bash
--h_act softplus  # Smooth, default
--h_act relu      # Sparse, faster
--h_act tanh      # Bounded
--h_act gelu      # Smooth, modern
```

#### Sampling Strategies
```bash
# Sample proximity parameter
--sampling_t true

# Sample in latent space for attention
--sampling_gamma true
```

### Optimization Settings

#### Learning Rate Schedule
```bash
--lr 0.001          # Initial learning rate
--decay 0.9         # Decay factor
--scheduling_ss 100 # Apply decay every 100 epochs
```

#### Gradient Clipping
```bash
--cn 5   # Clip gradient norm to 5 (default)
--cn 10  # More lenient clipping
```

## Complete Example
```bash
julia runs/train_batch.jl \
  --data ./data/MCNDforTest/ \
  --lr 0.001 \
  --decay 0.95 \
  --cn 5 \
  --mti 200 \
  --mvi 40 \
  --seed 42 \
  --maxItBack 50 \
  --maxIt 100 \
  --maxItVal 200 \
  --maxEP 150 \
  --batch_size 4 \
  --incremental true \
  --h_representation 64 \
  --h_act softplus \
  --use_softmax true \
  --sampling_t true \
  --sampling_gamma false \
  --gamma 0.05 \
  --lambda 0.0 \
  --scheduling_ss 50
```

## Monitoring Training

### Console Output
```
Epoch 1 Training - lsp: 1234.56  gap: 15.2%
Epoch 1 Validation - lsp: 1250.30  gap: 12.8%
```

### TensorBoard
```bash
tensorboard --logdir resLogs/
```

**Key Plots**:
- `Train/GAP_percentage`: Training gap over time
- `Validation/GAP_percentage`: Validation gap
- `Train/Loss_value`: Training loss

### Interpreting Results

**Good Training**:
- Training gap decreases steadily
- Validation gap tracks training gap
- Loss decreases (becomes more negative)

**Overfitting**:
- Training gap << Validation gap
- Validation gap stops improving

**Underfitting**:
- Both gaps remain high
- Loss plateaus early

**Solutions**:
- Overfitting: Reduce model size, add regularization (gamma)
- Underfitting: Increase model size, train longer

## Output Files

After training, find in `resLogs/<experiment_name>/`:

- `nn.bson`: Final model
- `nn_best.bson`: Best validation model
- `loss.json`: Loss per epoch
- `gaps.json`: Training gaps
- `gaps_val.json`: Validation gaps
- `dataset.json`: Train/val split

## Next Steps

- Try [Episodic Training](@ref) for comparison
- Learn about [Testing](@ref) your model
- Explore [Hyperparameter Tuning](@ref)
