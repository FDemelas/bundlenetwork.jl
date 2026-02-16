# Episodic Training Tutorial

Episodic training processes one instance at a time, updating the model after each instance.

## When to Use Episodic Training

Use episodic training when:
- Instances vary greatly in size
- Memory is limited
- You want online learning behavior
- Testing instance-specific adaptation

## Basic Episodic Training
```bash
julia runs/train_episodic.jl \
  --data ./data/MCNDforTest/ \
  --lr 0.001 \
  --mti 100 \
  --mvi 20 \
  --seed 42 \
  --maxIT 50 \
  --maxEP 100
```

## Key Differences from Batch Training

| Feature | Batch Training | Episodic Training |
|---------|---------------|-------------------|
| Update frequency | Per batch | Per instance |
| Gradient stability | Higher | Lower |
| Memory usage | Higher | Lower |
| Training speed | Slower per epoch | Faster per epoch |
| Validation | maxItVal iterations | 5×maxIT iterations |

## Configuration Options

### Initialization Strategy

#### Zero Initialization (Default)
```bash
--cr_init false
```
Start dual variables at zero.

#### Cutting-Plane Relaxation
```bash
--cr_init true
```
Warm-start from CR solution (slower initialization, may improve convergence).

### Loss Functions

#### Standard Loss (Default)
```bash
--telescopic false
```
Loss based on final point only.

#### Telescopic Loss
```bash
--telescopic true \
--gamma 0.1
```
Loss includes all visited points: L = -Σ γ^i * ϕ(x_i)

### Feature Engineering

#### Instance Features
```bash
--instance_features true  # Include linear relaxation features
--instance_features false # Use only bundle state
```

**Instance features** include:
- Linear relaxation dual variables
- Constraint structure information
- Problem-specific characteristics

### Proximity Parameter Strategy

#### Learned Parameter (Default)
```bash
--single_prediction false
```
Neural network predicts t at each iteration.

#### Constant Parameter
```bash
--single_prediction true
```
Use fixed proximity parameter throughout.

## Complete Example
```bash
julia runs/train_episodic.jl \
  --data ./data/MCNDforTest/ \
  --lr 0.001 \
  --cn 5 \
  --mti 200 \
  --mvi 40 \
  --seed 42 \
  --maxIT 50 \
  --maxEP 100 \
  --cr_init false \
  --exactGrad true \
  --telescopic true \
  --gamma 0.1 \
  --instance_features true \
  --single_prediction false \
  --sample_inside true
```

## Validation Behavior

Episodic training validates at **three iteration counts**:

- **maxIT**: Same as training
- **2×maxIT**: Medium-length run
- **5×maxIT**: Extended run

This helps assess:
- Short-term performance
- Convergence stability
- Long-term behavior

## Monitoring

### TensorBoard Metrics
```bash
tensorboard --logdir resLogs/
```

**Unique to episodic training**:
- `Validation/GAP_percentage_li`: Gap at maxIT
- `Validation x2/GAP_percentage`: Gap at 2×maxIT
- `Validation x5/GAP_percentage`: Gap at 5×maxIT

### Comparing Validation Lengths

Good convergence pattern:
```
maxIT gap:   5.2%
2×maxIT gap: 3.1%
5×maxIT gap: 2.0%
```

Poor convergence:
```
maxIT gap:   5.2%
2×maxIT gap: 5.0%
5×maxIT gap: 4.9%
```

## Output Files

Similar to batch training, plus:
- Validation metrics at multiple iteration counts

## Batch vs. Episodic: When to Choose

Choose **Batch Training** if:
- ✓ Instances are similar in size
- ✓ You have sufficient memory
- ✓ You want stable gradients
- ✓ Training time is not critical

Choose **Episodic Training** if:
- ✓ Instances vary greatly
- ✓ Memory is limited
- ✓ You want faster epoch times
- ✓ Testing online learning

## Next Steps

- Compare with [Batch Training](@ref)
- Learn [Testing & Evaluation](@ref)
- See [Hyperparameter Tuning](@ref)
