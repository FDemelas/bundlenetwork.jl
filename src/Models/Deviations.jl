"""
    AbstractDeviation

Abstract type for deviation strategies used in models that predict only `t`.

# Purpose
Deviation strategies define how a model's predicted value should be applied
to update the temporal parameter t. Different strategies enable different
update behaviors, from simple replacement to additive or multiplicative updates.

# Type Hierarchy
```
AbstractDeviation
├── AdditiveDeviation       # t_new = t_old + dev
├── MultiplicativeDeviation # t_new = t_old × dev
└── NothingDeviation        # t_new = dev
```

# Design Rationale
This abstraction allows models to be agnostic about how their predictions
are applied. The same model architecture can be used with different update
strategies by simply changing the deviation type.


# When to Use Each Deviation Type
- **AdditiveDeviation**: For absolute adjustments to t
- **MultiplicativeDeviation**: For relative/proportional adjustments
- **NothingDeviation**: When the model predicts t directly

# See Also
- `RnnTModel`: Uses deviations for t prediction
- `RnnTModelSampleInside`: Uses deviations with transformed samples
"""
abstract type AbstractDeviation
end

"""
    AdditiveDeviation <: AbstractDeviation

Additive deviation strategy: t_new = t_old + dev

# Formula
```
t_new = t_old + dev
```

where:
- `t_old`: Current temporal parameter value
- `dev`: Deviation predicted by the model
- `t_new`: Updated temporal parameter value

# Characteristics
- **Type**: Absolute adjustment
- **Effect**: Shifts t by a constant amount
- **Sign**: Positive dev increases t, negative dev decreases t
- **Scale**: Independent of current t value

# Use Cases
Prefer AdditiveDeviation when:
- You want absolute step size control
- The magnitude of change should be independent of current t
- You're implementing gradient-descent-like updates
- The model learns optimal step sizes

# Mathematical Properties
- Linear: deviation(t, α·dev) = t + α·dev
- Commutative: deviation(deviation(t, dev1), dev2) = deviation(deviation(t, dev2), dev1)
- Identity: deviation(t, 0) = t
- Inverse: deviation(deviation(t, dev), -dev) = t

# Typical Training Behavior
During training, the model typically learns to predict:
- Small deviations (|dev| < 1) for fine-tuning
- Larger deviations (|dev| > 1) for aggressive exploration
- Near-zero deviations when t is already optimal

# Implementation Note
This struct is declared as a Flux layer (@layer) to ensure proper handling
during model serialization and GPU transfer, even though it has no parameters.
"""
struct AdditiveDeviation <: AbstractDeviation end

# Callable implementation: t_new = t_old + dev
(dt::AdditiveDeviation)(t, dev) = t .+ dev

# Declare as Flux layer for proper integration
Flux.@layer AdditiveDeviation

"""
    MultiplicativeDeviation <: AbstractDeviation

Multiplicative deviation strategy: t_new = t_old × dev

# Formula
```
t_new = t_old × dev
```

where:
- `t_old`: Current temporal parameter value
- `dev`: Deviation factor predicted by the model
- `t_new`: Updated temporal parameter value

# Characteristics
- **Type**: Relative adjustment
- **Effect**: Scales t proportionally
- **Sign**: dev > 1 increases t, 0 < dev < 1 decreases t
- **Scale**: Proportional to current t value

# Use Cases
Prefer MultiplicativeDeviation when:
- You want relative/percentage changes
- The adjustment should scale with current t value
- You're implementing momentum-like updates
- Learning rate adaptation is important

# Mathematical Properties
- Multiplicative: deviation(t, α·β) = t·α·β
- Associative: deviation(deviation(t, dev1), dev2) = deviation(t, dev1·dev2)
- Identity: deviation(t, 1) = t
- Inverse: deviation(deviation(t, dev), 1/dev) = t

# Special Cases
- `dev = 0`: Sets t to zero (collapse)
- `dev = 1`: No change (identity)
- `dev = 2`: Doubles the value of t
- `0 < dev < 1`: Decreases t proportionally
- `dev > 1`: Increases t proportionally
- `dev < 0`: Changes sign of t (typically undesirable)

# Typical Training Behavior
During training, the model typically learns to predict:
- Values near 1.0 (e.g., 0.9-1.1) for fine adjustments
- Values > 1.0 for increasing t (acceleration)
- Values < 1.0 for decreasing t (deceleration)
- The model learns that dev = 1.0 means "keep current value"

# Comparison with AdditiveDeviation
| Aspect | Additive | Multiplicative |
|--------|----------|----------------|
| Update | t + dev | t × dev |
| Small t | Same absolute change | Smaller absolute change |
| Large t | Same absolute change | Larger absolute change |
| Identity | dev = 0 | dev = 1 |
| Scaling | Absolute | Relative |

# Implementation Note
This struct is declared as a Flux layer (@layer) to ensure proper handling
during model serialization and GPU transfer, even though it has no parameters.

# Warning
Be cautious with predicted values:
- Ensure dev > 0 if negative t values are problematic
- Consider using softplus or exp activation on model output to ensure positivity
- Very large dev values can cause numerical instability
"""
struct MultiplicativeDeviation <: AbstractDeviation end

# Callable implementation: t_new = t_old × dev
(dt::MultiplicativeDeviation)(t, dev) = t .* dev

# Declare as Flux layer for proper integration
Flux.@layer MultiplicativeDeviation

"""
    NothingDeviation <: AbstractDeviation

Direct replacement deviation strategy: t_new = dev

# Formula
```
t_new = dev
```

where:
- `t_old`: Current temporal parameter value (ignored)
- `dev`: New value predicted directly by the model
- `t_new`: Updated temporal parameter value

# Characteristics
- **Type**: Direct prediction
- **Effect**: Completely replaces t with predicted value
- **Sign**: Model determines absolute value
- **Scale**: Independent of previous t value

# Use Cases
Prefer NothingDeviation when:
- The model should predict t directly, not a change
- Previous t value is irrelevant
- You want absolute control over t at each step
- The model architecture is designed for direct prediction

# Mathematical Properties
- Projection: deviation(_, dev) = dev
- Independent: deviation(t, dev) doesn't depend on t
- Idempotent: deviation(deviation(_, dev1), dev2) = deviation(_, dev2)
- The first argument (current t) is completely ignored

# Special Cases
Unlike other deviations, NothingDeviation:
- Ignores the current value of t entirely
- Treats each prediction as a fresh start
- Has no notion of "relative" change
- Makes the model responsible for absolute values

# Typical Training Behavior
During training, the model learns to predict:
- Absolute values of t appropriate for the current state
- The model must learn the valid range of t values
- No notion of "incremental" or "relative" changes
- Each prediction is made from scratch

# Comparison with Other Deviations
| Aspect | Additive | Multiplicative | Nothing |
|--------|----------|----------------|---------|
| Update | t + dev | t × dev | dev |
| Uses t_old | Yes | Yes | **No** |
| Type | Incremental | Relative | **Absolute** |
| Identity | dev = 0 | dev = 1 | **None** |
| Dependency | Linear in t | Linear in t | **Independent** |

# When to Use
Use NothingDeviation when:
✓ Model predicts t directly (not a change)
✓ History doesn't matter (Markov property)
✓ Simpler training objective (direct supervision)
✓ Model architecture outputs absolute values

Avoid NothingDeviation when:
✗ Incremental updates are more natural
✗ You want smooth transitions
✗ Previous state should influence update
✗ Model should learn adjustment strategies

# Implementation Note
This struct is declared as a Flux layer (@layer) to ensure proper handling
during model serialization and GPU transfer, even though it has no parameters.

The first argument is conventionally named `_` (underscore) to indicate
it is intentionally unused.

# Architectural Implications
When using NothingDeviation:
- Model output should be constrained to valid t range (e.g., via softplus)
- Training targets are absolute t values, not changes
- Model doesn't need to consider previous t in its predictions
- Potentially simpler to train (direct supervision)
- May be less stable (no smoothing from previous state)
"""
struct NothingDeviation <: AbstractDeviation end

# Callable implementation: t_new = dev (ignores t_old)
# The underscore (_) indicates the first argument is intentionally unused
(dt::NothingDeviation)(_, dev) = dev

# Declare as Flux layer for proper integration
Flux.@layer NothingDeviation