# Neural Network Architecture

## Overview

The neural network architecture is designed to predict bundle method parameters 
from the current optimization state.

## Model Components

### Encoder

Processes instance features and bundle state:
```
Input Features → Linear → Activation → Linear → Hidden Representation
```

**Input features include**:
- Bundle gradients (subgradients)
- Function values
- Dual variables (optional: from CR)
- Instance structure (optional: graph features)

### Decoder for Proximity Parameter (t)

Predicts the proximity parameter:
```
Hidden Representation → Linear → Softplus → t
```

**Output**: Scalar proximity parameter controlling trust region size.

### Decoder for Gradient Aggregation (γ)

Predicts weights for convex combination of gradients:
```
Hidden Representation → Attention → Distribution → θ
```

**Components**:
- **Query Network**: Generates query from hidden state
- **Key Network**: Generates keys from bundle components
- **Attention Mechanism**: Computes attention scores
- **Distribution**: Softmax or Sparsemax normalization

**Output**: Probability distribution over bundle components.

## Attention Mechanism

The attention mechanism computes how to aggregate bundle gradients:
```
Q = QueryNetwork(hidden_state)
K = KeyNetwork(bundle_gradients)
V = bundle_gradients

scores = Q · K^T
θ = softmax(scores)
aggregated_gradient = Σ θ_i * V_i
```

## Architecture Variants

### Standard Architecture (h3=false)

Three separate hidden representations:
- One for proximity parameter
- One for attention queries
- One for attention keys

### Compact Architecture (h3=true)

Single shared hidden representation for all outputs.

## Model Factory

Models are created via factories:
```julia
factory = BundleNetworks.AttentionModelFactory()
nn = BundleNetworks.create_NN(
    factory;
    h_representation=64,
    h_act=softplus,
    sampling_θ=false,
    sampling_t=true
)
```

**Parameters**:
- `h_representation`: Hidden layer size
- `h_act`: Activation function
- `sampling_θ`: Sample attention weights
- `sampling_t`: Sample proximity parameter

## Sampling Strategies

### Deterministic (Default for Testing)
```julia
nn.sample_t = false
nn.sample_γ = false
```

Outputs are mean predictions.

### Stochastic (Training)
```julia
nn.sample_t = true
nn.sample_γ = true
```

Samples from learned distributions for exploration.

## Graph Features

When `use_graph=true`, bipartite graph features are extracted:
```
Instance → Bipartite Graph → Graph Convolution → Features
```

This captures:
- Variable-constraint relationships
- Network structure
- Sparsity patterns

## Activation Functions

Supported activations:
- **softplus**: Smooth, always positive (good for t)
- **relu**: Sparse, fast
- **tanh**: Bounded, smooth
- **gelu**: Modern, smooth

## Parameter Count

Typical model sizes:

| h_representation | Parameters |
|------------------|-----------|
| 32               | ~10K      |
| 64               | ~40K      |
| 128              | ~160K     |

## Next Steps

- See [Bundle Methods](@ref) for algorithm details
- Learn about [Loss Functions](@ref)
- Explore [Hyperparameter Tuning](@ref)
