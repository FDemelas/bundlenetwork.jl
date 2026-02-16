"""
    AbstractModel

Root abstract type for all prediction models in the system.

# Type Hierarchy
```
AbstractModel
├── AbstractTModel
│   ├── RnnTModel
│   ├── RnnTModelSampleInside
│   └── ... (other t prediction models)
│
└── AbstractDirectionAndTModel
    ├── AttentionModel
    └── ... (other joint prediction models)
```

# Purpose
This abstract type serves as a common base for all machine learning models
used in the optimization system. It enables multiple dispatch and ensures
a consistent interface across different model architectures.

# Main Subtypes
- `AbstractTModel`: Models predicting only the temporal parameter t
- `AbstractDirectionAndTModel`: Models predicting both t and a direction (via attention)

# Expected Interface
All concrete subtypes of AbstractModel should implement:
- `(model)(features...)`: Forward pass (callable struct)
- `reset!(model, ...)`: State reset (for recurrent models)
- Compatibility with Flux.jl for automatic differentiation

# Example Usage
```julia
# Define a new model type
struct MyNewModel <: AbstractTModel
    encoder::Chain
    decoder::Chain
    # ... other fields
end

# The model automatically inherits from the hierarchy
model::AbstractModel = MyNewModel(...)
```

# Design Pattern
This hierarchy follows the "Strategy" design pattern where different
prediction strategies can be interchanged while respecting a common interface.

# See Also
- `AbstractTModel`: For temporal-only prediction
- `AbstractDirectionAndTModel`: For joint temporal and directional prediction
"""
abstract type AbstractModel end

"""
    AbstractTModel <: AbstractModel

Abstract type for models that predict **only the temporal parameter t**.

# Objective
These models focus on predicting a single value: the parameter t that controls
the temporal evolution of the optimization algorithm. They do not predict
directions or convex combinations of components.

# Typical Architecture
```
Features (bundle state)
    ↓
[Encoder: LSTM, GRU, Dense, ...]
    ↓
[Latent representation]
    ↓
[Decoder: MLP]
    ↓
t ∈ [t_min, t_max]
```

# Concrete Implementations
- **RnnTModel**: RNN model with variational sampling
  - Architecture: LSTM/GRU → MLP
  - Output: [μ, σ²] for t ~ N(μ, σ²)
  - Use case: t prediction with uncertainty
  
- **RnnTModelSampleInside**: RNN model with post-sampling transformation
  - Architecture: LSTM/GRU → [sampling] → MLP
  - Output: t after non-linear transformation
  - Use case: Complex relationships between state and t

# Common Functionality
AbstractTModel instances typically:
1. Take as input a feature vector from the bundle (e.g., 34 dimensions)
2. Produce a scalar value t or a distribution over t
3. Use variational sampling mechanisms (optional)
4. Apply deviation strategies (AbstractDeviation)
5. Constrain t to the interval [t_min, t_max]

# Expected Interface
```julia
# Feature creation
features = create_features(bundle, model)

# Forward pass
t = model(features, bundle)

# Reset (for recurrent models)
reset!(model)
```

# When to Use AbstractTModel
Choose an AbstractTModel when:
- You only want to control the time step of the optimization
- Direction is determined by other means (gradients, etc.)
- You need a simple and fast model
- Variational sampling of t is sufficient

# Comparison with AbstractDirectionAndTModel
| Aspect | AbstractTModel | AbstractDirectionAndTModel |
|--------|----------------|----------------------------|
| Output | t only | t + direction (attention) |
| Complexity | Simpler | More complex |
| Use case | Temporal control | Temporal + spatial control |
| Features | Global state | Global + local state |
| Training speed | Faster | Slower |

# Example
```julia
# Create an RnnTModel
factory = RnnTModelfactory()
model::AbstractTModel = create_NN(factory)

# Training loop
for iteration in 1:max_iterations
    features = create_features(bundle, model)
    t = model(features, bundle)
    # Use t to update optimization
end
```

# Design Rationale
This type separates temporal control from directional control, allowing
for modular design where different components can be combined as needed.
"""
abstract type AbstractTModel <: AbstractModel end

"""
    AbstractDirectionAndTModel <: AbstractModel

Abstract type for models that predict **both the temporal parameter t 
AND a direction** (convex combination of components).

# Objective
These models use attention or similar mechanisms to simultaneously predict:
1. **t**: The temporal parameter controlling algorithm evolution
2. **γ**: An attention distribution over bundle components

This joint prediction enables fine-grained control of optimization by
intelligently choosing both the "when" (t) and the "how" (direction).

# Typical Architecture
```
Features (state + components)
    ↓
[Encoder: LSTM with attention]
    ↓
[Latent representation]
    ↓        ↓
[Dec. t]  [Dec. attention]
    ↓        ↓
    t      γ (attention scores)
```

# Concrete Implementations
- **AttentionModel**: Model with key-query attention mechanism
  - Architecture: LSTM → VAE → [decoder_t, decoder_γq, decoder_γk]
  - Mechanism: Multi-head attention with keys and queries
  - Output: t + attention scores over components
  - Use case: Intelligent component selection at each iteration

# Attention Mechanism
The model maintains a key matrix (K) for all observed components:
```
Query (q): Current component
Keys (K): All previous components
Attention: γ = softmax(q^T K)
```

This allows computing a weighted convex combination of components based
on their similarity to the current state.

# Common Functionality
AbstractDirectionAndTModel instances typically:
1. Take as input features at two levels:
   - Global features (temporal state)
   - Per-component features (gradient, objective, etc.)
2. Produce two outputs:
   - t: Temporal parameter
   - γ: Attention scores (logits before softmax/sparsemax)
3. Maintain history of components (key matrix)
4. Use variational mechanisms for regularization

# Expected Interface
```julia
# Feature creation (two levels)
feat_t, feat_gamma = create_features(factory, bundle)

# Forward pass
t, attention_scores = model(feat_t, feat_gamma, idx, comps)

# Reset
reset!(model, batch_size, max_iterations)
```

# When to Use AbstractDirectionAndTModel
Choose an AbstractDirectionAndTModel when:
- You want to control both timing AND direction
- Bundle components have variable importance
- You want a learned selection mechanism (vs heuristic)
- Additional complexity is justified by performance gains
- You have sufficient data to train the attention mechanism

# Advantages
✓ Fine-grained optimization control
✓ Intelligent component selection
✓ Dynamic adaptation to bundle state
✓ Interpretable mechanism (attention scores)

# Disadvantages
✗ More complex to train
✗ More computationally expensive (especially with large bundles)
✗ Requires more data
✗ More hyperparameters to tune

# Comparison with AbstractTModel
| Aspect | AbstractTModel | AbstractDirectionAndTModel |
|--------|----------------|----------------------------|
| **Predictions** | t only | t + direction |
| **Input** | Global features | Global + local features |
| **Architecture** | Simple (RNN+MLP) | Complex (RNN+Attention) |
| **Memory** | Recurrent state | State + key matrix |
| **Use case** | Temporal control | Temporal + spatial control |
| **Interpretability** | Medium | High (via attention) |
| **Training time** | Fast | Slow |
| **Data requirements** | Moderate | High |

# Example
```julia
# Create an AttentionModel
factory = AttentionModelFactory()
model::AbstractDirectionAndTModel = create_NN(
    factory,
    h_representation = 256,
    h3_representations = true  # Three independent latent spaces
)

# Initialize
reset!(model, batch_size=1, max_iterations=500)

# Training loop
for iteration in 1:max_iterations
    # Extract two-level features
    feat_t, feat_gamma = create_features(factory, bundle)
    
    # Joint prediction
    t, attention_scores = model(feat_t, feat_gamma, iteration, 1:bundle.size)
    
    # Apply softmax/sparsemax to get weights
    weights = softmax(attention_scores)
    
    # Use t and weights to update optimization
    update_bundle!(bundle, t, weights)
end
```

# Design Pattern
This abstract type implements a variant of the "Observer" pattern where the model
observes the complete history of components (via stored keys) to make decisions
about the current component.

# Performance Considerations
The attention mechanism scales with the number of components:
- Memory: O(h_representation × max_iterations)
- Computation: O(h_representation × num_components)

For very large bundles, consider:
- Limiting the attention window
- Using sparse attention mechanisms
- Batching component processing
"""
abstract type AbstractDirectionAndTModel <: AbstractModel end

"""
    AbstractModelFactory

Root abstract type for all model factories.

# Objective
Factories encapsulate the logic for creating and configuring models.
They follow the "Factory Method" design pattern, which separates object
creation from usage.

# Type Hierarchy
```
AbstractModelFactory
├── AbstractTModelFactory
│   ├── RnnTModelfactory
│   ├── RnnTModelSampleInsidefactory
│   └── ... (other t model factories)
│
└── AbstractDirectionAndTModelFactory
    ├── AttentionModelFactory
    └── ... (other joint model factories)
```

# Factory Responsibilities
A factory is responsible for:
1. **Feature extraction**: Defining which features to use
2. **Model creation**: Instantiating models with correct architecture
3. **Configuration**: Managing hyperparameters and initialization
4. **Metadata**: Providing feature and output dimensions

# Advantages of the Factory Pattern
✓ **Separation of concerns**: Creation vs usage
✓ **Flexibility**: Easy to add new model types
✓ **Reusability**: Centralized and reusable configuration
✓ **Testability**: Easy to mock factories for testing
✓ **Consistency**: Ensures models are created correctly

# Expected Interface
All concrete factories must implement:
```julia
# Extract features from bundle
create_features(factory, bundle)

# Feature and output dimensions
size_features(factory)
size_output(factory)

# Create configured model
create_NN(factory, hyperparameters...)
```

# Typical Workflow
```julia
# 1. Create factory
factory = MyModelFactory()

# 2. Create model with hyperparameters
model = create_NN(
    factory,
    h_representation = 256,
    seed = 42
)

# 3. Use factory to extract features
features = create_features(factory, bundle)

# 4. Forward pass on model
output = model(features, bundle)
```

# Design Pattern
Implements the "Factory Method Pattern" where:
- Abstract factories define the interface
- Concrete factories implement specific creation logic
- Clients use the interface without knowing implementation details

# Complete Example
```julia
# Define a new factory
struct MyNewTModelFactory <: AbstractTModelFactory end

# Implement required methods
function create_features(factory::MyNewTModelFactory, B)
    # Feature extraction logic
    return features
end

function size_features(factory::MyNewTModelFactory)
    return 42  # Number of features
end

function create_NN(factory::MyNewTModelFactory; kwargs...)
    # Model creation logic
    return MyNewModel(...)
end

# Usage
factory = MyNewTModelFactory()
model = create_NN(factory, seed=123)
```

# See Also
- `AbstractTModelFactory`: For t-only prediction models
- `AbstractDirectionAndTModelFactory`: For joint prediction models
"""
abstract type AbstractModelFactory end

"""
    AbstractTModelFactory <: AbstractModelFactory

Abstract type for factories creating AbstractTModel instances.

# Objective
These factories specialize in creating models that predict only the temporal
parameter t. They configure feature extraction and architecture appropriate
for this specific task.

# Concrete Implementations
- **RnnTModelfactory**: Creates RnnTModel instances
  - Features: 34 dimensions (bundle state)
  - Architecture: LSTM/GRU + MLP
  - Output: 2 values [μ, σ²]
  
- **RnnTModelSampleInsidefactory**: Creates RnnTModelSampleInside instances
  - Features: 34 dimensions (same as RnnTModel)
  - Architecture: LSTM/GRU + [sampling] + MLP
  - Output: 1 value (transformed deviation)

# Required Interface
Concrete factories must implement:
```julia
# Dimensions
size_features(factory)::Int      # e.g., 34
size_output(factory)::Int        # e.g., 2 for [μ, σ²]

# Feature extraction
create_features(factory, bundle::DualBundle)  # Returns features (34×1)

# Model creation
create_NN(factory; 
    recurrent_layer = LSTM,
    h_decoder = [512, 256],
    h_act = softplus,
    h_representation = 128,
    seed = 1,
    norm = false
)  # Returns AbstractTModel
```

# Typical Features (34 dimensions)
AbstractTModelFactory instances typically extract:
1. **Temporal** (1-6): t, norms, interactions
2. **Objectives** (7-9): Objective values and comparisons
3. **Indices** (10-12): Positions and alpha weights
4. **Geometric** (13-16): Norms of z and G
5. **Statistics** (17-24): Mean, std, min, max of G and z
6. **Products** (25-34): Matrix products and dot products

See `features_vector_i(B::DualBundle)` for complete details.

# Usage Workflow
```julia
# 1. Create factory
factory = RnnTModelfactory()

# 2. Check dimensions
@assert size_features(factory) == 34
@assert size_output(factory) == 2

# 3. Create model
model = create_NN(
    factory,
    recurrent_layer = LSTM,
    h_representation = 256,
    seed = 42
)

# 4. Training loop
for epoch in 1:num_epochs
    for bundle in training_data
        # Extract features
        features = create_features(factory, bundle)
        
        # Forward pass
        t = model(features, bundle)
        
        # Compute loss and backprop
        loss = compute_loss(t, target)
        Flux.train!(loss, model, optimizer)
    end
end

# 5. Inference
features = create_features(factory, test_bundle)
t_pred = model(features, test_bundle)
```

# Creating a Custom Factory
```julia
struct MyCustomTFactory <: AbstractTModelFactory end

function size_features(::MyCustomTFactory)
    return 50  # Custom feature count
end

function size_output(::MyCustomTFactory)
    return 2  # [μ, σ²]
end

function create_features(factory::MyCustomTFactory, B::DualBundle)
    # Extract custom features
    ϕ = Float32[
        B.params.t,
        sum(B.w' * B.w),
        # ... 48 more features
    ]
    return reshape(ϕ, (50, 1))
end

function create_NN(factory::MyCustomTFactory; kwargs...)
    # Create custom model
    rng = MersenneTwister(get(kwargs, :seed, 1))
    encoder = LSTM(50 => 128)  # 50 features input
    decoder = Chain(Dense(128 => 256, gelu), Dense(256 => 2))
    model_chain = Chain(encoder, decoder)
    return RnnTModel(model_chain, rng, true)
end

# Usage
factory = MyCustomTFactory()
model = create_NN(factory, seed=123)
```

# Naming Conventions
- Factory: `*Modelfactory` (e.g., RnnTModelfactory)
- Model: `*Model` (e.g., RnnTModel)
- Abstract factory type: `Abstract*ModelFactory`
- Abstract model type: `Abstract*Model`

# See Also
- `RnnTModelfactory`: Standard RNN-based factory
- `RnnTModelSampleInsidefactory`: Two-stage architecture factory
- `features_vector_i`: Feature extraction implementation
"""
abstract type AbstractTModelFactory <: AbstractModelFactory end

"""
    AbstractDirectionAndTModelFactory <: AbstractModelFactory

Abstract type for factories creating AbstractDirectionAndTModel instances.

# Objective
These factories specialize in creating models that predict both the temporal
parameter t and an attention distribution over bundle components. They manage
two-level feature extraction and configuration of complex attention architectures.

# Concrete Implementations
- **AttentionModelFactory**: Creates AttentionModel instances
  - Temporal features: 16 dimensions (global state)
  - Component features: 20 dimensions (per component)
  - Architecture: LSTM + VAE + Attention (keys/queries)
  - Outputs: t (scalar) + γ (attention scores)

# Required Interface
Concrete factories must implement:
```julia
# Dimensions
size_features(factory)::Int           # e.g., 16 (temporal features)
size_comp_features(factory)::Int      # e.g., 20 (component features)
h_representation(model)::Int          # e.g., 128 (latent dimension)

# Feature extraction (two levels)
create_features(factory, bundle::SoftBundle)
    # Returns (feat_t, feat_gamma)
    # feat_t: (16, 1) - temporal features
    # feat_gamma: (20, 1) - component features

create_features(factory, bundle::BatchedSoftBundle)
    # Returns (feat_t, feat_theta)
    # feat_t: (16, batch_size) - batched temporal features
    # feat_theta: (20, batch_size) - batched component features

# Model creation
create_NN(factory;
    h_act = gelu,
    h_representation = 128,
    h_decoder = [1024],
    seed = 1,
    norm = false,
    sampling_t = false,
    sampling_θ = true,
    ot_act = softplus,
    rnn = true,
    h3_representations = false,
    repeated = true,
    use_tanh = false
)  # Returns AbstractDirectionAndTModel
```

# Two-Level Feature Structure

## Temporal Features (16 dimensions)
Global features describing the temporal state of optimization:
1. **Time and norms** (1-6): t, ‖w‖², t·‖w‖², α'θ, comparisons
2. **Objectives** (7-10): obj[li], obj[s], comparisons, indices
3. **Alphas** (11-12): α[s], α[li]
4. **Vector norms** (13-16): √(z'z), √(G'G)

## Component Features (20 dimensions)
Local features for each bundle component:
1. **Statistics G and z** (1-8): mean, std, min, max
2. **Product mins** (9-12): minimum(z'z), minimum(G'G), etc.
3. **Dot products** (13, 18): G'w
4. **Product maxs** (14-17): maximum(z'z), maximum(G'G), etc.
5. **Metadata** (19-20): α[li], obj[li]

# Typical Architecture Created
```
Input Level 1: Temporal features (16)
Input Level 2: Component features (20)
    ↓
[Concatenation: 16 + 20 = 36]
    ↓
╔════════════════════════════════════╗
║  ENCODER (LSTM optional)           ║
║  36 → h_representation × (2 or 8)  ║
╚════════════════════════════════════╝
    ↓
[Split μ and σ²]
    ↓ [VAE sampling]
    ↓
╔════════════╦═══════════╦═══════════╗
║  Dec. t    ║  Dec. γq  ║  Dec. γk  ║
║  h → ... →1║  h → ... →h║  h → ... →h║
╚════════════╩═══════════╩═══════════╝
    ↓            ↓           ↓
    t        query (q)   key (k)
    
    [Attention: γ = q^T · K]
```

# Usage Workflow
```julia
# 1. Create factory
factory = AttentionModelFactory()

# 2. Check dimensions
@assert size_features(factory) == 16
@assert size_comp_features(factory) == 20

# 3. Create model
model = create_NN(
    factory,
    h_representation = 256,
    h3_representations = true,  # 3 independent latent spaces
    sampling_t = true,
    sampling_θ = true,
    seed = 42
)

# 4. Initialize for a sequence
reset!(model, batch_size=1, max_iterations=500)

# 5. Training loop
for epoch in 1:num_epochs
    for bundle in training_data
        # Extract two-level features
        feat_t, feat_gamma = create_features(factory, bundle)
        
        # Forward pass
        t, attention_scores = model(
            feat_t, 
            feat_gamma, 
            bundle.li,           # Current index
            1:bundle.size        # Components to consider
        )
        
        # Apply softmax to get weights
        weights = softmax(attention_scores)
        
        # Compute loss and backprop
        loss = compute_loss(t, weights, targets)
        Flux.train!(loss, model, optimizer)
    end
    
    # Reset between epochs
    reset!(model)
end

# 6. Inference
reset!(model, batch_size=1, max_iterations=100)
for iteration in 1:100
    feat_t, feat_gamma = create_features(factory, bundle)
    t, scores = model(feat_t, feat_gamma, iteration, 1:bundle.size)
    weights = sparsemax(scores)  # Or softmax
    # Use t and weights for optimization
end
```

# Batch Processing
Factories support two bundle types:
```julia
# Single bundle (one example)
feat_t, feat_gamma = create_features(factory, bundle::SoftBundle)
# feat_t: (16, 1), feat_gamma: (20, 1)

# Batched bundle (multiple examples)
feat_t, feat_theta = create_features(factory, bundle::BatchedSoftBundle)
# feat_t: (16, batch_size), feat_theta: (20, batch_size)
```

# Attention Mechanism Details
The model maintains a **key matrix K** (stored in `model.Ks`):
```julia
# At each iteration i:
# 1. Compute key k_i = decoder_γk(z_i)
# 2. Store in K[:, i] = k_i
# 3. Compute query q_i = decoder_γq(z_i)
# 4. Compute scores: γ_i = q_i^T · K[:, 1:i]
# 5. Apply softmax/sparsemax: weights = softmax(γ_i)
```

This approach allows the model to "look back" and weight all previous
components based on their similarity to the current state.

# Creating a Custom Factory
```julia
struct MyCustomAttentionFactory <: AbstractDirectionAndTModelFactory end

function size_features(::MyCustomAttentionFactory)
    return 20  # Custom temporal features
end

function size_comp_features(::MyCustomAttentionFactory)
    return 30  # Custom component features
end

function create_features(
    factory::MyCustomAttentionFactory, 
    B::SoftBundle
)
    # Extract temporal features (20)
    feat_t = Float32[
        sum(B.t),
        sum(B.w' * B.w),
        # ... 18 more features
    ]
    
    # Extract component features (30)
    feat_gamma = Float32[
        mean(B.G[:, B.li]),
        std(B.z[:, B.li]),
        # ... 28 more features
    ]
    
    return device(feat_t), device(feat_gamma)
end

function create_NN(factory::MyCustomAttentionFactory; kwargs...)
    # Create custom attention model
    h_rep = get(kwargs, :h_representation, 128)
    
    # Encoder
    encoder = LSTM(50 => 2*h_rep)  # 20+30=50 features
    
    # Decoders
    decoder_t = Chain(Dense(h_rep => 512, gelu), Dense(512 => 1, softplus))
    decoder_k = Chain(Dense(h_rep => 256, gelu), Dense(256 => h_rep))
    decoder_q = Chain(Dense(h_rep => 256, gelu), Dense(256 => h_rep))
    
    # Create model
    rng = MersenneTwister(get(kwargs, :seed, 1))
    model = AttentionModel(
        device(encoder),
        device(decoder_t),
        device(decoder_k),
        device(decoder_q),
        rng,
        # ... other parameters
    )
    
    return model
end

# Usage
factory = MyCustomAttentionFactory()
model = create_NN(factory, h_representation=256, seed=123)
```

# Advantages of This Approach
✓ **Clear separation**: Temporal vs local features
✓ **Flexibility**: Easy to experiment with different features
✓ **Reusability**: Feature extraction logic is centralized
✓ **Batch-friendly**: Native batching support
✓ **Maintainability**: Feature modifications don't affect model code

# Naming Conventions
- Factory: `*ModelFactory` (e.g., AttentionModelFactory)
- Model: `*Model` (e.g., AttentionModel)
- Temporal features: `feat_t`, `ϕ`
- Component features: `feat_gamma`, `feat_theta`, `ϕγ`

# See Also
- `AttentionModelFactory`: Key-query attention factory
- `create_features`: Feature extraction methods
- `AttentionModel`: Concrete model implementation
"""
abstract type AbstractDirectionAndTModelFactory <: AbstractModelFactory end