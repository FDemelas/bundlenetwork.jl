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

# Design Pattern
Implements the "Factory Method Pattern" where:
- Abstract factories define the interface
- Concrete factories implement specific creation logic
- Clients use the interface without knowing implementation details


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

# Typical Features (34 dimensions)
AbstractTModelFactory instances typically extract:
1. **Temporal** (1-6): t, norms, interactions
2. **Objectives** (7-9): Objective values and comparisons
3. **Indices** (10-12): Positions and alpha weights
4. **Geometric** (13-16): Norms of z and G
5. **Statistics** (17-24): Mean, std, min, max of G and z
6. **Products** (25-34): Matrix products and dot products

See `features_vector_i(B::DualBundle)` for complete details.

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