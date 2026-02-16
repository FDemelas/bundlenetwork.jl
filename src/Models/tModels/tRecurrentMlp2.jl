"""
    RnnTModelSampleInside <: AbstractTModel

RNN model with variational sampling **integrated into the architecture**.

# Key Difference from RnnTModel
Unlike `RnnTModel` where sampling directly produces a deviation, here the
sampling is followed by **transformation through an additional network**:
- RnnTModel: t = deviation(t_current, μ + ϵ·σ)
- RnnTModelSampleInside: t = deviation(t_current, **MLP(μ + ϵ·σ)**)

This two-stage architecture allows the model to learn a more complex non-linear
transformation of the latent sample before applying it as a deviation.

# Architecture
```
Input (34 features)
    ↓
Recurrent Encoder (LSTM/GRU)
    34 → 2*h_representation (for μ and σ²)
    ↓
Sampling: z ~ N(μ, σ²)
    ↓
Decoder MLP (transforms z)
    h_representation → ... → 1
    ↓
Deviation application
    t = deviation(t_current, MLP(z))
```

# Fields
- `model::Any`: **Vector of 2 components** [Encoder, Decoder]
  - model[1]: Recurrent encoder producing [μ, σ²]
  - model[2]: MLP decoder transforming the sample
- `rng::Any`: Random number generator (MersenneTwister)
- `sample::Bool`: If true, samples z ~ N(μ, σ²); otherwise uses μ
- `ϵs::AbstractVector`: Storage for random components ϵ ~ N(0,1) (training)
- `deviation::AbstractDeviation`: Strategy for applying deviation to t
- `train_mode::Bool`: Training mode (true) or inference mode (false)

# Advantages of This Architecture
1. **Increased expressiveness**: The MLP can learn complex transformations
2. **Flexibility**: The applied deviation can be non-linear and contextual
3. **Regularization**: Sampling in the middle adds regularizing noise

# Use Cases
Prefer this architecture when:
- The relationship between state and deviation is complex and non-linear
- You want more modeling capacity
- You have sufficient data to train the additional MLP

# Example
```julia
rng = MersenneTwister(42)
encoder = LSTM(34 => 256)  # Produces 2*128 = 256 for μ and σ²
decoder = Chain(Dense(128 => 512, gelu), Dense(512 => 1, softplus))
model = RnnTModelSampleInside(
    [encoder, decoder],
    rng,
    true,  # sampling enabled
    [],
    NothingDeviation(),
    true   # training mode
)
```

# See Also
- `RnnTModel`: Simpler variant without post-sampling transformation
- `AbstractDeviation`: Different deviation strategies
"""
mutable struct RnnTModelSampleInside <: AbstractTModel
	model::Any
	rng::Any
	sample::Bool
	ϵs::AbstractVector
	deviation::AbstractDeviation
	train_mode::Bool
	
	# Constructor with default values
	RnnTModelSampleInside(model, rng, sample = false, es = [], deviation = NothingDeviation(), train_mode = true) = 
		new(model, rng, sample, es, deviation, train_mode)
end

"""
    RnnTModelSampleInsidefactory <: AbstractTModelFactory

Factory for creating RnnTModelSampleInside models with two-stage architecture.

This factory automatically configures:
- A recurrent encoder producing 2*h_representation outputs (for μ and σ²)
- An MLP decoder transforming latent samples
- Appropriate connections between the two components

# Difference from RnnTModelfactory
- RnnTModelfactory: Output dimension 2 [μ, σ²], no post-sampling transformation
- RnnTModelSampleInsidefactory: Intermediate output [μ, σ²], then MLP transforms sample

# Design Pattern
Implements the Factory Method pattern with a two-stage architecture specification.
"""
struct RnnTModelSampleInsidefactory <: AbstractTModelFactory end

"""
    create_features(B::DualBundle, _::RnnTModelSampleInside)

Creates the feature vector from a DualBundle for an RnnTModelSampleInside model.

# Arguments
- `B::DualBundle`: Bundle containing all information from current iteration
- `_::RnnTModelSampleInside`: Model instance (used for dispatch)

# Returns
- Matrix of dimensions (34, 1) containing features extracted from bundle

# Note
This function uses exactly the same features as RnnTModel (34 dimensions).
See the documentation for `features_vector_i(B::DualBundle)` in rnn_t_model.jl
for detailed feature descriptions.
"""
function create_features(B::DualBundle, _::RnnTModelSampleInside)
	# Extract 34 features from current iteration
	φ = features_vector_i(B)
	
	# Reshape to matrix format (features, batch_size)
	# batch_size = 1 since processing a single bundle
	return reshape(φ, (length(φ), 1))
end

"""
    size_features(_::RnnTModelSampleInside)

Returns the dimension of the feature vector for an RnnTModelSampleInside.

# Arguments
- `_::RnnTModelSampleInside`: Model instance (used for dispatch)

# Returns
- `34`: Number of features extracted from DualBundle

# Note
Identical to RnnTModel - both models use the same input features.
The difference lies in the internal architecture, not the inputs.
"""
function size_features(_::RnnTModelSampleInside)
	return 34
end


"""
    size_features(_::RnnTModelSampleInsidefactory)

Returns the dimension of the feature vector for models created by 
RnnTModelSampleInsidefactory.

# Arguments
- `_::RnnTModelSampleInsidefactory`: Model factory (used for dispatch)

# Returns
- `34`: Number of features extracted from DualBundle
"""
function size_features(_::RnnTModelSampleInsidefactory)
	return 34
end

"""
    (m::RnnTModelSampleInside)(ϕ, B, ϵ = randn(B.nn.rng, Float32, 1))

Forward pass of the RnnTModelSampleInside with post-sampling transformation.

# Arguments
- `ϕ`: Feature vector (dimension: 34 × 1)
- `B`: Bundle containing parameters and constraints
- `ϵ`: Random vector N(0,1) for reparameterization trick (optional)

# Returns
- `t::Float32`: Predicted temporal parameter value, constrained to [t_min, t_max]

# Forward Pass Architecture

```
ϕ (34,1)
    ↓
m.model[1]: Recurrent encoder
    ↓
[μ, σ²] (2*h_representation)
    ↓ [Split into two]
μ (h_representation)   σ² (h_representation)
    ↓                       ↓
    └──────[z = μ + ϵ·σ]────┘  (Reparameterization trick)
              ↓
        z (h_representation)
              ↓
    m.model[2]: MLP decoder
              ↓
        dev_transformed (1)
              ↓
    deviation(t_current, dev_transformed)
              ↓
        t final (constrained)
```

# Detailed Process

## 1. Encoding
`f = m.model[1](ϕ)`: The recurrent encoder processes features
Output: vector of dimension 2*h_representation

## 2. Splitting μ and σ²
`μ, σ² = chunk(f, 2)`: Split into two equal parts
- μ: first half (h_representation dimensions)
- σ²: second half (h_representation dimensions)

## 3. Variance Stabilization (if sample=true)
Successive transformations to ensure σ² > 0 and stable:
```julia
σ² = 2 - softplus(2 - σ²)        # Upper bound
σ² = -6 + softplus(σ² + 6)       # Lower bound  
σ² = exp(σ²)                      # Convert to positive
σ = √σ² / 100                     # Standard deviation with scaling
```

## 4. Sampling (Reparameterization Trick)
If `m.sample = true`:
- Generates ϵ ~ N(0,1)
- Computes z = μ + ϵ·σ
- Saves ϵ in training mode

If `m.sample = false`:
- Uses z = μ directly

## 5. **MLP Transformation** (KEY DIFFERENCE)
`dev_transformed = m.model[2](z)`: The MLP decoder transforms the sample
This step allows learning a complex non-linear function of z

## 6. Deviation Application
`t = m.deviation(t_current, dev_transformed)`
The transformed deviation is applied to current time

## 7. Interval Constraint
`t = min(t_max, |t| + t_min)`: Ensures t ∈ [t_min, t_max]

# Comparison with RnnTModel

| Step | RnnTModel | RnnTModelSampleInside |
|------|-----------|----------------------|
| Encoding | LSTM → [μ, σ²] | LSTM → [μ, σ²] |
| Sampling | z = μ + ϵ·σ | z = μ + ϵ·σ |
| **Transformation** | **None** | **MLP(z) ← KEY DIFFERENCE** |
| Deviation | deviation(t, z) | deviation(t, MLP(z)) |

# Why the Additional MLP?
The MLP transformation layer allows the model to:
1. Learn complex non-linear relationships between latent state and deviation
2. Adapt the deviation magnitude and direction based on context
3. Provide more modeling flexibility than direct sampling

This is particularly useful when the optimal deviation strategy is not
a simple linear function of the latent state.

# Example
```julia
# Forward pass
ϕ = create_features(bundle, model)
t_pred = model(ϕ, bundle)

# With explicit epsilon
ϵ = randn(rng, Float32, 1)
t_pred = model(ϕ, bundle, ϵ)
```

# Note on Gradients
Gradients propagate through all steps thanks to automatic differentiation,
including the MLP transformation. The reparameterization trick enables
gradient flow through the sampling operation.
"""
function (m::RnnTModelSampleInside)(ϕ, B, ϵ = randn(B.nn.rng, Float32, 1))
	# Step 1: Pass through recurrent encoder (first model component)
	# Produces a vector of dimension 2*h_representation
	f = m.model[1](ϕ)
	
	# Step 2: Split into mean (μ) and raw variance (σ²)
	# Each part has dimension h_representation
	μ, σ2 = Flux.MLUtils.chunk(f, 2, dims = 1)
	
	if m.sample
		# Step 3: Stabilize variance using softplus transformations
		# These ensure σ² > 0 and is numerically stable
		
		# First transformation: upper bound
		σ2 = 2.0f0 .- softplus.(2.0f0 .- σ2)
		
		# Second transformation: lower bound
		σ2 = -6.0f0 .+ softplus.(σ2 .+ 6.0f0)
		
		# Convert to positive space and compute standard deviation
		σ2 = exp.(σ2)
		sigma = sqrt.(σ2) / 100  # Scaling by 100
		
		if m.train_mode
			# In training mode: save ϵ for analysis
			# ignore_derivatives prevents affecting gradients
			ignore_derivatives() do
				push!(m.ϵs, ϵ)
			end
		end
		
		# Step 4: Reparameterization trick
		# z ~ N(μ, σ²) ⟺ z = μ + σ·ϵ where ϵ ~ N(0,1)
		# Enables gradient propagation through sampling
		dev = (μ .+ ϵ .* sigma)
	else
		# No sampling: use mean directly
		# Useful for deterministic inference
		dev = μ
	end
	
	# Step 5: MLP TRANSFORMATION (second model component)
	# THIS IS THE KEY DIFFERENCE FROM RnnTModel
	# The MLP decoder learns a non-linear transformation of the sample
	dev_transformed = m.model[2](dev)
	
	# Step 6: Apply deviation strategy to current time
	# The transformed deviation is applied according to chosen strategy
	t = m.deviation(B.params.t, dev_transformed)
	
	# Step 7: Constrain t to interval [t_min, t_max]
	# |t| ensures positivity, then add t_min and cap at t_max
	t = min.(B.params.t_max, abs.(t) .+ B.params.t_min)
	
	return mean(t)
end

# Declare model as Flux layer (for automatic differentiation)
Flux.@layer RnnTModelSampleInside

"""
    size_output(_::RnnTModelSampleInsidefactory)

Returns the dimension of the MLP decoder output.

# Arguments
- `_::RnnTModelSampleInsidefactory`: Model factory (used for dispatch)

# Returns
- `1`: The MLP decoder produces a single value (the transformed deviation)

# Note
Unlike RnnTModelfactory which returns 2 [μ, σ²], here we return 1
because the decoder transforms the latent sample into a single deviation value.

The encoder still produces 2*h_representation values (for μ and σ²), but
the final decoder reduces to 1 dimension.
"""
function size_output(_::RnnTModelSampleInsidefactory)
	return 1
end

"""
    create_NN(lt::RnnTModelSampleInsidefactory; kwargs...)

Creates and initializes an RnnTModelSampleInside with two-stage architecture.

# Arguments
- `lt::RnnTModelSampleInsidefactory`: Factory for model creation

# Keyword Arguments
- `recurrent_layer`: Type of recurrent layer to use (default: LSTM)
  Can be LSTM, GRU, or any other Flux recurrent layer
- `h_decoder::Vector{Int}`: Hidden layer dimensions for MLP decoder (default: [512, 256])
- `h_act`: Activation function for decoder hidden layers (default: softplus)
- `h_representation::Int`: Dimension of latent representation (default: 128)
- `seed::Int`: Random seed for initialization (default: 1)
- `norm::Bool`: If true, normalizes inputs (default: false)

# Returns
- `RnnTModelSampleInside`: Initialized model with sampling enabled by default

# Architecture Created

```
Input (34 features)
    ↓
[Optional Normalization]
    ↓
╔════════════════════════════════╗
║  ENCODER (m.model[1])          ║
║  Recurrent Layer (LSTM/GRU)    ║
║  34 → 2*h_representation       ║
║       (2*128 = 256)            ║
╚════════════════════════════════╝
    ↓
[μ, σ²] (each h_representation)
    ↓
[Sampling: z = μ + ϵ·σ]
    ↓
z (h_representation = 128)
    ↓
╔════════════════════════════════╗
║  DECODER (m.model[2])          ║
║  Multi-Layer Perceptron        ║
║  128 → 512 → 256 → 1           ║
║  (softplus on output)          ║
╚════════════════════════════════╝
    ↓
dev_transformed (1 value)
```

# Architecture Details

## Encoder (model[1])
- **Type**: Recurrent layer (LSTM by default)
- **Input**: 34 features (normalized if norm=true)
- **Output**: 2*h_representation (= 256 if h_representation=128)
  - First half: μ (mean)
  - Second half: σ² (variance)
- **Role**: Captures temporal dependencies and encodes to latent space

## Decoder (model[2])
- **Type**: Multi-Layer Perceptron
- **Input**: h_representation (= 128) from sample z
- **Hidden layers**: h_decoder (default [512, 256])
  - Activations: h_act (softplus)
- **Output**: 1 value with softplus activation
  - Ensures positive deviation
- **Role**: Transforms latent sample into applicable deviation

# Key Difference from create_NN of RnnTModelfactory

| Aspect | RnnTModelfactory | RnnTModelSampleInsidefactory |
|--------|------------------|------------------------------|
| Structure | Single encoder | **Encoder + Decoder separate** |
| Encoder output | 2 [μ, σ²] | **2*h_rep** [μ, σ²] each h_rep |
| Transformation | None | **MLP transforms sample** |
| Final output | 2 values | **1 value** |
| Returned model | Single Chain | **Vector [encoder, decoder]** |

# Why Two Stages?
The two-stage architecture provides:
1. **Separation of concerns**: Encoding (temporal patterns) vs transformation (deviation function)
2. **Increased capacity**: Additional parameters for complex relationships
3. **Flexibility**: Can learn sophisticated deviation strategies

The encoder learns to compress temporal information into a latent representation,
while the decoder learns how to transform that representation into an effective
deviation value.

# Example Usage

```julia
# Default configuration
factory = RnnTModelSampleInsidefactory()
model = create_NN(factory)
# Architecture: LSTM(34→256) then MLP(128→512→256→1)

# Custom configuration with GRU
model = create_NN(
    factory,
    recurrent_layer = GRU,
    h_decoder = [1024, 512, 256],
    h_representation = 256,
    h_act = gelu,
    seed = 42,
    norm = true
)
# Architecture: GRU(34→512) then MLP(256→1024→512→256→1)

# Minimal configuration
model = create_NN(
    factory,
    h_decoder = [256],  # Single hidden layer
    h_representation = 64
)
# Architecture: LSTM(34→128) then MLP(64→256→1)
```

# Hyperparameter Guidelines

## h_representation
- Larger values give more capacity to latent space
- Should be balanced with available training data
- Typical range: 64-256

## h_decoder
- Controls complexity of deviation transformation
- Deeper networks can learn more complex functions
- Risk of overfitting with limited data

## h_act
- softplus: Smooth, always positive (default)
- gelu: Modern choice, good performance
- relu: Fast, may have dead neurons

# Important Notes
- **The returned model is a vector** [encoder, decoder], not a single Chain
- The encoder produces 2*h_representation to enable sampling
- The decoder has **softplus activation on output** to ensure dev > 0
- `sample=true` by default (variational sampling enabled)
- Initialization is reproducible via seed
- This model is better suited for complex non-linear relationships

# When to Use This vs RnnTModel?
Use RnnTModelSampleInside when:
- ✓ The deviation strategy is complex and non-linear
- ✓ You have sufficient data to train the additional MLP
- ✓ Direct sampling doesn't provide enough flexibility
- ✓ You want to learn sophisticated deviation functions

Use RnnTModel when:
- ✓ Simplicity and speed are priorities
- ✓ Training data is limited
- ✓ Direct sampling is sufficient
- ✓ Interpretability is important
"""
function create_NN(
	lt::RnnTModelSampleInsidefactory,
	recurrent_layer = LSTM, 
	h_decoder::Vector{Int} = [512, 256], 
	h_act = softplus, 
	h_representation::Int = 128, 
	seed::Int = 1, 
	norm::Bool = false
)
	# Normalization function (identity if norm=false)
	f_norm(x) = norm ? Flux.normalise(x) : identity(x)

	# Initialize random number generator and weight initializer
	rng = MersenneTwister(seed)
	init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = 0.01)

	# ═══════════════════════════════════════════════════════════
	# ENCODER: Recurrent Layer
	# ═══════════════════════════════════════════════════════════
	# Produces 2*h_representation outputs (for μ and σ²)
	# Example: if h_representation=128, output dimension is 256
	encoder_layer = recurrent_layer(size_features(lt) => 2*h_representation)

	# ═══════════════════════════════════════════════════════════
	# DECODER: Multi-Layer Perceptron
	# ═══════════════════════════════════════════════════════════
	
	# Input layer: h_representation → first hidden layer
	# Takes sample z (dimension h_representation)
	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	
	# Intermediate hidden layers
	# Built according to h_decoder specification
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) 
	                    for i in 1:length(h_decoder)-1]
	
	# Output layer: last hidden layer → 1
	# Softplus activation to ensure positive output
	o_decoder_layers = Dense(h_decoder[end] => size_output(lt), softplus; init)
	
	# Assemble decoder into a Chain
	decoder = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	# ═══════════════════════════════════════════════════════════
	# COMPLETE MODEL: Vector [Encoder, Decoder]
	# ═══════════════════════════════════════════════════════════
	# NOTE: Unlike RnnTModel which uses a single Chain,
	# here we create a vector of two separate components
	# - model[1]: Encoder (with optional normalization)
	# - model[2]: Decoder
	model = Chain(f_norm, encoder_layer, decoder)

	# Create and return RnnTModelSampleInside
	# Default: sample=true (variational sampling enabled)
	return RnnTModelSampleInside(model, rng, true)
end