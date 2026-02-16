"""
    RnnTModel <: AbstractTModel

Recurrent Neural Network (RNN) model for predicting the temporal parameter `t` 
with optional variational sampling.

# Architecture
The model uses an encoder-decoder architecture:
- **Encoder**: Recurrent layer (LSTM, GRU, etc.) that processes features sequentially
- **Decoder**: MLP that produces mean (μ) and variance (σ²) of a Gaussian distribution
- **Sampling**: Reparameterization trick to sample t ~ N(μ, σ²)

# Fields
- `model::Any`: Neural network (Flux.jl Chain)
- `rng::Any`: Random number generator (MersenneTwister)
- `sample::Bool`: If true, uses variational sampling; otherwise uses μ directly
- `ϵs::AbstractVector`: Vector storing random components ϵ ~ N(0,1) during training
- `deviation::AbstractDeviation`: Structure defining how to apply deviation to t (see deviations.jl)
- `train_mode::Bool`: If true, training mode (saves ϵ); if false, inference mode

# Operation Flow
1. Features φ are passed to the recurrent encoder
2. Decoder produces μ and σ² for distribution N(μ, σ²)
3. If sample=true: samples t = μ + ϵ·σ with ϵ ~ N(0,1)
4. If sample=false: uses t = μ directly
5. Applies deviation according to chosen strategy
6. Constrains t to interval [t_min, t_max]

# Example
```julia
# Create model with sampling
rng = MersenneTwister(42)
model_chain = Chain(LSTM(34 => 128), Dense(128 => 2))
rnn_model = RnnTModel(
    model_chain, 
    rng, 
    true,  # sampling enabled
    [],    # ϵs empty at start
    NothingDeviation(),  # no special deviation
    true   # training mode
)
```

# See Also
- `RnnTModelSampleInside`: Variant with post-sampling transformation
- `AbstractDeviation`: For different deviation strategies
"""
mutable struct RnnTModel <: AbstractTModel
	model::Any
	rng::Any
	sample::Bool
	ϵs::AbstractVector
	deviation::AbstractDeviation
	train_mode::Bool
	
	# Constructor with default values
	RnnTModel(model, rng, sample = false, es = [], deviation = NothingDeviation(), train_mode = true) = 
		new(model, rng, sample, es, deviation, train_mode)
end

"""
    RnnTModelfactory <: AbstractTModelFactory

Factory for creating RnnTModel instances with standardized configuration.

This structure without fields serves as a type marker for multiple dispatch
and allows centralized creation and configuration of RnnTModel instances.

# Design Pattern
Implements the Factory Method pattern, separating model creation from usage.

# See Also
- `create_NN`: Creates configured RnnTModel instances
- `create_features`: Extracts features for the model
"""
struct RnnTModelfactory <: AbstractTModelFactory end

"""
    create_features(B::DualBundle, _::RnnTModel)

Creates the feature vector from a DualBundle for an RnnTModel.

# Arguments
- `B::DualBundle`: Bundle containing all information from current iteration
- `_::RnnTModel`: Model instance (used only for dispatch)

# Returns
- Matrix of dimensions (34, 1) containing features extracted from bundle

# Implementation Note
This function wraps `features_vector_i` and reshapes the result to match
the format expected by the model (batch_size = 1).

# Example
```julia
features = create_features(bundle, model)
t_pred = model(features, bundle)
```
"""
function create_features(B::DualBundle, _::RnnTModel)
	# Extract features from current iteration
	φ = features_vector_i(B)
	
	# Reshape to matrix format (features, batch_size)
	# Here batch_size = 1 since we process a single bundle
	return reshape(φ, (length(φ), 1))
end

"""
    features_vector_i(B::DualBundle)

Extracts and computes the 34 features characterizing the current bundle state.

# Arguments
- `B::DualBundle`: Bundle containing optimization history and current state

# Returns
- Vector of 34 Float32 values containing the features

# Feature Descriptions
The 34 features are organized into several categories:

## Temporal and Progression Features (1-6)
1. `t`: Total accumulated time
2. `qp`: Quadratic norm of weights w (‖w‖²)
3. `t*qp`: Time-norm interaction term
4. `lp`: Dot product α'θ (linear combination)
5. `qp>lp`: Binary indicator (norm > dot product)
6. `10000*qp>lp`: Binary indicator with scale factor

## Objective Features (7-9)
7. `realObj`: Real objective value at last index
8. `sObj`: Objective value at current index s
9. `realObj<sObj`: Binary comparison indicator

## Index and Weight Features (10-12)
10. `B.li`: Last index used
11. `0.0`: Reserved feature (currently unused)
12. `α[B.li]`: Alpha value at last index

## Norms of z and G Vectors (13-16)
13. Norm of z at last index: √(z'z[li])/2
14. Norm of z at current index: √(z'z[s])/2
15. Norm of G at current index: √(G'G[s])/2
16. Norm of G at last index: √(G'G[li])/2

## Statistics on G at Last Index (17-24)
17. `mean(G[:, li])`: Mean of G
18. `mean(z[:, li])`: Mean of z
19. `std(G[:, li])`: Standard deviation of G
20. `std(z[:, li])`: Standard deviation of z
21. `minimum(G[:, li])`: Minimum of G
22. `minimum(z[:, li])`: Minimum of z
23. `maximum(G[:, li])`: Maximum of G
24. `maximum(z[:, li])`: Maximum of z

## Minimums and Maximums of Matrix Products (25-33)
25-28. Minimums of zz, zsz, gsg, gg
29. Dot product G[:,li]'w
30-33. Maximums of zz, zsz, gsg, gg

## Final Feature (34)
34. `G[:,s]'w`: Dot product at current index

# Interpretation
These features capture different aspects of the optimization state:
- Geometry (norms, dot products)
- Temporal progression
- Relationships between bundle components

# Implementation Notes
All features are computed as Float32 for memory efficiency and GPU compatibility.
The vector is reshaped to column format for consistency with matrix operations.
"""
function features_vector_i(B::DualBundle)
	# Total accumulated time
	t = sum(B.params.t)
	
	# Extract bundle components
	obj = B.obj
	θ = B.θ[1:end]
	α = B.α[1:min(length(B.θ), B.size)]
	i, s, e = 1, 1, length(B.w)
	
	# Compute linear dot product and quadratic norm
	lp = sum(α[1:length(θ)]' * θ)
	qp = sum(B.w' * B.w)

	# Compute matrix products for z and G
	# These matrices capture relationships between components
	zz = B.z[:, :]' * B.z[:, B.li]
	zsz = B.z[:, :]' * B.z[:, B.s]
	gg = B.G[:, :]' * B.G[:, B.li]
	gsg = B.G[:, :]' * B.G[:, B.s]

	# Objective values
	realObj = B.obj[B.li]
	sObj = objS(B)

	# Build feature vector (34 dimensions)
	ϕ = Float32[
		# Temporal and norm features (1-6)
		t,
		qp,
		t*qp,
		lp,
		qp>lp,
		10000*qp>lp,
		
		# Objective features (7-9)
		realObj,
		sObj,
		realObj<sObj,
		
		# Index and weight features (10-12)
		B.li,
		0.0,  # Reserved feature
		B.α[B.li],
		
		# Norms of z and G vectors (13-16)
		sum(sqrt(zz[B.li]) / 2),
		sum(sqrt(zsz[B.s]) / 2),
		sum(sqrt(gsg[B.s]) / 2),
		sum(sqrt(gg[B.li]) / 2),
		
		# Statistics on G and z at last index (17-24)
		mean(B.G[:, B.li]),
		mean(B.z[:, B.li]),
		std(B.G[:, B.li]),
		std(B.z[:, B.li]),
		minimum(B.G[:, B.li]),
		minimum(B.z[:, B.li]),
		maximum(B.G[:, B.li]),
		maximum(B.z[:, B.li]),
		
		# Extrema of matrix products (25-33)
		minimum(zz),
		minimum(zsz),
		minimum(gsg),
		minimum(gg),
		B.G[:, B.li]'*B.w,
		maximum(zz),
		maximum(zsz),
		maximum(gsg),
		maximum(gg),
		
		# Final dot product (34)
		B.G[:, B.s]'*B.w
	]
	
	return reshape(ϕ, :)
end


"""
    size_features(_::RnnTModel)

Returns the dimension of the feature vector for an RnnTModel.

# Arguments
- `_::RnnTModel`: Model instance (used for dispatch)

# Returns
- `34`: Number of features extracted from DualBundle

# Note
This value must match exactly the dimension of the vector returned by `features_vector_i`.
"""
function size_features(_::RnnTModel)
	return 34
end


"""
    size_features(_::RnnTModelfactory)

Returns the dimension of the feature vector for models created by RnnTModelfactory.

# Arguments
- `_::RnnTModelfactory`: Model factory (used for dispatch)

# Returns
- `34`: Number of features extracted from DualBundle
"""
function size_features(_::RnnTModelfactory)
	return 34
end

"""
    (m::RnnTModel)(ϕ, B, ϵ = randn(B.nn.rng, Float32, 1))

Forward pass of the RnnTModel.

# Arguments
- `ϕ`: Feature vector (dimension: 34 × 1)
- `B`: Bundle containing parameters and constraints
- `ϵ`: Random vector N(0,1) for reparameterization trick (optional)

# Returns
- `t::Float32`: Predicted temporal parameter value, constrained to [t_min, t_max]

# Detailed Process

## 1. Model Pass
The network produces a vector of dimension 2: [μ, σ²]

## 2. Variance Computation (if sample=true)
Raw variance σ² is stabilized via two successive softplus transformations:
- `σ² = 2 - softplus(2 - σ²)`: Upper bound
- `σ² = -6 + softplus(σ² + 6)`: Lower bound
- `σ² = exp(σ²)`: Convert to positive space
- `σ = √σ² / 100`: Compute standard deviation with scaling

## 3. Sampling (Reparameterization Trick)
If `m.sample = true`:
- Generates ϵ ~ N(0,1)
- Computes dev = μ + ϵ·σ
- Saves ϵ in m.ϵs if in training mode

If `m.sample = false`:
- Uses dev = μ directly

## 4. Deviation Application
Applies deviation strategy: `t = m.deviation(t_current, dev)`
Example strategies:
- NothingDeviation: t = t_current + dev
- MultiplicativeDeviation: t = t_current × (1 + dev)
- etc.

## 5. Interval Constraint
Ensures t ∈ [t_min, t_max]:
`t = min(t_max, |t| + t_min)`

# Reparameterization Trick
Instead of sampling directly from t ~ N(μ, σ²) (non-differentiable),
we use: t = μ + ε·σ where ε ~ N(0,1)

This allows gradients to backpropagate through the sampling operation:
- ∂t/∂μ = 1
- ∂t/∂σ = ε

# Variance Stabilization
The two softplus transformations ensure:
- σ² stays within reasonable bounds
- Numerical stability during training
- Smooth gradient flow

The transformations create a "soft clamping" effect that's differentiable,
unlike hard bounds which would stop gradients.

# Backward Pass Note
The backward pass is handled automatically by Flux/Zygote through automatic
differentiation. The reparameterization trick enables gradient propagation
through the sampling operation.

# Example
```julia
# Forward pass
ϕ = create_features(bundle, model)
t_pred = model(ϕ, bundle)

# With explicit epsilon (for reproducibility)
ϵ = randn(rng, Float32, 1)
t_pred = model(ϕ, bundle, ϵ)
```
"""
function (m::RnnTModel)(ϕ, B, ϵ = randn(B.nn.rng, Float32, 1))
	# Pass through neural network
	f = m.model(ϕ)
	
	# Split into mean (μ) and raw variance (σ²)
	μ, σ2 = Flux.MLUtils.chunk(f, 2, dims = 1)
	
	if m.sample
		# Stabilize variance using softplus transformations
		# These ensure σ² remains in a reasonable range and avoid numerical issues
		
		# First transformation: upper bound
		σ2 = 2.0f0 .- softplus.(2.0f0 .- σ2)
		
		# Second transformation: lower bound
		σ2 = -6.0f0 .+ softplus.(σ2 .+ 6.0f0)
		
		# Convert to positive space and compute standard deviation
		σ2 = exp.(σ2)
		sigma = sqrt.(σ2) / 100  # Division by 100 for scaling
		
		if m.train_mode
			# In training mode, save ϵ for analysis/debugging
			# ignore_derivatives prevents saving from affecting gradients
			ignore_derivatives() do
				push!(m.ϵs, ϵ)
			end
		end
		
		# Reparameterization trick: X ~ N(μ, σ²) ⟺ X = μ + σ·Y where Y ~ N(0,1)
		# This allows gradient propagation through sampling
		dev = (μ .+ ϵ .* sigma)
	else
		# No sampling: use mean directly
		# Useful for deterministic inference
		dev = μ
	end
	
	# Apply deviation strategy to current time
	# The deviation defines how to modify t based on dev
	t = m.deviation(B.params.t, dev)
	
	# Constrain t to interval [t_min, t_max]
	# Absolute value ensures t is positive, then add t_min
	t = min.(B.params.t_max, abs.(t) .+ B.params.t_min)
	
	return mean(t)
end

# Declare model as Flux layer (for automatic differentiation)
Flux.@layer RnnTModel

"""
    size_output(_::RnnTModelfactory)

Returns the dimension of the output of models created by RnnTModelfactory.

# Arguments
- `_::RnnTModelfactory`: Model factory (used for dispatch)

# Returns
- `2`: The model produces 2 values [μ, σ²] for the Gaussian distribution

# Note
These two values represent the parameters of a normal distribution
used to sample or predict the temporal parameter t.
"""
function size_output(_::RnnTModelfactory)
	return 2
end

"""
    create_NN(lt::RnnTModelfactory; kwargs...)

Creates and initializes an RnnTModel with the specified architecture.

# Arguments
- `lt::RnnTModelfactory`: Factory for model creation

# Keyword Arguments
- `recurrent_layer`: Type of recurrent layer to use (default: LSTM)
  Can be LSTM, GRU, or any other Flux recurrent layer
- `h_decoder::Vector{Int}`: Hidden layer dimensions for decoder (default: [512, 256])
- `h_act`: Activation function for hidden layers (default: softplus)
- `h_representation::Int`: Dimension of latent representation (default: 128)
- `seed::Int`: Random seed for initialization (default: 1)
- `norm::Bool`: If true, normalizes inputs (default: false)

# Returns
- `RnnTModel`: Initialized model with sampling enabled by default

# Architecture Created

```
Input (34 features)
    ↓
[Optional Normalization]
    ↓
Recurrent Encoder (LSTM/GRU/...)
    34 → h_representation (128)
    ↓
Decoder MLP
    128 → 512 → 256 → 2
    ↓
Output [μ, σ²]
```

# Architecture Details

## Encoder
- Type: Recurrent layer (LSTM by default)
- Input: 34 features
- Output: h_representation (128)
- Role: Captures temporal dependencies in the sequence

## Decoder
- Type: Multi-Layer Perceptron (MLP)
- Layers: [h_representation] → h_decoder → [2]
- Activations: h_act (softplus) for hidden layers
- Output: Linear (no activation) for μ and σ²

# Hyperparameter Guidelines

## recurrent_layer
- **LSTM**: Good default, handles long-term dependencies well
- **GRU**: Faster than LSTM, fewer parameters, similar performance
- **RNN**: Simplest, may struggle with long sequences

## h_representation
- Small (32-64): Fast, may underfit
- Medium (128-256): Good balance (recommended)
- Large (512-1024): High capacity, slower, may overfit

## h_decoder
- Shallow [256]: Fast, simple patterns only
- Medium [512, 256]: Standard choice (default)
- Deep [1024, 512, 256]: Complex patterns, risk of overfitting

## h_act
- **softplus**: Smooth, always positive (default)
- **relu**: Fast, may have dead neurons
- **gelu**: Modern choice, good performance
- **tanh**: Classic, bounded output

# Example Usage

```julia
# Default configuration
factory = RnnTModelfactory()
model = create_NN(factory)

# Custom LSTM configuration
model = create_NN(
    factory,
    recurrent_layer = LSTM,
    h_decoder = [1024, 512, 256],
    h_representation = 256,
    h_act = gelu,
    seed = 42,
    norm = true
)

# GRU instead of LSTM
model = create_NN(
    factory,
    recurrent_layer = GRU,
    h_decoder = [256, 128],
    h_representation = 64
)

# Minimal configuration
model = create_NN(
    factory,
    h_decoder = [256],
    h_representation = 64,
    norm = false
)
```

# Implementation Notes
- Returned model has `sample=true` by default (variational sampling)
- Initialization uses truncated normal distribution (mean=0, std=0.01)
- Input normalization can improve training stability
- Weights are initialized reproducibly via seed
- Model is ready for training immediately after creation

# Training Tips
1. Start with default hyperparameters
2. Monitor variance σ² during training (should be reasonable, not too small/large)
3. Use `train_mode=true` during training to collect ϵ samples
4. Switch to `sample=false` for deterministic inference
5. Consider normalization (`norm=true`) if features have different scales
"""
function create_NN(
	lt::RnnTModelfactory,
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

	# Build recurrent encoder
	# Takes 34 features as input and produces h_representation values
	encoder_layer = recurrent_layer(size_features(lt) => h_representation)

	# Build MLP decoder
	# Input layer: h_representation → first hidden layer
	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	
	# Intermediate hidden layers
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) 
	                    for i in 1:length(h_decoder)-1]
	
	# Output layer: last hidden layer → 2 (μ and σ²)
	# No activation on output (linear)
	o_decoder_layers = Dense(h_decoder[end] => size_output(lt); init)
	
	# Assemble decoder
	decoder = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	# Build complete model: encoder + decoder
	# Normalization is applied before encoder if norm=true
	model = Chain(f_norm, encoder_layer, decoder)

	# Create and return RnnTModel
	# Default: sample=true (variational sampling enabled)
	return RnnTModel(model, rng, true)
end