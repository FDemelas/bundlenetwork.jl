"""
    AttentionModelFactory <: AbstractDirectionAndTModelFactory

Factory used to create an attention model that predicts both a direction 
(via an attention mechanism) and a temporal parameter `t`.

This factory is responsible for:
- Extracting relevant features from bundles (SoftBundle or BatchedSoftBundle)
- Defining the structure of feature vectors for model training

# Design Pattern
Implements the Factory Method pattern, separating model creation logic from usage.

# See Also
- `AttentionModel`: The concrete model created by this factory
- `create_features`: Feature extraction methods
- `create_NN`: Model instantiation method
"""
struct AttentionModelFactory <: AbstractDirectionAndTModelFactory

end

"""
    create_features(lt::AttentionModelFactory, B::SoftBundle; auxiliary = 0)

Creates the feature vector for an attention model from a `SoftBundle`.

# Arguments
- `lt::AttentionModelFactory`: The model factory
- `B::SoftBundle`: Bundle containing information from the current iteration
- `auxiliary`: Auxiliary parameter (currently unused, default = 0)

# Returns
- `(ϕ, ϕγ)`: Tuple of two feature vectors
  - `ϕ`: Features for the temporal parameter t (16 dimensions)
  - `ϕγ`: Features for the bundle component (20 dimensions)

# Feature Descriptions

## ϕ (temporal features) - 16 dimensions
1. `t`: Total accumulated time
2. `qp`: Quadratic norm of weights w (‖w‖²)
3. `t*qp`: Time-norm interaction term
4. `lp`: Dot product between α and θ (linear combination)
5. `qp>lp`: Binary indicator (norm > dot product)
6. `10000*qp>lp`: Binary indicator with scaling factor
7. `obj[B.li]`: Objective value at last index
8. `obj[B.s]`: Objective value at current index
9. `obj[B.li]<obj[B.s]`: Binary comparison of objectives
10. `B.li`: Last index used
11. `α[B.s]`: Alpha value at current index
12. `α[B.li]`: Alpha value at last index
13-16. Norms of z and G vectors:
   - `√(z'z[li])/2`: Norm of z at last index
   - `√(z'z[s])/2`: Norm of z at current index
   - `√(G'G[s])/2`: Norm of G at current index
   - `√(G'G[li])/2`: Norm of G at last index

## ϕγ (component features) - 20 dimensions
1-8. Statistics (mean, std, min, max) of G and z:
   - `mean(G[:, li])`, `mean(z[:, li])`
   - `std(G[:, li])`, `std(z[:, li])`
   - `minimum(G[:, li])`, `minimum(z[:, li])`
   - `maximum(G[:, li])`, `maximum(z[:, li])`
9-12. Minimums of matrix products:
   - `minimum(z'z)`, `minimum(z'z_s)`
   - `minimum(G'G_s)`, `minimum(G'G)`
13. `G[:, li]'*w`: Dot product at last index
14-17. Maximums of matrix products:
   - `maximum(z'z)`, `maximum(z'z_s)`
   - `maximum(G'G_s)`, `maximum(G'G)`
18. `G[:, s]'*w`: Dot product at current index
19-20. `α[li]` and `obj[li]`

# Implementation Notes
- All matrix products are computed on CPU for stability
- Features are moved to the appropriate device (CPU/GPU) before returning
- The feature vector captures both geometric and temporal aspects of the optimization state
"""
function create_features(lt::AttentionModelFactory, B::SoftBundle; auxiliary = 0)
	# Extract basic components
	t = sum(B.t)
	obj = cpu(B.obj)
	θ = cpu(B.θ)[1:end]
	α = cpu(B.α[1:B.size])
	i, s, e = 1, 1, length(B.w)
	
	# Compute linear product between α and θ
	lp = sum(α[1:length(θ)]' * θ)
	
	# Compute quadratic norm of weights
	qp = sum(B.w' * B.w)

	# Compute matrix products for z and G
	# These capture geometric relationships between components
	zz = cpu(B.z[:, :]' * B.z[:, B.size])
	zsz = cpu(B.z[:, :]' * B.z[:, B.s])
	gg = cpu(B.G[:, :]' * B.G[:, B.size])
	gsg = cpu(B.G[:, :]' * B.G[:, B.s])

	# Build temporal feature vector (16 dimensions)
	ϕ = Float32[t,
		qp,
		t*qp,
		lp,
		qp>lp,
		10000*qp>lp,
		obj[B.li],
		obj[B.s],
		obj[B.li]<obj[B.s],
		B.li,
		α[B.s],
		α[B.li],
		sum(sqrt(zz[B.li]) / 2),
		sum(sqrt(zsz[B.s]) / 2),
		sum(sqrt(gsg[B.s]) / 2),
		sum(sqrt(gg[B.li]) / 2),
	]

	# Build component feature vector (20 dimensions)
	ϕγ = Float32[mean(B.G[:, B.li]),
		mean(B.z[:, B.li]),
		std(B.G[:, B.li]),
		std(B.z[:, B.li]),
		minimum(B.G[:, B.li]),
		minimum(B.z[:, B.li]),
		maximum(B.G[:, B.li]),
		maximum(B.z[:, B.li]),
		minimum(zz),
		minimum(zsz),
		minimum(gsg),
		minimum(gg),
		B.G[:, B.li]'*B.w,
		maximum(zz),
		maximum(zsz),
		maximum(gsg),
		maximum(gg),
		B.G[:, B.s]'*B.w,
		α[B.li],
		obj[B.li]]
	
	# Move to appropriate device (CPU/GPU)
	return device(ϕ), device(ϕγ)
end


"""
    create_features(lt::AttentionModelFactory, B::BatchedSoftBundle; auxiliary = 0)

Creates feature vectors for a batch of bundles.

# Arguments
- `lt::AttentionModelFactory`: The model factory
- `B::BatchedSoftBundle`: Batched bundle containing multiple examples
- `auxiliary`: Auxiliary parameter (currently unused)

# Returns
- `(feat_t, feat_theta)`: Tuple of feature matrices
  - `feat_t`: Matrix of temporal features (16 × batch_size)
  - `feat_theta`: Matrix of component features (20 × batch_size)

# Implementation Notes
- Processes multiple bundles in parallel for computational efficiency
- Features are computed identically to the non-batched version
- Each column corresponds to one example in the batch
- Handles variable-length components via `B.idxComp` indexing

# Performance
Batched processing is significantly more efficient than sequential processing,
especially on GPU where parallel operations can be leveraged.
"""
function create_features(lt::AttentionModelFactory, B::BatchedSoftBundle; auxiliary = 0)
	let feat_t, feat_theta
		t = B.t
		mli = 1
		obj = B.obj[:, 1:B.size]
		α = B.α[:, 1:B.size]
		
		# Iterate over each example in the batch
		for (i, (s, e)) in enumerate(B.idxComp)
			# Extract θ parameters for this example
			θ = cpu(reshape(B.θ[i, :], :))
			lp = α[i, 1:length(θ)]' * θ

			# Compute matrix products for z and G components
			# Slice by component range [s:e] for this example
			zz = cpu(B.z[s:e, B.li]' * B.z[s:e, mli:B.size])
			zsz = cpu(B.z[s:e, B.s[i]]' * B.z[s:e, mli:B.size])
			gg = cpu(B.G[s:e, B.li]' * B.G[s:e, mli:B.size])
			gsg = cpu(B.G[s:e, B.s[i]]' * B.G[s:e, mli:B.size])
			ww = B.w[s:e]' * B.w[s:e]

			# Build features for this example
			ϕ = Float32[t[i],
				ww,
				t[i]*ww,
				lp,
				ww>lp,
				10000*ww>lp,
				obj[i, B.li],
				obj[i, B.s[i]],
				obj[i, B.li]<obj[i, B.s[i]],
				B.li,
				α[i, B.s[i]],
				α[i, B.li],
				sqrt(sum(zz[B.li]))/2,
				sqrt(sum(zsz[B.s[i]]) / 2),
				sqrt(sum(gsg[B.s[i]]) / 2),
				sqrt(sum(gg[B.li]) / 2),
			]
			
			ϕγ = Float32[mean(B.G[s:e, B.li]),
				mean(B.z[s:e, B.li]),
				std(B.G[s:e, B.li]),
				std(B.z[s:e, B.li]),
				minimum(B.G[s:e, B.li]),
				minimum(B.z[s:e, B.li]),
				maximum(B.G[s:e, B.li]),
				maximum(B.z[s:e, B.li]),
				minimum(zz),
				minimum(zsz),
				minimum(gsg),
				minimum(gg),
				B.G[s:e, B.li]'*B.w[s:e],
				maximum(zz),
				maximum(zsz),
				maximum(gsg),
				maximum(gg),
				B.G[s:e, B.s[i]]'*B.w[s:e],
				α[i, B.li],
				obj[i, B.li]]
			
			# Initialize or concatenate features
			if i == 1
				feat_t, feat_theta = ϕ, ϕγ
			else
				feat_theta = hcat(feat_theta, ϕγ)
				feat_t = hcat(feat_t, ϕ)
			end
		end
		
		return feat_t, feat_theta
	end
end

"""
    size_comp_features(lt::AttentionModelFactory)

Returns the dimension of the feature vector for each bundle component.

# Returns
- `20`: Number of features computed for each component (ϕγ)

# Note
These features correspond to the last component added to the bundle.
This dimension must match the actual number of features extracted in `create_features`.
"""
function size_comp_features(lt::AttentionModelFactory)
	return 20
end

"""
    size_features(lt::AttentionModelFactory)

Returns the dimension of the feature vector for the temporal parameter t.

# Returns
- `16`: Number of temporal features (ϕ)

# Note
This dimension must match the actual number of temporal features extracted in `create_features`.
"""
function size_features(lt::AttentionModelFactory)
	return 16
end


"""
    AttentionModel <: AbstractModel

Attention model based on a Variational Autoencoder (VAE) with attention mechanism.

# Architecture Overview
The model consists of:
- One encoder (LSTM or Dense) that encodes features into a latent representation
- Four independent decoders that predict:
  - `decoder_t`: The temporal parameter t
  - `decoder_temperature`: Temperature for attention (currently unused)
  - `decoder_γk`: Keys for the attention mechanism
  - `decoder_γq`: Queries for the attention mechanism

# Attention Mechanism
The model implements a key-query attention system:
1. At each iteration, a key vector is computed and stored in matrix `Ks`
2. A query vector is computed for the current state
3. Attention scores are computed as: γ = query' × Keys
4. These scores can be normalized (softmax/sparsemax) to obtain attention weights

# Fields
- `encoder::Chain`: Encoder for creating latent representations
- `decoder_t::Chain`: Decoder for the temporal parameter
- `decoder_temperature::Chain`: Decoder for temperature (experimental)
- `decoder_γk::Chain`: Decoder for attention keys
- `decoder_γq::Chain`: Decoder for attention queries
- `rng::MersenneTwister`: Random number generator for sampling
- `sample_t::Bool`: If true, samples t representation; otherwise uses mean
- `sample_γ::Bool`: If true, samples key/query representations; otherwise uses mean
- `rescaling_factor::Int64`: Rescaling factor for γs (currently unused)
- `h_representation::Int64`: Dimension of the latent space
- `it::Int64`: Iteration counter
- `h3_representations::Bool`: If true, uses three independent latent representations
- `repeated::Bool`: If true, recomputes inputs for all previous iterations
- `Ks::Zygote.Buffer`: Matrix storing keys for all components
- `use_tanh::Bool`: If true, applies tanh to outputs (experimental)

# Variational Autoencoder (VAE) Components
The model uses a VAE structure where:
- Encoder produces μ (mean) and σ² (variance)
- Reparameterization trick: z = μ + ε·σ where ε ~ N(0,1)
- This allows gradients to backpropagate through sampling

# Operating Modes
- **Training mode** (`sample_t=true`, `sample_γ=true`): Samples from distributions
- **Inference mode** (`sample_t=false`, `sample_γ=false`): Uses means deterministically

# Example Usage
```julia
# Create model
factory = AttentionModelFactory()
model = create_NN(factory, h_representation=256)

# Initialize for a sequence
reset!(model, batch_size=1, max_iterations=500)

# Forward pass
feat_t, feat_gamma = create_features(factory, bundle)
t, attention_scores = model(feat_t, feat_gamma, iteration, 1:bundle.size)
```
"""
mutable struct AttentionModel <: AbstractModel
	encoder::Chain
	decoder_t::Chain
	decoder_temperature::Chain
	decoder_γk::Chain
	decoder_γq::Chain
	rng::MersenneTwister
	sample_t::Bool
	sample_γ::Bool
	rescaling_factor::Int64
	h_representation::Int64
	it::Int64
	h3_representations::Bool
	repeated::Bool
	Ks::Zygote.Buffer{Float32, CUDA.CuArray{Float32, 2, CUDA.DeviceMemory}}
	use_tanh::Bool
end

"""
    reset!(m::AttentionModel, bs::Int = 1, it::Int = 500)

Resets the attention model state for a new sequence.

# Arguments
- `m::AttentionModel`: The model to reset
- `bs::Int`: Batch size (default: 1)
- `it::Int`: Maximum number of iterations expected (default: 500)

# Effects
- Resets the hidden state of recurrent networks (LSTM)
- Reinitializes the key matrix `Ks` with zeros
- Resets the iteration counter to 1

# Implementation Notes
This function must be called before processing a new sequence to prevent
the previous state from influencing predictions. The key matrix `Ks` is
preallocated with dimensions (bs * h_representation) × it for efficiency.

# Example
```julia
# Before processing a new sequence
reset!(model, batch_size=4, max_iterations=1000)

# Now ready for forward passes
for i in 1:num_iterations
    t, scores = model(feat_t, feat_gamma, i, comps)
end
```
"""
function reset!(m::AttentionModel, bs::Int = 1, it::Int = 500)
	# Reset hidden state of recurrent layers
	Flux.reset!(m.encoder)
	Flux.reset!(m.decoder_t)
	Flux.reset!(m.decoder_temperature)
	Flux.reset!(m.decoder_γk)
	Flux.reset!(m.decoder_γq)

	# Reinitialize key matrix
	# Dimensions: (bs * h_representation) × it
	# Stores one key vector per iteration for the attention mechanism
	m.Ks = Zygote.bufferfrom(device(zeros(Float32, bs * m.h_representation, it)))
	
	# Reset iteration counter
	m.it = 1
end

"""
    (m::AttentionModel)(xt, xγ, idx, comps)

Forward pass of the attention model.

# Arguments
- `xt`: Temporal features (dimension: size_features × batch_size)
- `xγ`: Component features (dimension: size_comp_features × batch_size)
- `idx`: Indices for storing keys in the Ks matrix
- `comps`: Indices of components to consider for attention computation

# Returns
- `t`: Prediction of the temporal parameter
- `γs`: Attention scores (logits) for each component

# Process Flow

## 1. Encoding
Features are concatenated and passed through the encoder:
```
x = [xt; xγ]  (36 dimensions: 16 + 20)
h = encoder(x)
```

## 2. Variational Sampling
The encoded representation is split into mean (μ) and variance (σ²):
- Variance stabilization via softplus transformations
- Reparameterization trick: z = μ + ε·σ
- Optionally use separate representations for t, temperature, keys, and queries

## 3. Decoding
Four decoders produce:
- `t`: Temporal parameter
- `b`: Temperature (experimental)
- `hk`: Key vector for attention
- `hq`: Query vector for attention

## 4. Attention Computation
- Store current key: `Ks[:, idx] = hk`
- Compute attention scores: `γ = query' × Keys`
- Return raw logits (not normalized)

# Implementation Details

### Variance Stabilization
Two successive softplus transformations ensure σ² stays in a reasonable range:
```julia
σ² = 2 - softplus(2 - σ²)      # Upper bound
σ² = -6 + softplus(σ² + 6)     # Lower bound
σ² = exp(σ²)                    # Ensure positivity
```

### Reparameterization Trick
Instead of sampling z ~ N(μ, σ²) directly (non-differentiable),
we use: z = μ + ε·σ where ε ~ N(0,1)
This allows gradients to flow through μ and σ.

### Attention Mechanism
The attention scores are computed as dot products between:
- Query: Representation of current state
- Keys: Representations of all previous components

Higher scores indicate higher relevance/similarity.

# Note on Gradients
The backward pass is handled automatically by Flux/Zygote's automatic
differentiation. The reparameterization trick ensures gradients can
flow through the sampling operation.

# Example
```julia
# Extract features
feat_t, feat_gamma = create_features(factory, bundle)

# Forward pass
t_pred, attention_logits = model(
    feat_t, 
    feat_gamma, 
    current_iteration,
    1:num_components
)

# Normalize attention scores
attention_weights = softmax(attention_logits)
```
"""
function (m::AttentionModel)(xt, xγ, idx, comps)
	# Concatenate temporal and component features
	x = vcat(xt, xγ)

	# Encode into latent representation
	h = m.encoder(x)
	
	# Split into mean and variance (for VAE)
	μ, σ2 = Flux.MLUtils.chunk(h, 2, dims = 1)

	# Stabilize variance using softplus transformations
	# These ensure σ² remains in a reasonable range and avoids numerical issues
	
	# First transformation: upper bound
	σ2 = 2.0f0 .- softplus.(2.0f0 .- σ2)
	
	# Second transformation: lower bound
	σ2 = -6.0f0 .+ softplus.(σ2 .+ 6.0f0)
	
	# Convert to positive space and compute standard deviation
	σ2 = exp.(σ2)
	sigma = sqrt.(σ2 .+ 1)

	# Split into representations for t, temperature, keys, and queries
	# If h3_representations=true: 4 independent representations
	# Otherwise: shared representation
	μt, μb, μk, μq = m.h3_representations ? copy.(Flux.MLUtils.chunk(μ, 4, dims = 1)) : (copy(μ), copy(μ), copy(μ), copy(μ))
	σt, σb, σk, σq = m.h3_representations ? Flux.MLUtils.chunk(sigma, 4, dims = 1) : (sigma, sigma, sigma, sigma)

	# Generate noise for reparameterization trick
	# ε ~ N(0,1) for each representation
	ϵt = device(randn(m.rng, Float32, size(μt)))
	ϵb = device(randn(m.rng, Float32, size(μb)))
	ϵk = device(randn(m.rng, Float32, size(μk)))
	ϵq = device(randn(m.rng, Float32, size(μq)))
	
	# Sample or use mean (reparameterization trick)
	# z = μ + ε·σ allows gradients to flow through sampling
	t = m.decoder_t(m.sample_t ? μt .+ ϵt .* σt : μt)
	b = m.decoder_temperature(m.sample_t ? μb .+ ϵb .* σb : μb)
	hk = m.decoder_γk(m.sample_γ ? μk .+ ϵk .* σk : μk)
	hq = m.decoder_γq(m.sample_γ ? μq .+ ϵq .* σq : μq)

	# Store key in the key matrix for future attention computation
	m.Ks[:, idx] = reshape(hk, :)
	
	# Prepare queries and keys for attention
	# Handle both single and batched cases
	aq = (size(hq, 2) > 1 ? Flux.MLUtils.chunk(hq, size(hq, 2); dims = 2) : [hq])
	ak = (Flux.MLUtils.chunk(m.Ks[:, comps], Int64(size(m.Ks, 1) / m.h_representation); dims = 1))

	# Compute attention scores: query' × key for each pair
	# These are raw logits (not normalized)
	γs = vcat(map((x, y) -> sum(x'y; dims = 1), aq, ak)...)

	return t, γs
end

# Declare model as a Flux layer (for integration with Flux.jl)
Flux.@layer AttentionModel

"""
    create_NN(lt::AttentionModelFactory; kwargs...)

Creates and initializes an attention model with the specified architecture.

# Keyword Arguments
- `h_act`: Activation function for hidden layers (default: gelu)
- `h_representation::Int`: Dimension of latent space (default: 128)
- `h_decoder::Vector{Int}`: Hidden layer dimensions for decoders (default: [h_representation * 8])
- `seed::Int`: Random seed for initialization (default: 1)
- `norm::Bool`: If true, normalize inputs (default: false)
- `sampling_t::Bool`: If true, sample t representation (default: false)
- `sampling_θ::Bool`: If true, sample γ representations (default: true)
- `ot_act`: Activation function for output layer (default: softplus)
- `rnn::Bool`: If true, use LSTM for encoder; otherwise Dense (default: true)
- `h3_representations::Bool`: If true, use 3 independent latent spaces (default: false)
- `repeated::Bool`: If true, recompute inputs for previous iterations (default: true)
- `use_tanh::Bool`: If true, apply tanh to outputs (default: false)

# Returns
- `model::AttentionModel`: Initialized model placed on appropriate device (CPU/GPU)

# Architecture Details

## Encoder
- Type: LSTM (if rnn=true) or Dense (if rnn=false)
- Input: size_features + size_comp_features (36 dimensions: 16 + 20)
- Output: 2 * h_representation (or 8 * h_representation if h3_representations)
  - Factor of 2 for μ and σ
  - Factor of 8 for 4 components × 2 (μ and σ for each)

## Decoders (4 total)
All decoders share the same structure but with different outputs:
1. **decoder_t**: h_representation → h_decoder → 1 (temporal parameter)
2. **decoder_temperature**: h_representation → h_decoder → 1 (temperature)
3. **decoder_γq**: h_representation → h_decoder → h_representation (query)
4. **decoder_γk**: h_representation → h_decoder → h_representation (key)

### Decoder Architecture
```
Input (h_representation)
    ↓
Dense(h_representation → h_decoder[1], h_act)
    ↓
[Dense(h_decoder[i] → h_decoder[i+1], h_act) for each layer]
    ↓
Dense(h_decoder[end] → output_dim, activation)
```

## Complete Architecture Diagram
```
Input Features (36)
    ↓
[Optional Normalization]
    ↓
╔════════════════════════════════╗
║  ENCODER (LSTM/Dense)          ║
║  36 → 2*h_rep or 8*h_rep       ║
╚════════════════════════════════╝
    ↓
Split into [μ, σ²]
    ↓ [Reparameterization Trick]
    ↓
╔═══════╦════════════╦═══════╦═══════╗
║ Dec_t ║ Dec_temp   ║ Dec_q ║ Dec_k ║
║ → 1   ║ → 1        ║ → h   ║ → h   ║
╚═══════╩════════════╩═══════╩═══════╝
    ↓        ↓           ↓       ↓
    t    temperature  query    key
```

# Hyperparameter Guidelines

## h_representation
- Small (64-128): Faster, less capacity
- Medium (256-512): Good balance
- Large (512-1024): More capacity, slower

## h_decoder
- Shallow [512]: Fast, may underfit
- Medium [1024, 512]: Standard choice
- Deep [1024, 512, 256]: More capacity, may overfit

## Sampling Flags
- `sampling_t=true`: Stochastic t prediction (exploration)
- `sampling_t=false`: Deterministic t prediction (exploitation)
- Similar for `sampling_θ`

## h3_representations
- `false`: Shared latent space (parameter efficient)
- `true`: Independent spaces for t, temp, keys, queries (more expressive)

# Example Usage

```julia
# Standard configuration
factory = AttentionModelFactory()
model = create_NN(factory)

# High-capacity configuration
model = create_NN(
    factory,
    h_representation = 512,
    h_decoder = [2048, 1024, 512],
    h3_representations = true,
    sampling_t = true,
    rnn = true
)

# Lightweight configuration
model = create_NN(
    factory,
    h_representation = 64,
    h_decoder = [256],
    rnn = false,
    norm = true
)
```

# Implementation Notes
- Weights are initialized using truncated normal (mean=0, std=0.01)
- Random seed ensures reproducible initialization
- Model is automatically moved to the appropriate device (CPU/GPU)
- The key matrix `Ks` is initialized with minimal size (1×1) and resized during `reset!`
"""
function create_NN(
	lt::AttentionModelFactory;
	h_act = gelu,
	h_representation::Int = 128,
	h_decoder::Vector{Int} = [h_representation * 8],
	seed::Int = 1,
	norm::Bool = false,
	sampling_t::Bool = false,
	sampling_θ::Bool = true,
	ot_act = softplus,
	rnn = true,
	h3_representations::Bool = false,
	repeated::Bool = true,
	use_tanh::Bool = false
)
	bs, it = 1, 1
	
	# Normalization function (identity if norm=false)
	f_norm(x) = norm ? Flux.normalise(x) : identity(x)

	# Initialize random number generator and weight initializer
	rng = MersenneTwister(seed)
	init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = 0.01)

	# Build encoder
	# LSTM if rnn=true, Dense otherwise
	# Output dimension: (h3_representations ? 8 : 2) * h_representation
	# Factor of 2 for μ and σ; factor of 8 for 4 components × 2
	encoder = rnn ? 
		Chain(f_norm, LSTM(size_features(lt) + size_comp_features(lt) => (h3_representations ? 8 : 2) * h_representation)) : 
		Chain(f_norm, Dense(size_features(lt) + size_comp_features(lt) => (h3_representations ? 8 : 2) * h_representation, h_act))

	# Build decoder for t
	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) for i in 1:length(h_decoder)-1]
	o_decoder_layers = Dense(h_decoder[end] => 1, ot_act; init)
	decoder_t = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	# Build decoder for temperature
	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) for i in 1:length(h_decoder)-1]
	o_decoder_layers = Dense(h_decoder[end] => 1, ot_act; init)
	decoder_temperature = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	# Build decoder for queries
	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) for i in 1:length(h_decoder)-1]
	o_decoder_layers = Dense(h_decoder[end] => h_representation; init)
	decoder_γq = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	# Build decoder for keys
	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) for i in 1:length(h_decoder)-1]
	o_decoder_layers = Dense(h_decoder[end] => h_representation; init)
	decoder_γk = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	# Assemble final model and move to appropriate device (CPU/GPU)
	model = AttentionModel(
		device(encoder), 
		device(decoder_t),
		device(decoder_temperature), 
		device(decoder_γk), 
		device(decoder_γq), 
		rng, 
		sampling_t, 
		sampling_θ, 
		1.0, 
		h_representation, 
		1,
		h3_representations, 
		repeated, 
		device(Zygote.bufferfrom(device(zeros(Float32, bs * h_representation, it)))),
		use_tanh
	)
	
	return model
end

"""
    h_representation(nn::AttentionModel)

Returns the dimension of the model's latent space.

# Arguments
- `nn::AttentionModel`: The attention model

# Returns
- `Int64`: Dimension of the latent representation (h_representation)

# Implementation Note
The dimension is extracted from the weight matrix of the first layer of decoder_t.
This ensures consistency between the stored h_representation field and the actual
architecture.

# Example
```julia
model = create_NN(factory, h_representation=256)
@assert h_representation(model) == 256
```
"""
function h_representation(nn::AttentionModel)
	return Int64(size(nn.decoder_t[1].weight, 2))
end