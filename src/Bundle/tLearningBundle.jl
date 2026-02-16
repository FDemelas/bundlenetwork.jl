"""
Abstract type for Machine Learning-based Bundle methods that specialize `DualBundle`.

Subtypes of this type still solve the Dual Master Problem to obtain the search
direction, but incorporate a neural network somewhere in the algorithm
(e.g., to predict the step size `t` instead of using a hand-crafted t-strategy).
"""
abstract type AbstractMLBundle <: DualBundle end


"""
DeepBundle structure.

A bundle method variant that solves the classical Dual Master Problem for the
search direction, but replaces the t-strategy with a neural network. At each
iteration, the network receives the current bundle state as features and
directly outputs the regularization/step-size parameter `t`.

This structure also supports an exact gradient computation for the backward pass
via a custom `ChainRulesCore.rrule`, exploiting the KKT conditions of the DMP.

# Fields
- `Q::Matrix{Float32}`: Gram matrix of subgradients, `Q = GᵀG` (size: max_β_size × max_β_size).
- `G::Matrix{Float32}`: Subgradient matrix, one column per bundle component (size: input_dim × max_β_size).
- `z::Matrix{Float32}`: Visited trial points matrix, one column per component (size: input_dim × max_β_size).
- `α::Vector{Float32}`: Linearization error vector, one entry per bundle component.
- `s::Int64`: Index of the current stabilization point column in `G`, `z`, and `obj`.
- `model::Model`: JuMP model representing the Dual Quadratic Master Problem.
- `w::Vector{Any}`: Current search direction (convex combination of bundle subgradients).
- `θ::Vector{Float32}`: Current DMP solution (simplex weights over bundle components).
- `objB::Float32`: Scaled DMP objective value (used in SS/NS test and t-strategies).
- `obj::Vector{Float32}`: Objective function values at stored bundle points.
- `cumulative_θ::Vector{Vector{Float32}}`: History of DMP solutions (used in `remove_outdated`).
- `params::BundleParameters`: Hyperparameters controlling the bundle behavior (e.g., `t`, `maxIt`, `m1`).
- `nn`: Neural network model used to predict the step size `t`.
- `lt::AbstractTModelFactory`: Factory used to create features for the neural network.
- `back_prop_idx::Any`: Indices of iterations selected for backpropagation (Serious Steps or all).
- `ws::Any`: History of search directions across iterations (used in the backward pass).
- `θ2s::Any`: History of the KKT-based correction vectors `θ2` (used for exact gradient computation).
- `features::Any`: History of input feature vectors fed to the neural network at each iteration.
- `ts::Any`: History of predicted step sizes `t` across iterations.
- `CSS::Int64`: Counter for consecutive Serious Steps.
- `CNS::Int64`: Counter for consecutive Null Steps.
- `ϕ0::AbstractVector`: Instance-level features used as global context for the neural network.
- `size::Int`: Current number of active bundle components.
- `all_objs::Vector{Float32}`: Full history of objective values at all visited points.
- `exactGrad::Bool`: If `true`, use the KKT-based correction in the backward pass for a better ∂w/∂t.
- `li::Int`: Index of the last inserted bundle component.
- `memorized::Dict`: Dictionary for storing per-iteration diagnostic data (e.g., iteration times).
- `vStar::Float32`: Current DMP objective value (used in SS/NS decision).
- `ϵ::Float32`: DMP objective at the reference step size `t_star` (used in stopping criterion).
- `linear_part::Float32`: Linear part `αᵀθ` of the DMP objective.
- `quadratic_part::Float32`: Quadratic part `‖w‖²` of the DMP objective.
"""
mutable struct DeepBundle <: AbstractMLBundle
    Q::Matrix{Float32}
    G::Matrix{Float32}
    z::Matrix{Float32}
    α::Vector{Float32}
    s::Int64
    model::Model
    w::Vector{Any}
    θ::Vector{Float32}
    objB::Float32
    obj::Vector{Float32}
    cumulative_θ::Vector{Vector{Float32}}
    params::BundleParameters
    nn::Any
    lt::AbstractTModelFactory
    back_prop_idx::Any
    ws::Any
    θ2s::Any
    features::Any
    ts::Any
    CSS::Int64
    CNS::Int64
    ϕ0::AbstractVector
    size::Int
    all_objs::Vector{Float32}
    exactGrad::Bool
    li::Int
    memorized::Dict
    vStar::Float32
    ϵ::Float32
    linear_part::Float32
    quadratic_part::Float32
end


"""
    initializeBundle(bt, ϕ, t, z, lt, nn, max_bundle_size; exactGrad, instance_features) -> DeepBundle

Construct and initialize a `DeepBundle` for maximizing the concave function `ϕ`.

The bundle is initialized at the starting point `z`. The neural network `nn` is
immediately used to predict the initial step size `t`, after which the first
Dual Master Problem is created and solved, and the initial search direction is computed.

# Arguments
- `bt::tLearningBundleFactory`: Factory type used for dispatch.
- `ϕ`: The concave objective function to maximize.
- `t`: Initial regularization/step-size parameter (will be overridden by the network prediction).
- `z`: Starting point for the optimization.
- `lt::AbstractTModelFactory`: Factory used to create input features for the neural network.
- `nn`: Neural network model predicting `t` (default: `create_NN(lt)`).
- `max_bundle_size`: Maximum number of bundle components to store (default: `500`).

# Keyword Arguments
- `exactGrad::Bool`: If `true`, use the KKT-based correction in the backward pass (default: `true`).
- `instance_features::Bool`: If `true`, extract problem-specific quantile features from `ϕ.inst`
  to populate `B.ϕ0`; otherwise `ϕ0` is a zero vector of length 20 (default: `false`).

# Returns
An initialized `DeepBundle` with the DMP solved and the initial direction computed.
"""
function initializeBundle(bt::tLearningBundleFactory, ϕ, t, z, lt::AbstractTModelFactory, nn::Any = create_NN(lt), max_bundle_size = 500; exactGrad = true, instance_features::Bool = false)

    # Allocate the bundle with placeholder values; all fields will be properly set below
    B = DeepBundle(
        [;;], [;;], [;;], [], -1,
        Model(Gurobi.Optimizer), [], [], 0, [0], [Float32[]],
        BundleParameters(), nn, lt, [], [], [], [], [],
        0, 0, [], 1, Float32[], exactGrad, 1,
        Dict("times" => []), 0.0, 0.0, 0.0, 0.0
    )

    # Set the maximum bundle size (adding 1 to account for the stabilization point slot)
    B.params.max_β_size = max_bundle_size + 1

    # Reset the neural network's hidden state (relevant for recurrent architectures)
    Flux.reset!(nn)

    # Oracle call: evaluate objective and subgradient at the starting point
    obj, g = value_gradient(ϕ, z)

    # The stabilization point starts at column 1 (the initialization point)
    B.s = 1

    # Flatten the starting point and subgradient to 1D vectors for storage
    z = reshape(z, :)
    g = reshape(g, :)

    if 0 < B.params.max_β_size < Inf
        # Fixed-size storage: pre-allocate all matrices to max_β_size columns/rows
        B.α   = zeros(Float32, B.params.max_β_size)
        B.G   = zeros(Float32, (length(g), B.params.max_β_size))
        B.z   = zeros(Float32, (length(z), B.params.max_β_size))
        B.G[:, 1] = g      # Column 1: subgradient at the starting point
        B.z[:, 1] = z      # Column 1: starting point
        B.obj = zeros(Float32, B.params.max_β_size)
        B.obj[1] = obj     # Entry 1: objective value at the starting point

        # Pre-allocate the Gram matrix Q = GᵀG with the first diagonal entry
        B.Q         = zeros(Float32, (B.params.max_β_size, B.params.max_β_size))
        B.Q[1, 1]   = g' * g
    else
        # Variable-size storage: grow arrays dynamically as new components are added
        B.α   = [0]
        B.G   = reshape(g, (length(g), 1))
        B.Q   = Float32[g'g;;]   # 1×1 Gram matrix
        B.obj = [obj]
        B.z   = z
    end

    # Build the initial DMP with the provided starting t (will be replaced by network prediction)
    B.model = create_DQP(B, t)

    # Initialize the search direction to zero and θ to a single-component weight
    B.w = zeros(length(z))
    B.θ = ones(1)

    # Store the initial t parameter
    B.params.t = t

    # Build instance-level context features for the neural network:
    # If instance_features is true, extract distributional statistics (quantiles) from the problem.
    # Otherwise, use a zero vector as a neutral placeholder (20 dimensions by convention).
    B.ϕ0 = instance_features ?
        Float32.(vcat(
            quantile(ϕ.inst.r),
            quantile(ϕ.inst.f),
            quantile([ϕ.inst.K[i][3] for i in eachindex(ϕ.inst.K)]),
            quantile(zeros(size(ϕ.inst.c)))
        )) :
        zeros(Float32, 20)

    # First neural network prediction: use the initial bundle state to predict t
    f = create_features(B, B.nn)
    B.params.t = cpu(B.nn(f, B))

    # Rebuild the DMP with the network-predicted t and solve it
    B.model = create_DQP(B, B.params.t)
    solve_DQP(B)

    # Extract the initial search direction and derived quantities
    compute_direction(B)

    # Reset the search direction to the initial subgradient (overrides the DMP solution,
    # which may be degenerate at initialization with a single bundle component)
    B.w = B.G[:, 1]

    # Initialize the history buffers with the starting-point data
    B.cumulative_θ = [Float32[1.0]]   # Single-component simplex weight at initialization
    push!(B.ts,       B.params.t)     # Record the first predicted step size
    push!(B.ws,       B.w)            # Record the initial search direction
    push!(B.θ2s,      θ2(B))          # Record the KKT correction vector
    push!(B.features, ϕ)              # Record the initial feature input (the function itself)
    push!(B.all_objs, obj)            # Record the initial objective value

    return B
end


"""
    trial_point(B::DeepBundle) -> AbstractVector

Compute the next trial point for a `DeepBundle`.

Unlike the classical `DualBundle` (which uses a fixed `t`), the `DeepBundle`
calls the neural network at every trial point computation to obtain a fresh
prediction of the step size. The trial point is:
    z_new = zS(B) + nn(features(B)) * w

where `nn(features(B))` is the network-predicted step size and `w = B.w`
is the current search direction.
"""
function trial_point(B::DeepBundle)
    # Extract features from the current bundle state
    ϕ = create_features(B, B.nn)
    # Move features to the appropriate device (GPU if available)
    ϕ = device(ϕ)
    # Compute trial point: stabilization point + (predicted step size) * search direction
    return zS(B) + cpu(B.nn(ϕ, B)) .* B.w
end


# Helper: check if a value is NaN (used for numerical stability guards)
isNaN(x) = return (x === NaN)


"""
    θ2(B) -> AbstractVector

Compute the KKT-based correction vector used in the exact gradient of `ws` w.r.t. `t`.

For the active bundle components (those with `θ[i] > 1e-12`), this function
solves a reduced system derived from the KKT conditions of the DMP to compute
a second-order correction vector `θ2`. This correction improves the accuracy of
`∂w/∂t` during backpropagation when `B.exactGrad = true`.

The correction is computed as:
    θ2[base] = Q[base,base]⁻¹ * ((eᵀ Q⁻¹ α) / (eᵀ Q⁻¹ e) * e - α[base])

where `e` is the all-ones vector and `base` are the indices of active components.
Returns the correction projected back to the full subgradient space: `G * θ2`.

# Notes
- If the reduced Gram matrix `Q[base, base]` is rank-deficient, the correction
  is skipped and a zero vector is returned.
"""
function θ2(B)
    # Identify active components: those with non-negligible DMP weight
    base = [i for i in eachindex(B.θ) if B.θ[i] > 10^(-12)]
    e    = ones(length(base))
    θ2   = zeros(length(B.α))

    # Extract the reduced Gram matrix for active components
    Qb = B.Q[base, base]

    # Only compute the correction if the reduced Gram matrix is full-rank
    if rank(Qb) >= size(Qb, 1)
        Qm1        = inv(Qb)
        # KKT-derived correction: projects α onto the orthogonal complement of e in Q-metric
        θ2[base]   = Qm1 * ((e' * Qm1 * B.α[base]) / (e' * Qm1 * e) * e - B.α[base])
    end

    # Project back to the subgradient space: return G * θ2
    return B.G * θ2
end


"""
    ws(t, i, B) -> AbstractVector

Compute the trial direction displacement at iteration `i` as `B.ws[i] * t`.

This function represents the contribution of the step-size prediction `t` to
the trial point displacement at iteration `i`. It is used inside the training
loss to make the computation differentiable with respect to `t` (and hence
with respect to the neural network parameters).

# Arguments
- `t`: Step size predicted by the neural network at iteration `i`.
- `i::Int`: Iteration index used to look up the stored search direction `B.ws[i]`.
- `B::DeepBundle`: The bundle object containing the history of search directions.
"""
ws(t, i, B) = B.ws[i] .* t


"""
    ChainRulesCore.rrule(::typeof(ws), t, i, B)

Custom reverse-mode differentiation rule for `ws`.

Defines how gradients flow through the `ws` function during backpropagation.
When `B.exactGrad = true`, the gradient of `w` with respect to `t` includes
the KKT-based correction `θ2`, yielding a more accurate second-order approximation:
    ∂w/∂t ≈ w - (1/t) * θ2(B)

When `B.exactGrad = false`, only the first-order term `w` is used.

# Returns
- `value`: The forward computation `B.ws[i] .* t`.
- `loss_pullback`: A function mapping the output cotangent `dl` to input cotangents.
  Only the cotangent with respect to `t` is non-trivial; `i` and `B` receive `NoTangent()`.
"""
function ChainRulesCore.rrule(::typeof(ws), t, i, B)
    # Forward pass: compute the trial direction displacement
    value = B.ws[i] .* t

    # Gradient of w w.r.t. t: start with the search direction itself (first-order term)
    gs = B.ws[i]

    if B.exactGrad
        # Apply KKT-based second-order correction for a better approximation of ∂w/∂t
        gs .-= (1 / t) * B.θ2s[i]
    end

    # Pullback function: maps output cotangent dl to input cotangent w.r.t. t
    loss_pullback(dl) = (NoTangent(), gs' * dl, NoTangent(), NoTangent())
    return value, loss_pullback
end


"""
    Bundle_value_gradient!(B::DeepBundle, ϕ, sample, single_prediction) -> Bool

Run the bundle method forward pass and collect all information needed for the
backward pass, without computing gradients yet.

At each iteration:
1. The neural network predicts `t` from the current bundle features.
2. The DMP is updated and solved to obtain the search direction `w`.
3. A trial point is computed as `zS + t * w` and evaluated via the oracle.
4. The bundle is updated with the new (z, g, obj) triple.
5. The Serious Step / Null Step decision is made and `B.s` is updated if needed.
6. All quantities needed for the backward pass (`ws`, `θ2s`, `ts`, `features`) are stored.

# Arguments
- `B::DeepBundle`: The bundle object (will be mutated in place).
- `ϕ::AbstractConcaveFunction`: The concave function to maximize.
- `sample::Bool`: If `true`, use stochastic perturbations in the network (default from `B.nn.sample`).
- `single_prediction::Bool`: If `true`, the network predicts `t` only once at the start
  and reuses it for all iterations, rather than re-predicting at each step (default: `false`).

# Returns
`false` (always; the function is designed for its side effects on `B`).

# Side effects
- Resets `B.features`, `B.ws`, `B.ts`, `B.θ2s`, `B.back_prop_idx` at the start.
- Populates them with per-iteration data for use in `train!`.
"""
function Bundle_value_gradient!(B::DeepBundle, ϕ::AbstractConcaveFunction, sample = true, single_prediction::Bool = false)

    # Reset all history buffers before the new forward pass
    B.features     = []
    B.ws           = []
    B.ts           = []
    B.θ2s          = []
    B.back_prop_idx = []

    # Compute the initial feature vector (used if single_prediction = true)
    ϕ0 = device(create_features(B, B.nn))

    if (single_prediction)
        # Single prediction mode: predict t once and reuse it for all iterations
        B.params.t = cpu(B.nn(ϕ0, B))
    end

    for epoch in 1:B.params.maxIt
        if !(single_prediction)
            # Standard mode: re-predict t at every iteration using fresh features
            f          = device(create_features(B, B.nn))
            B.params.t = cpu(B.nn(f, B))
        else
            # Reuse the initial feature vector for all iterations
            f = ϕ0
        end

        # Update and solve the DMP with the new t, then extract the search direction
        update_DQP!(B)
        solve_DQP(B)
        compute_direction(B)

        # Compute the linear part of the DMP objective (used for bundle maintenance)
        δ = LinearAlgebra.dot(B.α[1:B.size], B.θ)

        # Remove stale bundle components if enough iterations have passed
        if epoch >= B.params.remotionStep
            remove_outdated(B)
        end

        # Compute the new trial point: move from the stabilization point along w by step t
        z = cpu(zS(B) + B.params.t * B.w)

        # Oracle call: evaluate objective and subgradient at the trial point
        obj, g = value_gradient(ϕ, z)

        # Snapshot the bundle state before the update (for SS/NS decision below)
        old_size, old_s, old_objS = B.size, B.s, B.obj[B.s]

        # Update the bundle with the new trial point information
        update_Bundle(B, z, g, obj)
        new_size, new_s = B.size, B.s

        # Guard against NaN objectives (numerical instability)
        if isnan(obj)
            println("Error the objective is NaN")
            return B
        end

        # --- Serious Step / Null Step decision ---
        changed_s = false
        if B.obj[B.li] - old_objS > B.params.m1 * B.objB
            # Serious Step: the new trial point sufficiently improves the objective
            B.CSS += 1
            B.CNS  = 0
            B.s    = B.li   # Move stabilization point to the new trial point
            update_linearization_errors(B)
            changed_s = true
        else
            # Null Step: insufficient improvement; keep the current stabilization point
            if new_s == old_s && new_size == old_size
                # Bundle was not structurally changed; the DMP update can be warm-started
                changed_s = true
            end
            B.CNS += 1
            B.CSS  = 0
        end

        # Record per-iteration data for the backward pass
        push!(B.ws,       B.w)          # Search direction at this iteration
        push!(B.θ2s,      θ2(B))        # KKT correction vector at this iteration
        push!(B.ts,       B.params.t)   # Predicted step size at this iteration
        push!(B.features, f)            # Feature vector fed to the network at this iteration
        append!(B.all_objs, B.obj[B.li]) # Objective value at the new trial point

        # Mark this iteration for backpropagation if the stabilization point changed
        if changed_s
            push!(B.back_prop_idx, epoch)
        end
    end

    return false
end


"""
    train!(B::DeepBundle, ϕ, state; kwargs...) -> Float32

Perform a full forward + backward pass and update the neural network parameters.

The forward pass is delegated to `Bundle_value_gradient!`, which runs the bundle
method and stores all intermediate quantities. The backward pass then differentiates
a scalar loss through the stored `ws` and `features` history.

Two loss formulations are supported:
- **Non-telescopic** (`telescopic=false`): The loss is the objective value at the
  final point obtained by summing all Serious Step displacements from `z0`.
- **Telescopic** (`telescopic=true`): The loss combines both Serious Step and
  Null Step contributions via `both_contributions`, with a discount factor `γ`.

# Arguments
- `B::DeepBundle`: The bundle object (mutated in place during the forward pass).
- `ϕ`: The concave objective function.
- `state`: Flux optimizer state for updating the neural network parameters.

# Keyword Arguments
- `samples::Int`: Number of gradient samples to average (default: `1`).
- `telescopic::Bool`: If `true`, use the telescopic loss including NS contributions (default: `false`).
- `γ::Float64`: Discount factor for the telescopic loss (default: `0.9`).
- `δ::Float64`: Small weight for Null Step contributions in the loss (default: `0.00001`).
- `normalization_factor`: Scalar used to normalize the loss (default: `1`).
- `oneshot::Bool`: Reserved flag, currently unused (default: `true`).
- `gold`: Reference objective value for comparison, currently unused (default: `zeros(1)`).
- `single_prediction::Bool`: If `true`, reuse one network prediction for all iterations (default: `false`).

# Returns
The negated loss value `-v` (since the method maximizes `ϕ`, the loss is negated for minimization).
"""
function train!(B::DeepBundle, ϕ, state; samples = 1, telescopic = false, γ = 0.9, δ = 0.00001, normalization_factor = 1, oneshot = true, gold = zeros(1), single_prediction::Bool = false)
    sample = B.nn.sample
    par    = Flux.trainable(B.nn)

    # Snapshot the current stabilization point before the forward pass
    z0 = zS(B)

    # Snapshot the current step size (will be restored between samples)
    t = B.params.t

    first_run = B.size <= 1 ? true : false

    # Reset the network hidden state and run the full bundle forward pass
    Flux.reset!(B.nn)
    rng = B.nn.rng
    Bundle_value_gradient!(B, ϕ, sample)

    # Snapshot the noise samples generated during the forward pass (for reproducibility)
    ϵs          = B.nn.ϵs
    B.nn.rng    = rng
    first_sample = true

    let v, vss, vns, grad
        for sample_idx in 1:samples
            # Scale factor for the second and subsequent samples (reduce contribution)
            γS = 1
            if sample_idx > 1
                γS = 0.1
            end

            # Restore t to the value before the forward pass
            B.params.t = t

            fs = 1

            fs = 2  # Reserved: may be used for future multi-step feature indexing

            # Reset the network hidden state for a clean backward pass
            Flux.reset!(B.nn)
            pred = z0  # Starting point for the loss computation

            # Prepend a noise sample entry for the initialization step
            ϵs = vcat([1], ϵs)

            if B.back_prop_idx != []
                if !telescopic
                    # Non-telescopic loss: sum only the Serious Step displacements,
                    # then evaluate ϕ at the resulting point.
                    # Only the iterations in back_prop_idx contribute to the loss.
                    v, grad = withgradient(
                        (m) -> -1 / normalization_factor *
                            ϕ(reshape(
                                z0 + sum([ws(m(B.features[i], B, sample ? ϵs[i] : 0), i, B)
                                          for i in eachindex(B.features)][B.back_prop_idx]),
                                sizeInputSpace(ϕ)
                            )),
                        B.nn
                    )
                else
                    # Telescopic loss: combine both SS and NS contributions with discount γ
                    v, grad = withgradient(
                        (m) -> -1 / normalization_factor *
                            both_contributions(
                                B, pred,
                                [m(B.features[i], B, sample ? ϵs[i] : 0) for i in eachindex(B.ws)],
                                ϕ, γ, δ
                            ),
                        B.nn
                    )
                end
            else
                # If no iterations were flagged for backprop, fall back to the telescopic loss
                v, grad = withgradient(
                    (m) -> -1 / normalization_factor *
                        both_contributions(
                            B, pred,
                            [m(B.features[i], B, sample ? ϵs[i] : 0) for i in eachindex(B.ws)],
                            ϕ, γ, δ
                        ),
                    B.nn
                )
            end

            # Apply the computed gradient to update the neural network parameters
            Flux.update!(state, B.nn, grad[1])
            first_sample = false
        end

        # Clear the noise sample buffer after the update
        B.nn.ϵs = []

        # Return the negated loss (positive value = better objective)
        return -v
    end
end


"""
    both_contributions(B, pred, ts, ϕ, γ, δ) -> Float32

Compute the combined training loss from both Serious Step (SS) and Null Step (NS) iterations.

The loss is a weighted combination of:
- **Serious Step contribution**: A discounted sum `Σ γⁱ * ϕ(z_ss_i)` where each `z_ss_i`
  is the cumulative point reached after `i` Serious Steps. Earlier SS contribute less
  (discount `γ < 1` with increasing exponent for later steps).
- **Null Step contribution**: A small regularization term `δ * Σ ϕ(z_ns_j)` where each
  `z_ns_j` is the point at a Null Step, computed by adding the NS displacement to the
  cumulative point from the most recent SS before it.

The full displacement matrix `wss` is built as:
    wss[:, 1]   = pred (starting point)
    wss[:, i+1] = ws(ts[i], i, B)  for each iteration i

# Arguments
- `B::DeepBundle`: The bundle object providing `back_prop_idx` and search direction history.
- `pred`: The starting point `z0` (displacement origin).
- `ts`: Vector of step sizes predicted by the network at each iteration.
- `ϕ`: The concave objective function to evaluate.
- `γ::Float64`: Discount factor for Serious Step contributions (default: `0.1`).
- `δ::Float64`: Weight for Null Step contributions (default: `0.00001`).
"""
function both_contributions(B, pred, ts, ϕ, γ = 0.1, δ = 0.00001)
    # Build the full displacement matrix: column 1 = pred, columns 2..end = trial displacements
    wss = hcat(pred, [ws(ts[i], i, B) for i in eachindex(B.ws[1:end])]...)

    # Identify Null Step column indices (not in B.back_prop_idx, offset by 1 for pred column)
    ns = [i for i in 2:size(wss, 2) if !(i in B.back_prop_idx .+ 1)]

    # Identify Serious Step column indices (include column 1 = starting point)
    ss = vcat(1, B.back_prop_idx .+ 1)

    # For each NS index, find the most recent SS that preceded it
    # (used to build the NS cumulative point correctly)
    ss_ns = [maximum(vcat(1, [j for j in B.back_prop_idx if j <= i])) + 1 for i in ns]

    # Discount factors for SS and NS contributions
    γs  = [γ^i for i in 0:length(ss)-1]
    γns = [γ^i for i in 0:length(ns)-1]

    # Ensure δ (NS weight) does not exceed the smallest SS discount factor
    if γs != []
        δ = min(δ, minimum(γs))
    end

    # Objective at each NS point: cumulative SS displacement up to ss_ns[j], plus the NS displacement
    loss_values_ns = [ϕ(reshape(
        sum(wss[:, i] for i in 1:ss_ns[j]) + wss[:, j],
        sizeInputSpace(ϕ)
    )) for j in eachindex(ns)]

    # Objective at each SS point: cumulative displacement through all SS columns up to j
    loss_values_ss = [ϕ(reshape(
        sum(wss[:, i] for i in ss[1:j]),
        sizeInputSpace(ϕ)
    )) for j in eachindex(ss)]

    # Total loss = δ * (sum of NS objective values) + Σ γⁱ * (SS objective values)
    return δ * sum(vcat(loss_values_ns, 0.0)) + sum(vcat(γs .* loss_values_ss, 0.0))
end


"""
    bundle_execution(B::DeepBundle, ϕ; t_strat, unstable, force_maxIt, inference) -> result

Run the `DeepBundle` optimization loop to maximize `ϕ`.

At each iteration the neural network predicts the step size `t` via `trial_point`,
the DMP is updated and re-solved, and the Serious Step / Null Step decision is made.

!!! note
    This function wraps most side-effectful operations in `ignore_derivatives()` blocks
    to allow it to be called within a differentiable context (e.g., for meta-learning).
    The actual gradient flow passes through `both_contributions` at the end.

# Arguments
- `B::DeepBundle`: The initialized bundle object.
- `ϕ::AbstractConcaveFunction`: The concave function to maximize.

# Keyword Arguments
- `t_strat::abstract_t_strategy`: t-strategy applied after the neural network prediction 
  (default: `constant_t_strategy()`).
- `unstable::Bool`: If `true`, always move the stabilization point (default: `false`).
- `force_maxIt::Bool`: If `true`, run all `maxIt` iterations regardless of stopping criterion (default: `true`).
- `inference::Bool`: If `true`, check the stopping criterion and return early if satisfied (default: `false`).

# Returns
- **Inference mode** (`inference=true`): `(converged::Bool, times::Dict)`.
- **Training mode** (`inference=false`): The scalar loss from `both_contributions`.
"""
function bundle_execution(B::DeepBundle, ϕ::AbstractConcaveFunction; t_strat::abstract_t_strategy = constant_t_strategy(), unstable::Bool = false, force_maxIt::Bool = true, inference::Bool = false)
    times = Dict(
        "times"           => [],
        "trial point"     => [],
        "ϕ"               => [],
        "update β"        => [],
        "SS/NS"           => [],
        "update DQP"      => [],
        "solve DQP"       => [],
        "remove outdated" => []
    )

    ignore_derivatives() do
        t0 = time()
        t1 = time()
    end

    for epoch in 1:B.params.maxIt
        ignore_derivatives() do
            t1 = time()
        end

        # --- Step 1: Compute the new trial point using the neural network ---
        # The network predicts t internally inside trial_point(B)
        z = trial_point(B)

        ignore_derivatives() do
            append!(times["trial point"], (time() - t1))
            t0 = time()
        end

        # --- Step 2: Oracle call — evaluate objective and subgradient at z ---
        obj, g = value_gradient(ϕ, z)

        ignore_derivatives() do
            append!(times["ϕ"], (time() - t0))

            # --- Step 3: Update the bundle with the new (z, g, obj) data ---
            update_Bundle(B, z, g, obj)
        end

        # Snapshot the current t and stabilization point index before the t-strategy
        t = B.params.t
        s = B.s

        # --- Step 4: Apply the t-strategy (and possibly move the stabilization point) ---
        t_strategy(B, B.li, t_strat, unstable)

        ignore_derivatives() do
            append!(times["update β"], (time() - t0))
            t0 = time()

            # --- Step 5: Incrementally update the DMP ---
            # Pass flags to enable selective (cheaper) updates
            update_DQP!(B, t == B.params.t, s == B.s)
            append!(times["update DQP"], (time() - t0))

            t0 = time()
            # --- Step 6: Re-solve the DMP ---
            solve_DQP(B)
            append!(times["solve DQP"], (time() - t0))

            # --- Step 7: Extract the new search direction and derived quantities ---
            compute_direction(B)
        end

        # --- Step 8: Update the search direction via the ws function ---
        # NOTE: ws() is called here without arguments — this appears to be a placeholder
        # that should pass the current t and iteration index (e.g., ws(B.params.t, epoch, B))
        B.w = ws()

        ignore_derivatives() do
            t0 = time()
            # --- Step 9 (disabled): Remove outdated bundle components ---
            if epoch >= B.params.remotionStep
                # remove_outdated(B)  # Disabled; uncomment to enable bundle compression
            end
            append!(times["remove outdated"], (time() - t0))

            # Record the current step size in history
            append!(B.ts, B.params.t)
        end

        # --- Step 10: Check stopping criterion (inference mode only) ---
        ignore_derivatives() do
            if inference && !(force_maxIt) && stopping_criteria(B)
                println("Satisfied stopping criteria")
                return true, times
            end
        end

        ignore_derivatives() do
            push!(B.memorized["times"], time() - t1)
        end
    end

    ignore_derivatives() do
        times["times"] = B.memorized["times"]
    end

    if inference
        ignore_derivatives() do
            return false, times
        end
    else
        # Training mode: return the differentiable loss for backpropagation
        # NOTE: both_contributions() is called here without arguments — this appears
        # to be incomplete and should pass (B, pred, ts, ϕ, γ, δ) explicitly
        return both_contributions()
    end
end