"""
BatchedSoftBundle structure.

A batched variant of the SoftBundle that supports parallel execution over multiple
bundle instances, enabling mini-batch training of the neural network policy.
Like SoftBundle, it does not solve a classical Dual Master Problem to find the 
search direction; instead, it delegates this to an external model (typically a 
neural network).

# Fields
- `G`: Matrix of subgradients, one column per visited point (size: total_input_dim × (maxIt+1)).
- `z`: Matrix of visited trial points, one column per iteration (size: total_input_dim × (maxIt+1)).
- `α`: Matrix of linearization errors, one row per objective function (size: batch_size × (maxIt+1)).
- `s`: Vector of stabilization point indices, one per objective in the batch.
- `w`: Current search direction (convex combination of stored subgradients).
- `θ`: Current DMP solution (simplex weights over bundle components).
- `γ`: Discount factor used in the telescopic training loss.
- `objB`: Scalar value of the Dual Master Problem objective.
- `obj`: Matrix of objective function values at visited points (size: batch_size × (maxIt+1)).
- `cumulative_θ`: Accumulated θ values over iterations (used for feature extraction).
- `lt`: Factory of the model used for predictions in place of the Dual Master Problem.
- `back_prop_idx`: Indices used to control which iterations participate in backpropagation.
- `CSS`: Counter for serious steps (iterations where the stabilization point is updated).
- `CNS`: Counter for null steps (iterations where the stabilization point is not updated).
- `ϕ0`: Initial objective values (used as a reference for features).
- `li`: Index of the last inserted bundle component (current bundle size pointer).
- `info`: Dictionary for storing diagnostic/logging information.
- `maxIt`: Maximum number of bundle iterations.
- `t`: Vector of step sizes, one per objective in the batch.
- `idxComp`: Vector of (start, end) index pairs for each objective's slice in the concatenated vectors.
- `size`: Current number of bundle components stored.
- `lis`: History of `li` values over iterations.
- `reduced_components`: If `true`, duplicate subgradients are removed from the active component set.
"""
mutable struct BatchedSoftBundle <: AbstractSoftBundle
    G::Any
    z::Any
    α::Any
    s::Vector{Int64}
    w::Any
    θ::Any
    γ::Any
    objB::Float32
    obj::Any
    cumulative_θ::Any
    lt::AbstractModelFactory
    back_prop_idx::Any
    CSS::Int64
    CNS::Int64
    ϕ0::Any
    li::Int64
    info::Dict
    maxIt::Int
    t::Vector{Float32}
    idxComp::AbstractArray
    size::Int64
    lis::Vector{Int64}
    reduced_components::Bool
end

"""
    initializeBundle(bt::BatchedSoftBundleFactory, ϕs, z, lt, maxIt, reduced_components) -> BatchedSoftBundle

Construct and initialize a `BatchedSoftBundle` from a batch of concave objective functions.

# Arguments
- `bt::BatchedSoftBundleFactory`: The factory type dispatching this constructor.
- `ϕs::Vector{<:AbstractConcaveFunction}`: Vector of concave objective functions to optimize (one per batch element).
- `z::Vector{<:AbstractArray}`: Vector of starting points, one per objective. Must have the same length as `ϕs`.
- `lt`: Factory of the model used for predictions in place of the Dual Master Problem.
- `maxIt::Int`: Maximum number of bundle iterations (default: 10).
- `reduced_components::Bool`: If `true`, duplicate subgradients are filtered from the active component set (default: `false`).

# Returns
An initialized `BatchedSoftBundle` with all matrices pre-allocated and populated 
at the starting points.

# Notes
- The bundle matrices (`G`, `z`, `α`, `obj`) are allocated with `maxIt + 1` columns,
  where column 1 holds the initialization values.
- `idxComp` stores index ranges to slice the concatenated gradient/point vectors 
  back into per-objective components.
"""
function initializeBundle(bt::BatchedSoftBundleFactory, ϕs::Vector{<:AbstractConcaveFunction}, z::Vector{<:AbstractArray}, lt, maxIt::Int = 10, reduced_components::Bool = false)
    # Allocate the bundle with placeholder values; all fields will be properly set below
    B = BatchedSoftBundle([], [], [], [-1], [], [], [], Inf, [Inf], [Float32[]], lt, [], 0, 0, [], 1, Dict(), maxIt, zeros(length(ϕs)), [], 1, [1], reduced_components)

    # The batch size equals the number of input functions
    batch_size = length(ϕs)

    # Accumulators for the concatenated subgradients, objective values, and index ranges
    sLM = []       # List of input space dimensions for each objective
    gs = []        # Concatenated subgradients at the starting points
    objs = []      # Objective values at the starting points
    idxComp = []   # (start, end) row index pairs for each objective's slice
    tmp = 0        # Running row offset for building index ranges

    # Evaluate the objective and subgradient at each starting point
    # and record the index range for each objective's slice in the global vectors
    for (idx, ϕ) in enumerate(ϕs)
        lLM = prod(sizeInputSpace(ϕ))           # Total dimension of the input space for ϕ
        push!(idxComp, (tmp + 1, tmp + lLM))    # Store (start, end) row indices for this objective
        tmp += lLM                              # Advance the row offset
        append!(sLM, prod(lLM))                 # Record the input dimension

        # Evaluate objective value and subgradient at the starting point z[idx]
        obj, g = value_gradient(ϕ, reshape(z[idx], sizeInputSpace(ϕ)))
        g = reshape(g, :)   # Flatten the subgradient to a 1D vector
        append!(objs, obj)
        append!(gs, g)
    end

    # Allocate the visited-points matrix (rows = total input dim, cols = maxIt + 1)
    B.z = zeros(Float32, sum(sLM), B.maxIt + 1)
    # Column 1 is the concatenation of all starting points
    B.z[:, 1] = vcat([zi for zi in z]...)

    # Allocate the linearization error matrix (rows = batch_size, cols = maxIt + 1)
    # All zeros initially because the stabilization point coincides with the starting point
    B.α = zeros(Float32, batch_size, B.maxIt + 1)

    # Allocate the subgradient matrix (rows = total input dim, cols = maxIt + 1)
    B.G = zeros(Float32, sum(sLM), B.maxIt + 1)
    # Column 1 holds the subgradients at the starting points
    B.G[:, 1] = gs

    # Allocate the objective value matrix (rows = batch_size, cols = maxIt + 1)
    B.obj = zeros(Float32, batch_size, B.maxIt + 1)
    # Column 1 holds the objective values at the starting points
    B.obj[:, 1] = objs

    # Set the initial DMP objective value (norm squared of the initial subgradient)
    B.objB = gs'gs

    # The initial search direction is simply the subgradient at the starting point
    B.w = gs

    # Initialize θ as a uniform weight vector (one component per objective)
    # At initialization the bundle has a single component, so all weight goes to it
    B.θ = ones(batch_size, 1)

    # Store the index ranges for slicing the concatenated vectors per objective
    B.idxComp = idxComp

    # The last inserted component index starts at 1 (the initialization point)
    B.li = 1

    # Each objective's stabilization point starts at column 1 of the bundle matrices
    B.s = ones(length(idxComp))

    return B
end

"""
    reinitialize_Bundle!(B::BatchedSoftBundle)

Reset a `BatchedSoftBundle` to its initial state, preserving only the initialization point.

This allows the same bundle object to be reused across multiple optimization runs
(e.g., different training episodes) without re-allocating memory.
It should be called before each bundle execution, whether or not a backward pass 
will be performed.

# Side effects
- Resets all bundle size counters (`li`, `size`, `lis`).
- Resets stabilization point indices `s` to point to the first column.
- Reinitializes all bundle matrices (`G`, `z`, `α`, `obj`) using `Zygote.bufferfrom`
  to keep them compatible with automatic differentiation: the first column 
  (initialization point) is preserved, and all subsequent columns are zeroed.
- Resets `θ` to a uniform weight vector and `w` to the initial subgradient.
"""
function reinitialize_Bundle!(B::BatchedSoftBundle)
    # Reset the bundle size: only the initialization point remains
    B.li = 1
    B.size = 1
    B.lis = [1]

    # Reset all stabilization point indices to the first column (initialization point)
    B.s = ones(length(B.idxComp))

    # Reinitialize bundle matrices, keeping column 1 and zeroing the rest.
    # Zygote.bufferfrom is used to keep the result differentiable (mutable buffer for AD).
    # G and z are moved to device (GPU if available); α and obj stay on CPU.
    B.G   = Zygote.bufferfrom(device(hcat([B.G[:, 1], zeros(size(B.G, 1), B.maxIt)]...)))
    B.z   = Zygote.bufferfrom(device(hcat([B.z[:, 1], zeros(size(B.z, 1), B.maxIt)]...)))
    B.α   = Zygote.bufferfrom(cpu(hcat(B.α[:, 1], zeros(size(B.α, 1), B.maxIt))))
    B.obj = Zygote.bufferfrom(cpu(hcat(B.obj[:, 1], zeros(size(B.obj, 1), B.maxIt))))

    # Reset θ to uniform weights (one component = initialization point)
    B.θ = ones(length(B.idxComp), 1)

    # Reset the search direction to the initial subgradient (first column of G)
    # (B.w could remain all-zero before the first prediction, but this is more consistent)
    B.w = device(B.G[:, 1])
end

"""
    bundle_execution(B, ϕ, m; kwargs...) -> loss or (obj, times)

Run the `BatchedSoftBundle` optimization loop to maximize the functions in `ϕ`,
using the model `m` in place of the classical Dual Master Problem and t-strategy.

The function is designed to be fully differentiable via Zygote's automatic 
differentiation, making it suitable as the inner loop of a learning-to-optimize 
training pipeline.

# Arguments
- `B::BatchedSoftBundle`: The bundle object (must be reinitialized before calling).
- `ϕ::Vector{<:AbstractConcaveFunction}`: Batch of concave objective functions to maximize.
- `m::AbstractModel`: Neural network model providing the step size `t` and 
  the raw DMP weights `γ` at each iteration.

# Keyword Arguments
- `soft_updates::Bool`: If `true`, update the stabilization point using a softmax 
  blending of the new trial point and the current stabilization point, rather than 
  a hard improvement check (default: `false`).
- `λ::Float64`: Weight of the final trial point in the training loss. 
  `(1-λ)` weights the final stabilization point (default: `0.0`).
- `γ::Float64`: Discount factor for the telescopic sum term of the training loss. 
  Set to `0.0` to disable (default: `0.0`).
- `δ::Float64`: Reserved parameter (currently unused, default: `0.0`).
- `distribution_function`: Function used to map raw model outputs `γs[it]` to 
  simplex weights `θ[it]` (default: `softmax`).
- `verbose::Int`: Verbosity level (default: `0`, silent).
- `max_inst::Real`: Maximum number of past bundle components used as model input. 
  Use `Inf` to use all components (default: `Inf`).
- `metalearning::Bool`: Flag for meta-learning mode (currently unused, default: `false`).
- `unstable::Bool`: If `true`, always update the stabilization point to the latest 
  trial point (i.e., no serious/null step distinction) (default: `false`).
- `inference::Bool`: If `true`, run in inference mode: returns the mean final 
  objective value and a timing dictionary instead of the training loss (default: `false`).
- `z_bar`: Differentiable buffer holding the current stabilization point (initialized 
  to the per-objective stabilization columns of `B.z`).
- `z_new`: Differentiable buffer for the new trial point (initialized to the last 
  visited point column of `B.z`).
- `act`: Activation function applied to the new trial point after the gradient step 
  (default: `identity`).

# Returns
- **Inference mode** (`inference=true`): `(mean_obj, times)` where `mean_obj` is the 
  mean objective value at the final stabilization points across the batch, and `times` 
  is a dictionary of per-phase timing measurements.
- **Training mode** (`inference=false`): A scalar loss value combining:
  - `vλ`: A convex combination of the final trial point and stabilization point objectives.
  - `vγ`: A discounted telescopic sum over all visited points (zero if `γ = 0`).
"""
function bundle_execution(
    B::BatchedSoftBundle,
    ϕ::Vector{<:AbstractConcaveFunction},
    m::AbstractModel;
    soft_updates = false,
    λ = 0.0,
    γ = 0.0,
    δ = 0.0,
    distribution_function = softmax,
    verbose::Int = 0,
    max_inst = Inf,
    metalearning = false,
    unstable = false,
    inference = false,
    z_bar = Zygote.bufferfrom(Float32.(vcat([B.z[s:e, B.s[i]] for (i, (s, e)) in enumerate(B.idxComp)]...))),
    z_new = Zygote.bufferfrom(B.z[:, B.li]),
    act = identity
)
    # Variables used across `ignore_derivatives()` blocks must be declared in the
    # enclosing `let` scope to be visible to both AD and non-AD code paths
    let xt, xγ, z_copy, LR_vec, Baseline, obj_new, obj_bar, g, t0, t1, times, maxIt, t, γs, θ, comps

        # --- Initialization (outside AD graph) ---
        ignore_derivatives() do
            # Initialize the timing dictionary for profiling each phase of the loop
            times = Dict(
                "init"             => 0.0,
                "iters"            => [],
                "model"            => [],
                "distribution"     => [],
                "features"         => [],
                "stab_point"       => [],
                "update_bundle"    => [],
                "update_direction" => [],
                "update_point"     => [],
                "lsp"              => []
            )
            maxIt = B.maxIt
            t0 = time()
        end

        # Global features placeholder (unused for standard AttentionModel; kept for API compatibility)
        featG = 0.0  # function_features(B, B.lt)

        # --- Initialize buffers used inside the loop ---
        ignore_derivatives() do
            # Buffer for the subgradient at the current trial point
            g = Zygote.bufferfrom(device(zeros(size(B.w))))

            # Buffer for objective values at the new trial point (initialized to the current li column)
            obj_new = Zygote.bufferfrom(cpu(B.obj[B.li .* ones(Int64, length(B.s))]))

            # Objective value at the stabilization point (same as obj_new at initialization)
            obj_bar = obj_new

            # Pre-allocate raw model output buffers (one per iteration, growing size)
            # γs[it] has shape (batch_size × it) — one column per active bundle component
            γs = Zygote.bufferfrom([device(zeros(length(B.idxComp), it)) for it in 1:maxIt])

            # Pre-allocate simplex weight buffers (same shape as γs, values in [0,1])
            θ  = Zygote.bufferfrom([device(zeros(length(B.idxComp), it)) for it in 1:maxIt])

            # Reset the DMP objective value
            B.objB = 0

            # Record initialization time
            times["init"] = time() - t0

            # Active component index sets: comps[it] holds the column indices of the
            # bundle components used at iteration `it`
            comps = Zygote.bufferfrom([ones(Int64, it) for it in 1:maxIt+1])
        end

        # --- Main bundle iteration loop ---
        for it in 1:maxIt
            ignore_derivatives() do
                t0 = time()
            end

            # --- Feature extraction (outside AD graph) ---
            ignore_derivatives() do
                # Create input features for the model from the current bundle state
                xt, xγ = device(create_features(B.lt, B; auxiliary = featG))

                # Ensure features are 2D (column vector) for matrix operations
                if size(xt)[1] == length(xt)
                    xt = reshape(xt, (length(xt), 1))
                end
                if size(xγ)[1] == length(xγ)
                    xγ = reshape(xγ, (length(xγ), 1))
                end
            end

            ignore_derivatives() do
                append!(times["features"], time() - t0)  # Log feature extraction time
            end

            ignore_derivatives() do
                t1 = time()
            end

            # --- Model forward pass (inside AD graph) ---
            # The model outputs: a step size vector `t` and raw logits `γs[it]` over bundle components
            t, γs[it] = m(xt, xγ, B.li, comps[it])
            B.t = device(reshape(t, :))  # Flatten t to a 1D vector and move to device

            # Limit the number of past components seen by the model (for memory/speed control)
            min_idx = Int64(max(1, length(comps[it]) - max_inst))

            ignore_derivatives() do
                append!(times["model"], time() - t1)  # Log model forward pass time
            end

            ignore_derivatives() do
                t1 = time()
            end

            # --- Compute simplex weights from raw model output ---
            # Apply the distribution function (e.g. softmax) along the component dimension
            θ[it] = distribution_function(γs[it][:, :]; dims = 2)

            ignore_derivatives() do
                B.θ = θ[it]  # Store for use in feature extraction at the next iteration
                append!(times["distribution"], time() - t1)
            end

            ignore_derivatives() do
                t1 = time()
            end

            # --- Compute new search direction ---
            # w = convex combination of stored subgradients weighted by θ[it],
            # computed independently for each objective's slice of the global gradient vector
            B.w = vcat([B.G[s:e, comps[it]] * θ[it][i, :] for (i, (s, e)) in enumerate(B.idxComp)]...)

            ignore_derivatives() do
                append!(times["update_direction"], time() - t1)
            end

            ignore_derivatives() do
                t1 = time()
            end

            # --- Compute new trial point ---
            # z_new = act(z_bar + t * w), applied per-objective using the respective step size
            z_new[:] = act(z_bar[:] + vcat([B.t[i] * B.w[s:e] for (i, (s, e)) in enumerate(B.idxComp)]...))

            ignore_derivatives() do
                append!(times["update_point"], time() - t1)
            end

            ignore_derivatives() do
                t1 = time()
            end

            # --- Oracle call: evaluate objectives and subgradients at z_new ---
            # In Lagrangian relaxation settings, this corresponds to solving the Lagrangian subproblem
            for (i, (s, e)) in enumerate(B.idxComp)
                v, g_tmp = value_gradient(ϕ[i], z_new[s:e])
                g[s:e]   = g_tmp   # Store subgradient slice for objective i
                obj_new[i] = v     # Store objective value for objective i
            end

            # Advance the bundle size pointer to the newly added component
            B.size += 1
            B.li = B.size

            # --- Update the active component set for the next iteration ---
            if B.reduced_components
                # If reduced_components is enabled, check for duplicate subgradients
                # (i.e., components already in the bundle with the same gradient as g)
                # and remove them to avoid redundancy
                already_in = false
                j = []
                for i in comps[it]
                    if sum(B.G[:, i] - g[:]) < 1.0e-6  # Approximate equality check
                        already_in = true
                        ignore_derivatives() do
                            push!(j, i)  # Mark this component for removal
                        end
                    end
                end

                if already_in
                    # Remove duplicate components and add the new one
                    comps[it+1] = vcat([k for k in comps[it] if !(k in j)], B.size)
                else
                    # No duplicates: simply append the new component
                    comps[it+1] = vcat(comps[it], B.size)
                end
            else
                # Standard behavior: always append the new component without filtering
                comps[it+1] = vcat(comps[it], B.size)
            end

            ignore_derivatives() do
                append!(times["lsp"], time() - t1)  # Log oracle call + bundle update time
            end

            ignore_derivatives() do
                t1 = time()
            end

            # --- Update stabilization point ---
            if unstable
                # Unstable mode: always move the stabilization point to the latest trial point
                z_bar   = z_new
                obj_bar = obj_new
                B.s = B.li .* ones(Int64, length(B.s))
            else
                if !soft_updates
                    # Hard update: move stabilization point only if the new trial point
                    # strictly improves the objective value (classical serious step condition)
                    for (i, (s, e)) in enumerate(B.idxComp)
                        z_bar[s:e]  = (obj_new[i] > obj_bar[i] ? z_new[s:e]  : z_bar[s:e])
                        obj_bar[i]  = (obj_new[i] > obj_bar[i] ? obj_new[i]  : obj_bar[i])
                    end
                else
                    # Soft update: blend the new and current stabilization point
                    # using softmax weights derived from their objective values
                    for (i, (s, e)) in enumerate(B.idxComp)
                        sm       = softmax([obj_new[i], obj_bar[i]])
                        obj_bar[i] = sm' * [obj_new[i], obj_bar[i]]
                        z_bar[s:e] = device(sm' * cpu.([z_new[s:e], z_bar[s:e]]))
                    end
                end

                # Update the stabilization point column index (used for feature extraction)
                B.s = vcat([obj_new[i] > obj_bar[i] ? B.li : B.s[i] for i in 1:length(B.s)]...)
            end

            ignore_derivatives() do
                append!(times["stab_point"], time() - t1)  # Log stabilization point update time
            end

            ignore_derivatives() do
                t1 = time()
            end

            # --- Update bundle matrices with the new trial point data ---
            B.G[:, B.li]   = device(g[:])      # Store new subgradient
            B.z[:, B.li]   = z_new[:]          # Store new trial point
            B.obj[:, B.li] = obj_new[:]        # Store new objective value

            # Update linearization errors: α[i,k] = (f(z_k) - f(z_bar_i)) - g_k^T (z_k - z_bar_i)
            # This measures how well the cutting plane at z_k approximates f around the stabilization point
            for (i, (s, e)) in enumerate(B.idxComp)
                B.α[i, :] = (B.obj[i, :] .- obj_bar[i, :]) .-
                             cpu(sum(B.G[s:e, :] .* (B.z[s:e, :] .- z_bar[s:e]); dims = 1))'
            end

            ignore_derivatives() do
                append!(times["update_bundle"], time() - t1)  # Log bundle matrix update time
                append!(times["iters"],         time() - t0)  # Log total iteration time
            end
        end

        # --- Return value depending on mode ---
        if inference
            # Inference mode: return mean objective at the final stabilization point across the batch
            return mean(
                ϕ[iϕ](reshape(z_bar[s:e], sizeInputSpace(ϕ[iϕ])))
                for (iϕ, (s, e)) in enumerate(B.idxComp)
            ), times
        else
            # Training mode: return a scalar loss to backpropagate through

            # vγ: discounted telescopic sum of objective values over all visited points
            # Encourages the model to improve the objective as early as possible
            # (zero if γ = 0, i.e., discount is disabled)
            vγ = (γ > 0
                ? mean(
                    γ * mean(
                        [γ^(maxIt+1 - i) for i in (maxIt+1):-1:1] .*
                        [ϕ[iϕ](reshape(z[s:e], sizeInputSpace(ϕ[iϕ]))) for z in eachcol(B.z[:, 1:(maxIt+1)])]
                    )
                    for (iϕ, (s, e)) in enumerate(B.idxComp)
                  )
                : 0)

            # vλ: convex combination of the final trial point and stabilization point objectives
            # λ = 0 → use only the stabilization point (conservative)
            # λ = 1 → use only the final trial point (aggressive)
            vλ = mean(
                (1 - λ) * ϕ[iϕ](reshape(z_bar[s:e], sizeInputSpace(ϕ[iϕ]))) +
                λ       * ϕ[iϕ](reshape(z_new[s:e], sizeInputSpace(ϕ[iϕ])))
                for (iϕ, (s, e)) in enumerate(B.idxComp)
            )

            # Total training loss = telescopic term + terminal term
            return vγ + vλ
        end
    end
end