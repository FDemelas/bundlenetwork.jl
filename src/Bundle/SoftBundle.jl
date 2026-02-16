"""
SoftBundle structure.

A single-function variant of the soft bundle method. Unlike the classical
`DualBundle`, it does not solve a Dual Master Problem to find the search
direction; instead, it delegates both the step-size selection and the
direction computation to an external model (typically a neural network).

Compared to `BatchedSoftBundle`, this structure handles a single objective
function rather than a batch, making it suitable for single-instance
optimization or sequential training episodes.

# Fields
- `G`: Matrix of subgradients, one column per visited point (size: input_dim × (maxIt+1)).
- `z`: Matrix of visited trial points, one column per iteration (size: input_dim × (maxIt+1)).
- `α`: Matrix of linearization errors (size: 1 × (maxIt+1)).
- `s::Int64`: Index of the current stabilization point column in `G`, `z`, and `obj`.
- `w`: Current search direction (convex combination of stored subgradients).
- `θ`: Current DMP solution (simplex weights over bundle components).
- `γ`: Discount factor used in the telescopic training loss.
- `objB::Float32`: Value of the Dual Master Problem objective (proxy, not solved exactly).
- `obj`: Matrix of objective function values at visited points (size: 1 × (maxIt+1)).
- `cumulative_θ`: History of simplex weight vectors over iterations (used for feature extraction).
- `lt::AbstractModelFactory`: Factory of the model predicting the search direction and step size.
- `back_prop_idx`: Indices controlling which iterations participate in backpropagation.
- `CSS::Int64`: Counter for consecutive Serious Steps.
- `CNS::Int64`: Counter for consecutive Null Steps.
- `ϕ0`: Initial objective value(s) used as a reference baseline for features.
- `li::Int64`: Index of the last inserted bundle component (current bundle pointer).
- `info::Dict`: Dictionary for storing diagnostic and logging information.
- `maxIt::Int`: Maximum number of bundle iterations.
- `t::Vector{Float32}`: Step size vector (scalar wrapped in a vector for AD compatibility).
- `size::Int64`: Current number of bundle components stored.
- `lis::Vector{Int64}`: History of `li` values across iterations.
- `reduced_components::Bool`: If `true`, duplicate subgradients are removed from the active component set.
"""
mutable struct SoftBundle <: AbstractSoftBundle
    G::Any
    z::Any
    α::Any
    s::Int64
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
    size::Int64
    lis::Vector{Int64}
    reduced_components::Bool
end


"""
    initializeBundle(bt::SoftBundleFactory, ϕ, z, lt, maxIt, reduced_components) -> SoftBundle

Construct and initialize a `SoftBundle` for optimizing a single concave function.

The bundle is initialized at the starting point `z`: the oracle is called once
to compute the initial objective value and subgradient, which populate column 1
of the bundle matrices. All subsequent columns are pre-allocated to zero.

# Arguments
- `bt::SoftBundleFactory`: Factory type used for dispatch.
- `ϕ::AbstractConcaveFunction`: The concave objective function to maximize.
- `z::AbstractArray`: Starting point for the optimization.
- `lt`: Factory of the neural network model used in place of the Dual Master Problem.
- `maxIt::Int`: Maximum number of bundle iterations (default: `10`).
- `reduced_components::Bool`: If `true`, duplicate subgradients are filtered from
  the active component set during the execution (default: `false`).

# Returns
An initialized `SoftBundle` with all matrices pre-allocated and populated at `z`.
"""
function initializeBundle(bt::SoftBundleFactory, ϕ::AbstractConcaveFunction, z::AbstractArray, lt, maxIt::Int = 10, reduced_components::Bool = false)
    # Allocate the bundle with placeholder values; all fields will be properly set below
    B = SoftBundle([], [], [], -1, [], [], [], Inf, [Inf], [Float32[]], lt, [], 0, 0, [], 1, Dict(), maxIt, [1.0], 1, [1], reduced_components)

    # Oracle call at the starting point: compute objective value and subgradient
    obj, g = value_gradient(ϕ, reshape(z, sizeInputSpace(ϕ)))

    # Flatten the subgradient to a 1D vector and move to CPU for storage
    g = reshape(cpu(g), :)

    # The stabilization point starts at column 1 (the initialization point)
    B.s = 1

    # Allocate the visited-points matrix (rows = input_dim, cols = maxIt + 1)
    B.z = zeros(Float32, length(z), B.maxIt + 1)
    # Column 1 is the flattened starting point
    B.z[:, 1] = reshape(z, :)

    # Allocate the linearization error matrix (1 row since single function, cols = maxIt + 1)
    # All zeros initially: the stabilization point coincides with the starting point
    B.α = zeros(Float32, 1, B.maxIt + 1)

    # Allocate the subgradient matrix (rows = input_dim, cols = maxIt + 1)
    B.G = zeros(Float32, length(g), B.maxIt + 1)
    # Column 1 holds the subgradient at the starting point
    B.G[:, 1] = g

    # Allocate the objective value matrix (1 row, cols = maxIt + 1)
    B.obj = zeros(Float32, 1, B.maxIt + 1)
    # Column 1 holds the objective value at the starting point
    B.obj[:, 1] = [obj]

    # Set the initial DMP objective value proxy to the squared norm of the subgradient
    B.objB = g' * g

    # The initial search direction is the subgradient at the starting point
    B.w = g

    # Initialize θ: a single bundle component, so all weight on column 1
    B.θ = ones(1, 1)

    # The last inserted component index starts at 1 (the initialization point)
    B.li = 1

    return B
end


"""
    reinitialize_Bundle!(B::SoftBundle)

Reset a `SoftBundle` to its initial state, preserving only the initialization point.

Allows the same bundle object to be reused across multiple optimization runs
(e.g., training episodes) without re-allocating memory. Should be called before
each bundle execution, whether or not a backward pass will follow.

# Side effects
- Resets `li`, `s`, `size`, and `lis` to their initial values.
- Reinitializes all bundle matrices (`G`, `z`, `α`, `obj`) via `Zygote.bufferfrom`:
  column 1 (initialization data) is preserved; all other columns are zeroed.
  `G` and `z` are moved to device (GPU if available); `α` and `obj` stay on CPU.
- Resets `θ` to a uniform single-component weight and `w` to the initial subgradient.
"""
function reinitialize_Bundle!(B::SoftBundle)
    # Reset size counters: only the initialization point remains active
    B.li   = 1
    B.s    = 1
    B.size = 1
    B.lis  = [1]

    # Reinitialize bundle matrices, preserving column 1 (initialization point)
    # and zeroing all remaining columns. Zygote.bufferfrom keeps the arrays
    # as mutable AD-compatible buffers for use inside the differentiable execution loop.
    B.G   = Zygote.bufferfrom(device(hcat([B.G[:, 1], zeros(size(B.G, 1), B.maxIt)]...)))
    B.z   = Zygote.bufferfrom(device(hcat([B.z[:, 1], zeros(size(B.z, 1), B.maxIt)]...)))
    B.α   = Zygote.bufferfrom(cpu(hcat(B.α[:, 1], zeros(size(B.α, 1), B.maxIt))))
    B.obj = Zygote.bufferfrom(cpu(hcat(B.obj[:, 1], zeros(size(B.obj, 1), B.maxIt))))

    # Reset θ to a single-component weight vector (all mass on the initialization point)
    B.θ = ones(1, 1)

    # Reset the search direction to the initial subgradient (first column of G)
    B.w = device(B.G[:, 1])
end


"""
    bundle_execution(B::SoftBundle, ϕ, m; kwargs...) -> loss or (obj, times)

Run the `SoftBundle` optimization loop to maximize the concave function `ϕ`,
using model `m` in place of the classical Dual Master Problem and t-strategy.

The function is fully differentiable via Zygote's automatic differentiation,
making it suitable as the inner loop of a learning-to-optimize training pipeline.

# Arguments
- `B::SoftBundle`: The bundle object (must be reinitialized before calling).
- `ϕ::AbstractConcaveFunction`: The concave objective function to maximize.
- `m::AbstractModel`: Neural network model providing the step size `t` and
  the raw DMP weights `γ` at each iteration.

# Keyword Arguments
- `soft_updates::Bool`: If `true` (default), the stabilization point is updated
  using a softmax blending of the new trial point and current stabilization point.
  If `false`, a hard improvement check is used (update only if `obj_new > obj_bar`).
- `λ::Float64`: Weight of the final trial point in the training loss;
  `(1-λ)` weights the final stabilization point (default: `0.0`).
- `γ::Float64`: Discount factor for the telescopic sum term of the training loss.
  Set to `0.0` to disable (default: `0.0`).
- `δ::Float64`: Reserved parameter, currently unused (default: `0.0`).
- `distribution_function`: Function mapping raw model outputs to simplex weights
  (default: `softmax`).
- `verbose::Int`: Verbosity level (default: `0`, silent).
- `max_inst::Real`: Maximum number of past bundle components used as model input.
  Use `Inf` to use all available components (default: `Inf`).
- `metalearning::Bool`: Flag for meta-learning mode, currently unused (default: `false`).
- `unstable::Bool`: If `true`, always move the stabilization point to the latest
  trial point, ignoring the improvement condition (default: `false`).
- `inference::Bool`: If `true`, run in inference mode: returns the objective value
  at the final stabilization point and a timing dictionary, instead of the training
  loss (default: `false`).
- `z_bar`: Differentiable buffer for the current stabilization point (initialized
  to column `B.s` of `B.z`).
- `z_new`: Differentiable buffer for the new trial point (initialized to column
  `B.li` of `B.z`).
- `act`: Activation function applied to the new trial point after the gradient step
  (default: `identity`).

# Returns
- **Inference mode** (`inference=true`): `(obj, times)` where `obj` is the objective
  value at the final stabilization point and `times` is a per-phase timing dictionary.
- **Training mode** (`inference=false`): A scalar loss:
  - `vλ`: Convex combination of final trial point and stabilization point objectives.
  - `vγ`: Discounted telescopic sum over all visited points (zero if `γ = 0`).
  - Total: `vγ + vλ`.
"""
function bundle_execution(
    B::SoftBundle,
    ϕ::AbstractConcaveFunction,
    m::AbstractModel;
    soft_updates = true,
    λ = 0.0,
    γ = 0.0,
    δ = 0.0,
    distribution_function = softmax,
    verbose::Int = 0,
    max_inst = Inf,
    metalearning = false,
    unstable = false,
    inference = false,
    z_bar = Zygote.bufferfrom(Float32.(vcat([B.z[:, B.s]]...))),
    z_new = Zygote.bufferfrom(B.z[:, B.li]),
    act = identity
)
    # Variables used across `ignore_derivatives()` blocks must be declared in the
    # enclosing `let` scope to remain visible to both AD and non-AD code paths
    let xt, xγ, z_copy, LR_vec, Baseline, obj_new, obj_bar, g, t0, t1, times, maxIt, t, γs, θ, w, comps

        # --- Initialization (outside the AD graph) ---
        ignore_derivatives() do
            # Initialize the per-phase timing dictionary for profiling
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
            t0    = time()
        end

        # Global feature placeholder (unused for standard AttentionModel; kept for API compatibility)
        featG = 0.0  # function_features(B, B.lt)

        # --- Initialize working buffers ---
        ignore_derivatives() do
            # Buffer for the subgradient at the current trial point
            g = Zygote.bufferfrom(device(zeros(size(B.w))))

            # Buffer for the objective value at the new trial point
            obj_new = Zygote.bufferfrom(cpu(B.obj[B.li .* ones(Int64, 1)]))

            # Buffer for the objective value at the stabilization point
            obj_bar = Zygote.bufferfrom(cpu(B.obj[B.s .* ones(Int64, 1)]))

            # Pre-allocate a buffer of search direction vectors, one per iteration
            w = Zygote.bufferfrom([device(zeros(length(B.w))) for it in 1:maxIt])

            # Pre-allocate raw model output buffers (one per iteration, growing size)
            # γs[it] has shape (1 × it) — one column per active bundle component
            γs = Zygote.bufferfrom([device(zeros(1, it)) for it in 1:maxIt])

            # Pre-allocate simplex weight buffers (same shape as γs, values in [0,1])
            θ = Zygote.bufferfrom([device(zeros(1, it)) for it in 1:maxIt])

            # Reset the DMP objective proxy
            B.objB = 0

            # Record initialization time
            times["init"] = time() - t0

            # Active component index sets: comps[it] holds column indices of bundle
            # components used at iteration `it`; starts with just [1] (initialization point)
            comps = Zygote.bufferfrom([ones(Int64, it) for it in 1:maxIt+1])
        end

        # --- Main bundle iteration loop ---
        for it in 1:maxIt
            ignore_derivatives() do
                t0 = time()
            end

            # --- Feature extraction (outside AD graph) ---
            ignore_derivatives() do
                # Build input features from the current bundle state for the model
                xt, xγ = device(create_features(B.lt, B; auxiliary = featG))

                # Ensure features are 2D column vectors for matrix operations
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
            # The model outputs a scalar step size `t` and raw logits `γs[it]` over
            # the `it` active bundle components
            t, γs[it] = m(xt, xγ, B.li, comps[it])

            # Store B.t outside the AD graph (features extraction does not need gradients through t)
            ignore_derivatives() do
                B.t = reshape(t, :)
            end

            # Limit the number of past components visible to the model (for efficiency)
            min_idx = Int64(max(1, length(comps[it]) - max_inst))

            ignore_derivatives() do
                append!(times["model"], time() - t1)  # Log model forward pass time
            end

            ignore_derivatives() do
                t1 = time()
            end

            # --- Compute simplex weights from raw model output ---
            # Apply distribution_function (e.g. softmax) along the component dimension
            θ[it] = distribution_function(γs[it][:, :]; dims = 2)

            ignore_derivatives() do
                B.θ = θ[it]  # Store for use in feature extraction at the next iteration
                append!(times["distribution"], time() - t1)
            end

            ignore_derivatives() do
                t1 = time()
            end

            # --- Compute new search direction ---
            # w[it] = G[:, comps[it]] * θ[it][1, :] — convex combination of stored subgradients
            w[it] = B.G[:, comps[it]] * θ[it][1, :]

            ignore_derivatives() do
                B.w = w[it]  # Store for feature extraction at the next iteration
                append!(times["update_direction"], time() - t1)
            end

            ignore_derivatives() do
                t1 = time()
            end

            # --- Compute new trial point ---
            # z_new = act(z_bar + sum(t) * w[it])
            # sum(t) collapses the step-size vector to a scalar (single-function case)
            z_new[:] = act(z_bar[:] .+ sum(t) .* w[it])

            ignore_derivatives() do
                append!(times["update_point"], time() - t1)  # Log trial point update time
            end

            ignore_derivatives() do
                t1 = time()
            end

            # --- Oracle call: evaluate objective and subgradient at z_new ---
            # In Lagrangian relaxation settings, this is the Lagrangian subproblem solve
            v, g_tmp   = value_gradient(ϕ, z_new[:])
            g[:]       = device(g_tmp)  # Move subgradient to device
            obj_new[1] = v              # Store scalar objective value

            ignore_derivatives() do
                append!(times["lsp"], time() - t1)  # Log oracle call time
            end

            # Advance the bundle size pointer to the newly added component
            B.size += 1
            B.li    = B.size

            # --- Update the active component set for the next iteration ---
            if B.reduced_components
                # Check for duplicate subgradients: if a stored subgradient is
                # approximately equal to g (within 1e-6), mark it for removal
                already_in = false
                j = []
                for i in comps[it]
                    if sum(B.G[:, i] - g[:]) < 1.0e-6
                        already_in = true
                        ignore_derivatives() do
                            push!(j, i)  # Flag duplicate for removal
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
                # Standard behavior: always append the new component
                comps[it+1] = vcat(comps[it], B.size)
            end

            ignore_derivatives() do
                # Record the current bundle pointer in history (for features/diagnostics)
                append!(B.lis, B.li)
            end

            ignore_derivatives() do
                t1 = time()
            end

            # --- Update stabilization point ---
            if unstable
                # Unstable mode: always move the stabilization point to the latest trial point
                z_bar   = z_new
                obj_bar = obj_new
                B.s     = B.li .* ones(Int64, length(B.s))
            else
                if !soft_updates
                    # Hard update: move the stabilization point only if the new trial point
                    # strictly improves the objective (classical Serious Step condition)
                    z_bar[:]   = obj_new[:] > obj_bar[:] ? (z_new[:]) : (z_bar[:])
                    obj_bar[:] = (obj_new[:] > obj_bar[:] ? obj_new[:] : obj_bar[:])
                else
                    # Soft update: blend the new and current stabilization points using
                    # softmax weights derived from their objective values.
                    # Points with higher objective get more weight in the blend.
                    sm         = softmax([obj_new[1], obj_bar[1]])
                    obj_bar[1] = sm' * [obj_new[1], obj_bar[1]]
                    z_bar[:]   = device(sm' * cpu.([z_new[:], z_bar[:]]))
                end
                # Update the stabilization point index used for feature extraction
                B.s = sum(obj_new[:] > obj_bar[:] ? B.li : B.s)
            end

            ignore_derivatives() do
                append!(times["stab_point"], time() - t1)  # Log stabilization update time
            end

            ignore_derivatives() do
                t1 = time()
            end

            # --- Update bundle matrices with the new trial point data ---
            B.G[:, B.li]   = device(g[:])   # Store new subgradient
            B.z[:, B.li]   = z_new[:]       # Store new trial point
            B.obj[:, B.li] = obj_new[:]     # Store new objective value

            # Update linearization errors: α[1, k] = (f(z_k) - f(z_bar)) - g_k^T (z_k - z_bar)
            # Computed for all columns simultaneously using broadcasting
            B.α[1, :] = (B.obj[1, :] .- obj_bar[1, :]) .-
                         cpu(sum(B.G[:, :] .* (B.z[:, :] .- z_bar[:]); dims = 1))'

            ignore_derivatives() do
                append!(times["update_bundle"], time() - t1)  # Log bundle matrix update time
                append!(times["iters"],         time() - t0)  # Log total iteration time
            end
        end

        # --- Return value depending on mode ---
        if inference
            # Inference mode: return the objective at the final stabilization point
            return ϕ(reshape(z_bar, sizeInputSpace(ϕ))), times
        else
            # Training mode: return a scalar loss to backpropagate through

            # vγ: discounted telescopic sum over all visited points.
            # Encourages the model to improve the objective early in the episode.
            # Disabled when γ = 0.
            vγ = (γ > 0
                ? γ * mean(
                    [γ^(maxIt + 1 - i) for i in (maxIt+1):-1:1] .*
                    [ϕ(reshape(z, sizeInputSpace(ϕ))) for z in eachcol(B.z[:, 1:(maxIt+1)])]
                  )
                : 0)

            # vλ: convex combination of the final trial point and stabilization point objectives.
            # λ = 0 → only the stabilization point (conservative).
            # λ = 1 → only the final trial point (aggressive).
            vλ = (1 - λ) * ϕ(reshape(z_bar[:], sizeInputSpace(ϕ))) +
                      λ  * ϕ(reshape(z_new[:], sizeInputSpace(ϕ)))

            # Total training loss = telescopic term + terminal term
            return vγ + vλ
        end
    end
end