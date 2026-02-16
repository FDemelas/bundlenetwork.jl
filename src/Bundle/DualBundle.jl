"""
    create_DQP(B::DualBundle, t::Float64) -> Model

Build the Dual Quadratic Master Problem (DQP) for the bundle `B` using `t` as 
the regularization parameter.

The DQP has the form:
    min  (1/2) * ‖G θ‖² + (1/t) * αᵀθ
    s.t. 1ᵀθ = 1,   0 ≤ θ ≤ 1

where `G` is the subgradient matrix, `α` is the linearization error vector,
and `θ` is the simplex variable. The linear part is scaled by `1/t` (instead
of using `t` in the quadratic part) to enable faster re-optimization when
only `t` changes.

If `B.sign` is `true`, the problem includes non-negativity constraints on the
multipliers (useful for sign-constrained dual problems, e.g., Lagrangian relaxation
with non-negative multipliers).

!!! note
    It is recommended to call this function only once at initialization and then
    use `update_DQP!` for subsequent iterations, to exploit warm-starting.

# Arguments
- `B::DualBundle`: The bundle object containing subgradients, linearization errors, and parameters.
- `t::Float64`: Regularization parameter (controls the trade-off between the quadratic 
  proximal term and the linear cutting-plane model).

# Returns
A JuMP `Model` representing the DQP, with Gurobi as the solver.
"""
function create_DQP(B::DualBundle, t::Float64)
    # Create an empty JuMP model with Gurobi as the optimizer
    model = Model(Gurobi.Optimizer)

    # Allow non-convex quadratic objectives (required by Gurobi for QPs with
    # non-PSD quadratic matrices that may arise from the subgradient outer product)
    set_attribute(model, "NonConvex", 2)

    # Set a time limit of 1 second per solve to avoid hanging in degenerate cases
    set_time_limit_sec(model, 1.0)

    # Force Gurobi to use the Primal Simplex method (Method=0).
    # This is preferred over dual simplex or barrier because it allows
    # efficient warm-starting when the model is updated incrementally.
    set_attribute(model, "Method", 0)

    # Extract the subgradient matrix: use only the first `B.size` columns if
    # a maximum bundle size is enforced, otherwise use the full matrix
    g = 0 < B.params.max_β_size < Inf ? B.G[:, 1:B.size] : B.G

    # Extract the linearization error vector with the same size logic
    α = 0 < B.params.max_β_size < Inf ? B.α[1:B.size] : B.α

    # Define one simplex variable θ_i per bundle component, bounded in [0, 1]
    @variable(model, 1 >= θ[1:size_Bundle(B)[1]] >= 0)

    # Convex combination constraint: θ must lie in the probability simplex
    @constraint(model, conv_comb, ones(size_Bundle(B)[1])' * θ == 1)

    # Build the objective function.
    # When B.sign is true (sign-constrained dual, e.g., Lagrangian relaxation):
    #   - introduce auxiliary non-negative slack variables λ ≥ 0
    #   - add non-negativity constraints: t*(G θ + λ) ≥ -zS (component-wise)
    #   - this effectively enforces the dual variable to remain non-negative
    if B.sign
        @variable(model, λ[1:size(B.z[:, 1], 1)] >= 0)
        zs = zS(B)
        @constraint(model, non_negativity[i = 1:size(B.G, 1)], t * (g[i] * θ[1] + λ[i]) >= -zs[i])
    end

    # Quadratic part of the objective: ‖G θ‖² (or ‖G θ + λ‖² if sign-constrained)
    quadratic_part = B.sign ?
        @expression(model, LinearAlgebra.dot(g * θ + λ, g * θ + λ)) :
        @expression(model, LinearAlgebra.dot(g * θ, g * θ))

    # Linear part of the objective: αᵀθ (plus λᵀzS if sign-constrained)
    linear_part = B.sign ?
        @expression(model, LinearAlgebra.dot(α, θ) + LinearAlgebra.dot(λ, zS(B))) :
        @expression(model, LinearAlgebra.dot(α, θ))

    # Full objective: (1/2) * quadratic + (1/t) * linear
    # The 1/t scaling of the linear part (rather than t in the quadratic) allows
    # faster re-optimization: only linear coefficients need updating when t changes
    @objective(model, Min, (1 / 2) * quadratic_part .+ 1 / t * linear_part)

    # Store variable references in obj_dict for fast access during updates
    model.obj_dict = B.sign ? Dict(
        :θ            => θ,                # Simplex decision variables
        :λ            => λ,                # Non-negativity slack variables
        :non_negativity => non_negativity, # Sign constraints
        :conv_comb    => conv_comb         # Simplex equality constraint
    ) : Dict(
        :θ         => θ,                   # Simplex decision variables
        :conv_comb => conv_comb            # Simplex equality constraint
    )

    # Suppress solver output to keep the console clean
    set_silent(model)
    return model
end


"""
    solve_DQP(B::DualBundle)

Solve the Dual Quadratic Master Problem stored in `B.model` and update `B.objB`
with the rescaled optimal objective value.

The DQP is formulated with the linear part scaled by `1/t` for re-optimization
efficiency. After solving, the objective is rescaled back by multiplying by `t`
to recover the true DMP objective value used in the step acceptance criteria
and t-strategies.
"""
function solve_DQP(B::DualBundle)
    # Solve the Dual Quadratic Master Problem using Gurobi
    optimize!(B.model)

    # Rescale the objective value back: the DQP uses (1/t)*linear internally,
    # so multiply by t to recover the true dual objective value
    B.objB = B.params.t * JuMP.objective_value(B.model)
end


"""
    update_DQP!(B::DualBundle, t_change=true, s_change=true)

Incrementally update the Dual Quadratic Master Problem formulation after a bundle
iteration, exploiting warm-starting to reduce re-optimization cost.

Three independent update cases are handled:

1. **New bundle component** (when `length(θ) < B.size`): A new JuMP variable is
   added to the model, inserted into the simplex constraint, and its quadratic
   and linear contributions are added to the objective.

2. **Stabilization point change** (`s_change=true`): The linear part of the
   objective is updated to reflect the new linearization errors `B.α` (which
   depend on the stabilization point).

3. **Regularization parameter change** (`t_change=true`): All linear coefficients
   (scaled by `1/t`) and the non-negativity constraint coefficients (scaled by `t`)
   are updated to reflect the new value of `B.params.t`.

# Arguments
- `B::DualBundle`: The bundle containing the JuMP model and current parameters.
- `t_change::Bool`: Whether `t` has changed since the last update (default: `true`).
- `s_change::Bool`: Whether the stabilization point has changed (default: `true`).

# Notes
- If a bundle component with a duplicate subgradient was found, `B.size` is not
  incremented, so the variable count check `length(θ) == B.size` correctly 
  detects whether a new variable must be added.
"""
function update_DQP!(B::DualBundle, t_change = true, s_change = true)

    # --- Case 1: New bundle component was added ---
    # Check if the number of JuMP variables matches the current bundle size
    if !(length(B.model.obj_dict[:θ]) == B.size)

        # Add a new JuMP variable θ_new ∈ [0, 1] for the new bundle component
        θ_tmp = @variable(B.model, upper_bound = 1, lower_bound = 0)

        # Insert θ_tmp into the simplex constraint (coefficient = 1)
        set_normalized_coefficient(B.model.obj_dict[:conv_comb], θ_tmp, 1)

        # Update the quadratic part of the objective with the new cross-terms.
        # Each off-diagonal entry Q[i, new] appears twice (symmetric), so the
        # net contribution is 2 * (1/2) * Q[i,new] * θ_i * θ_new = Q[i,new] * θ_i * θ_new
        for (idx, θ) in enumerate(B.model.obj_dict[:θ])
            set_objective_coefficient(B.model, θ, θ_tmp, B.Q[idx, B.li])
        end

        # Add the diagonal term (1/2) * Q[new, new] * θ_new²
        # The 1/2 factor avoids double-counting the self-interaction
        set_objective_coefficient(B.model, θ_tmp, θ_tmp, 1 / 2 * B.Q[B.li, B.li])

        # Add the linear term (1/t) * α[new] * θ_new to the objective
        set_objective_coefficient(B.model, θ_tmp, 1 / (B.params.t) * B.α[B.li])

        if B.sign
            # For sign-constrained problems, also update the non-negativity constraints
            # by adding the contribution of the new gradient column G[:, new]
            for i in 1:size(B.G, 1)
                set_normalized_coefficient(
                    B.model.obj_dict[:non_negativity][i],
                    θ_tmp,
                    (B.params.t) * B.G[i, B.li],
                )
            end

            # Update the cross-terms between the new θ_new and existing λ variables
            # in the objective function
            for (idx, λ) in enumerate(B.model.obj_dict[:λ])
                set_objective_coefficient(B.model, λ, θ_tmp, B.G[idx, B.li])
            end
        end

        # Register the new variable in the model's variable dictionary
        push!(B.model.obj_dict[:θ], θ_tmp)
    end

    # --- Case 2: Stabilization point changed ---
    # Update all linear objective coefficients since α depends on the stabilization point
    if s_change
        set_objective_coefficient(B.model, B.model.obj_dict[:θ], 1 / B.params.t * B.α)

        if B.sign
            # Update the right-hand sides of the non-negativity constraints
            # and the linear objective coefficients for λ to reflect the new zS
            zs = Float64.(zS(B))
            for i in 1:size(B.G, 1)
                set_normalized_rhs(B.model.obj_dict[:non_negativity][i], -zs[i])
                set_objective_coefficient(B.model, B.model.obj_dict[:λ][i], 1 / B.params.t * zs[i])
            end
        end
    end

    # --- Case 3: Regularization parameter t changed ---
    # Re-scale all (1/t)-dependent linear coefficients and t-dependent constraint coefficients
    if t_change
        # Update the (1/t) * α linear coefficients for all θ variables
        set_objective_coefficient(B.model, B.model.obj_dict[:θ], 1 / B.params.t * B.α)

        if B.sign
            m  = size(B.G, 1)
            zs = Float64.(zS(B))
            for i in 1:m
                # Update the (1/t) * zS[i] coefficient for λ[i] in the objective
                set_objective_coefficient(B.model, B.model.obj_dict[:λ][i], 1 / B.params.t * zs[i])

                # Update the t * λ[i] coefficient in the i-th non-negativity constraint
                set_normalized_coefficient(
                    B.model.obj_dict[:non_negativity][i],
                    B.model.obj_dict[:λ][i],
                    (B.params.t),
                )

                # Update the t * G[i,j] coefficients for all θ[j] in the i-th constraint
                for j in eachindex(B.model.obj_dict[:θ])
                    set_normalized_coefficient(
                        B.model.obj_dict[:non_negativity][i],
                        B.model.obj_dict[:θ][j],
                        (B.params.t) * B.G[i, j],
                    )
                end
            end
        end
    end
end


"""
    αS(B::DualBundle) -> Real

Return the linearization error α associated with the current stabilization point.
The linearization error at the stabilization point is always zero by definition,
but this accessor is provided for API consistency.
"""
function αS(B::DualBundle)
    return B.α[B.s]
end

"""
    gS(B::DualBundle) -> AbstractVector

Return the subgradient stored in the bundle at the current stabilization point index `B.s`.
"""
function gS(B::DualBundle)
    return B.G[:, B.s]
end

"""
    zS(B::DualBundle) -> AbstractVector

Return the current stabilization point, i.e., the iterate around which the
proximal term is centered and linearization errors are computed.
"""
function zS(B::DualBundle)
    return B.z[:, B.s]
end

"""
    objS(B::DualBundle) -> Real

Return the objective function value at the current stabilization point.
Used in the Serious Step / Null Step acceptance test and linearization error updates.
"""
function objS(B::DualBundle)
    return B.obj[B.s]
end

"""
    size_Bundle(B::DualBundle) -> Int

Return the current number of active components (cutting planes) stored in the bundle.
"""
function size_Bundle(B::DualBundle)
    return B.size
end


"""
    linearization_error(B::DualBundle, i::Int) -> Real

Return the linearization error of the i-th bundle component with respect to
the current stabilization point.

The linearization error measures the gap between the cutting-plane approximation
at z_i and the true objective at the stabilization point:
    α_i = gᵢᵀ(zS - zᵢ) - (f(zS) - f(zᵢ))

!!! note
    If `i == B.s` (querying the stabilization point itself), returns 0 immediately
    since the linearization error at the stabilization point is always zero by definition.

# Arguments
- `B::DualBundle`: The bundle object.
- `i::Int`: Index of the bundle component to query.
"""
function linearization_error(B::DualBundle, i::Int)
    if i == B.s
        # Short-circuit: the linearization error at the stabilization point is identically 0
        return 0
    end
    return linearization_error(B.G[:, i], zS(B), B.z[:, i], objS(B), B.obj[i])
end

"""
    linearization_error(g, zS, z, objS, obj) -> Real

Compute the linearization error of a cutting plane at `z` with gradient `g` and
objective value `obj`, evaluated with respect to a reference point `zS` with
objective value `objS`.

The linearization error is defined as:
    α = gᵀ(zS - z) - (f(zS) - f(z))

For a concave function, α ≥ 0 always holds (the cutting plane overestimates the function).
A small α indicates that the cut at `z` is nearly tight at `zS`.

# Arguments
- `g::AbstractVector`: Subgradient at point `z`.
- `zS::AbstractVector`: Stabilization (reference) point.
- `z::AbstractVector`: Point at which the cutting plane was computed.
- `objS::Real`: Objective value at `zS`.
- `obj::Real`: Objective value at `z`.
"""
function linearization_error(g::AbstractVector, zS::AbstractVector, z::AbstractVector, objS::Real, obj::Real)
    return g' * (zS - z) - (objS - obj)
end

"""
    update_linearization_errors(B::DualBundle)

Recompute and overwrite all linearization errors in `B.α` with respect to the
current stabilization point.

Should be called whenever the stabilization point changes (i.e., after a Serious Step),
since all linearization errors depend on the reference point `zS(B)`.
"""
function update_linearization_errors(B::DualBundle)
    for i in 1:size_Bundle(B)
        B.α[i] = linearization_error(B, i)
    end
end


"""
    remove_outdated(B::DualBundle, ϵ=1e-6)

Remove bundle components that have not contributed meaningfully to the DMP solution
for several consecutive iterations, to keep the bundle compact.

A component `i` is considered "active" if its cumulative weight `cumulative_θ[j][i]`
has exceeded `ϵ` in at least one of the last `remotionStep` iterations. Components
that have never been meaningfully active are removed.

If `B.params.max_β_size` is finite, an additional cap is enforced by iteratively
removing the component with the smallest current weight `B.θ[i]` until the bundle
size is within the limit.

After identifying which components to remove, the function:
- Compacts the bundle matrices `G`, `Q`, `α`, `z`, `obj` and the weight vector `θ`.
- Updates the indices `B.li` and `B.s` to account for removed components.
- Deletes the corresponding JuMP variables from the DMP model.
- Trims the `cumulative_θ` history to the last `remotionStep` entries.

# Arguments
- `B::DualBundle`: The bundle to prune.
- `ϵ::Float64`: Minimum weight threshold below which a component is considered inactive (default: `1e-6`).

# Notes
- The stabilization point component (index `B.s`) is always retained.
- A warning is printed if the stabilization point is inadvertently flagged for removal.
"""
function remove_outdated(B::DualBundle, ϵ = 1e-6)
    sB = size_Bundle(B)
    remove_idx = []  # Indices of components to be removed
    keep_idx   = []  # Indices of components to be kept

    for i in 1:sB
        if i == B.s
            # Always keep the stabilization point component
            append!(keep_idx, i)
        else
            # Check whether component i has been active in recent iterations
            keep = 0               # Number of iterations where this component had weight > ϵ
            how_much_iter_in = 0   # Number of iterations where this component existed

            for j in eachindex(B.cumulative_θ)
                if i <= length(B.cumulative_θ[j])
                    keep += (B.cumulative_θ[j][i] > ϵ)   # Count iterations with meaningful weight
                    how_much_iter_in += 1
                end
            end

            # Keep the component if it was active at any point, or if it has not yet
            # had enough iterations to evaluate it fairly (younger than remotionStep)
            if keep > 0 || (how_much_iter_in <= B.params.remotionStep)
                append!(keep_idx, i)
            else
                append!(remove_idx, i)
                B.size -= 1
            end
        end
    end

    # --- Enforce maximum bundle size cap (if finite) ---
    if 0 < B.params.max_β_size < Inf
        B.size = length(keep_idx)
        while (B.size > B.params.max_β_size)
            # Iteratively remove the component with the smallest current DMP weight
            θ = B.θ[keep_idx]
            i = argmin(θ)
            append!(remove_idx, keep_idx[i])
            popat!(keep_idx, keep_idx[i])
            B.size -= 1
        end
    end

    # --- Compact bundle matrices to contiguous storage ---
    first_idxs = collect(1:length(keep_idx))
    if 0 < B.params.max_β_size < Inf
        # Fixed-size storage: overwrite in-place using compact indices
        first_idxs         = collect(1:length(keep_idx))
        B.G[:, first_idxs] = B.G[:, keep_idx]
        B.Q[first_idxs, first_idxs] = B.Q[keep_idx, keep_idx]
        B.α[first_idxs]    = B.α[keep_idx]
        B.z[:, first_idxs] = B.z[:, keep_idx]
        B.obj[first_idxs]  = B.obj[keep_idx]
        B.θ                = B.θ[keep_idx]
        B.size             = length(keep_idx)
    else
        # Variable-size storage: slice and reassign (reallocates memory)
        B.G   = B.G[:, keep_idx]
        B.Q   = B.Q[keep_idx, keep_idx]
        B.α   = B.α[keep_idx]
        B.z   = B.z[:, keep_idx]
        B.obj = B.obj[keep_idx]
        B.size = length(keep_idx)
        B.θ   = B.θ[keep_idx]
    end

    # --- Remove components from the JuMP model and update dependent indices ---
    sort!(remove_idx, rev = true)   # Process in descending order to avoid index shifting issues
    for h in remove_idx
        # Adjust B.li (last inserted index) if a component before it was removed
        if h < B.li
            B.li -= 1
        end

        # Adjust B.s (stabilization point index) if a component before it was removed
        if h < B.s
            B.s -= 1
        elseif h == B.s
            println("Trying to remove stabilization point")  # Safeguard: should never happen
        end

        # Remove the JuMP variable θ[h] from the DMP model
        delete(B.model, B.model.obj_dict[:θ][h])
        deleteat!(B.model.obj_dict[:θ], h)

        # Trim the cumulative weight history to remove the deleted component's entry
        for i in 1:size(B.cumulative_θ, 1)
            if h < size(B.cumulative_θ[i], 1)
                deleteat!(B.cumulative_θ[i], h)
            end
        end
    end

    # Trim cumulative_θ history to only the last `remotionStep` entries
    B.cumulative_θ = B.cumulative_θ[max(1, end - B.params.remotionStep):end]
end


"""
    compute_direction(B::DualBundle)

Extract the DMP solution `B.θ`, compute the new search direction `B.w`, and
update all auxiliary quantities used in the Serious Step / Null Step decision
and the t-strategies.

Specifically, this function:
1. Reads `θ` from the solved JuMP model and stores it in `B.θ`.
2. Appends `θ` to `B.cumulative_θ` (used by `remove_outdated`).
3. Computes the search direction `w = G θ` (convex combination of bundle subgradients).
4. Computes the linear part: `linear_part = αᵀθ`.
5. Computes the quadratic part: `quadratic_part = ‖w‖²`.
6. Computes `vStar = t * ‖w‖² + αᵀθ` — the true DMP objective (used in SS/NS test).
7. Computes `ϵ = αᵀθ + (t_star/2) * ‖w‖²` — the DMP objective evaluated at a
   fixed reference step size `t_star`, used in the stopping criterion.
"""
function compute_direction(B::DualBundle)
    # Read the optimal simplex weights from the solved JuMP model
    B.θ = value.(B.model.obj_dict[:θ])

    # Append to history for use in the component removal heuristic
    push!(B.cumulative_θ, copy(B.θ))

    # Compute the new search direction as a convex combination of bundle subgradients
    # (restrict to the active columns if a max bundle size is enforced)
    B.w = (0 < B.params.max_β_size < Inf) ?
        (B.G[:, 1:B.size] * B.θ) :
        (B.G * B.θ)

    # Linear part of the DMP objective: αᵀθ (weighted sum of linearization errors)
    B.linear_part = B.α[1:B.size]' * B.θ

    # Quadratic part of the DMP objective: ‖w‖² = ‖G θ‖²
    B.quadratic_part = B.w' * B.w

    # True DMP objective value: vStar = t * ‖w‖² + αᵀθ
    # Used in the Serious Step acceptance condition and t-strategies
    B.vStar = (B.params.t * B.quadratic_part + B.linear_part)

    # Alternative DMP objective at the fixed reference step size t_star (not the current t).
    # Used in the stopping criterion: ϵ = αᵀθ + (t_star/2) * ‖w‖²
    B.ϵ = B.linear_part + B.params.t_star * B.quadratic_part / 2
end


"""
    update_Bundle(B::DualBundle, z, g, obj)

Insert the information from a new trial point into the bundle.

If the subgradient `g` at `z` is already present in the bundle (up to a tolerance
of `1e-3` in L1 norm), the existing component is updated in-place: its linearization
error, objective value, and point are refreshed. No new variable is added to the DMP.

If `g` is genuinely new, a new row is appended (variable-size storage) or written
to the next slot (fixed-size storage), and the Gram matrix `Q = GᵀG` is updated
by appending the new column/row of inner products.

# Arguments
- `B::DualBundle`: The bundle to update.
- `z`: New trial point (will be reshaped to a vector internally).
- `g`: Subgradient at `z` (will be cast to `Float32` and flattened).
- `obj`: Objective function value at `z`.

# Notes
- Setting `B.li` to the index of the new (or updated) component allows
  `update_DQP!` to correctly identify which column of `G` to add to the model.
- The Gram matrix update uses `g'g + 1e-5` on the diagonal for numerical stability.
"""
function update_Bundle(B::DualBundle, z, g, obj)
    # Flatten the trial point and subgradient to 1D vectors
    z = reshape(z, :)
    g = Float32.(reshape(g, :))

    already_exists = false
    for j in 1:B.size
        # Check if the new subgradient is (approximately) identical to one already in the bundle
        if (sum(abs.(B.G[:, j] .- g)) < 1.0e-3)
            already_exists = true

            # Update the existing component in-place: refresh α, objective, and point
            B.α[j]    = linearization_error(g, zS(B), z, objS(B), obj)
            B.obj[j]  = obj
            B.z[:, j] = z

            # Point B.li to this component (even though it is not a new column,
            # update_DQP! uses B.li to determine which terms to refresh)
            B.li = j

            # Record this objective value in the full history
            push!(B.all_objs, obj)

            # No new variable needs to be added to the DMP; exit early
            return
        end
    end

    # --- The subgradient is genuinely new: add a new bundle component ---
    if !(already_exists)
        if 0 < B.params.max_β_size < Inf
            # Fixed-size storage: write to the next pre-allocated slot
            i = B.size + 1
            B.obj[i]    = obj
            B.z[:, i]   = z
            B.G[:, i]   = g

            # Compute the linearization error for the new component w.r.t. current zS
            B.α[i] = linearization_error(B, i)

            # Update the Gram matrix Q = GᵀG with the new column of inner products
            q = B.G[:, 1:B.size]' * g
            B.Q[1:i-1, i] = q
            B.Q[i, 1:i-1] = q
            B.Q[i, i]     = g' * g   # Diagonal entry: squared norm of new subgradient
        else
            # Variable-size storage: grow all arrays dynamically

            # Extend the visited-points matrix with the new trial point
            B.z = hcat(B.z, z)

            # Compute the new column of the Gram matrix before appending g
            q = B.G' * g

            # Append the new subgradient as a new column of G
            B.G = hcat(B.G, g)

            # Extend Q by adding the new row and column; add 1e-5 to the diagonal
            # for numerical stability (avoids a singular Gram matrix)
            new_diag = dot(g, g) + 1.0e-5
            B.Q = [B.Q      q;
                   q'  new_diag]

            # Append the new objective value
            push!(B.obj, obj)

            # Compute and append the linearization error for the new component
            α = linearization_error(B, size_Bundle(B) + 1)
            push!(B.α, α)
        end

        # Update B.li to point to the newly added component
        B.li = B.size + 1

        # Record the objective value in the full iteration history
        push!(B.all_objs, obj)

        # Increment the bundle size counter
        B.size += 1
    end
end


"""
    stopping_criteria(B::DualBundle) -> Bool

Return `true` if the bundle method's stopping criterion is satisfied, `false` otherwise.

The criterion checks whether the DMP objective value (evaluated at the fixed reference
step size `t_star`) is small relative to the magnitude of the current stabilization point:

    t_star * ‖w‖² + αᵀθ ≤ ϵ * (max(0, f(zS)) + 1)

This condition implies that no cutting plane can certify a significant improvement
over the current stabilization point, making it an approximate optimality certificate.

# Arguments
- `B::DualBundle`: The bundle object (must have `compute_direction` called beforehand).
"""
function stopping_criteria(B::DualBundle)
    return B.params.t_star * B.quadratic_part + B.linear_part <= B.params.ϵ * (max(0, objS(B)) + 1)
end


"""
    trial_point(B::DualBundle) -> AbstractVector

Compute the next trial point by taking a proximal gradient step from the
current stabilization point along the search direction `B.w`.

The update rule is:
    z_new = zS(B) + t * w

where `t = B.params.t` is the current regularization/step-size parameter.

If `B.sign` is `true` (sign-constrained dual), the DMP also optimizes auxiliary
slack variables `λ`, which are added to the direction. The resulting point is
then projected onto the non-negative orthant via `relu` to enforce non-negativity:
    z_new = relu(zS(B) + t * (w + λ))
"""
function trial_point(B::DualBundle)
    tp = zS(B) + B.params.t * (B.w .+ (B.sign ? reshape(value.(B.model.obj_dict[:λ]), :) : 0.0))
    # Apply relu projection if non-negativity is required (sign-constrained case)
    return B.sign ? relu(tp) : tp
end


"""
    solve!(B::DualBundle, ϕ; t_strat, unstable, force_maxIt) -> (Bool, Dict)

Run the bundle method to maximize the concave function `ϕ` starting from
the current state of bundle `B`.

At each iteration the algorithm:
1. Computes a new trial point via `trial_point`.
2. Evaluates the oracle `ϕ` at the trial point (objective + subgradient).
3. Updates the bundle with the new information.
4. Applies the t-strategy (and possibly moves the stabilization point).
5. Updates and re-solves the Dual Quadratic Master Problem.
6. Extracts the new search direction and checks the stopping criterion.

# Arguments
- `B::DualBundle`: The initialized bundle object.
- `ϕ::AbstractConcaveFunction`: The concave function to maximize.

# Keyword Arguments
- `t_strat::abstract_t_strategy`: The t-strategy controlling how `t` is updated.
  Defaults to `constant_t_strategy()` (fixed `t`).
- `unstable::Bool`: If `true`, always move the stabilization point to the latest 
  trial point regardless of objective improvement (default: `false`).
- `force_maxIt::Bool`: If `true`, ignore the stopping criterion and always run 
  all `maxIt` iterations (default: `true`).

# Returns
A tuple `(converged::Bool, times::Dict)`:
- `converged`: `true` if the stopping criterion was satisfied early, `false` otherwise.
- `times`: Dictionary with per-phase timing vectors for profiling.
"""
function solve!(B::DualBundle, ϕ::AbstractConcaveFunction; t_strat::abstract_t_strategy = constant_t_strategy(), unstable::Bool = false, force_maxIt::Bool = true)
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
    t0 = time()

    for epoch in 1:B.params.maxIt
        t1 = time()

        # --- Step 1: Compute the new trial point ---
        z = trial_point(B)
        append!(times["trial point"], (time() - t1))

        t0 = time()
        # --- Step 2: Oracle call — evaluate objective and subgradient at z ---
        obj, g = value_gradient(ϕ, z)
        g = g
        append!(times["ϕ"], (time() - t0))

        t0 = time()
        # --- Step 3: Update the bundle with the new (z, g, obj) information ---
        update_Bundle(B, z, g, obj)

        # Snapshot the current t and stabilization point index before the t-strategy
        # (used to determine which parts of the DMP need updating)
        t = B.params.t
        s = B.s

        # --- Step 4: Apply the t-strategy (and possibly move the stabilization point) ---
        t_strategy(B, B.li, t_strat, unstable)
        append!(times["update β"], (time() - t0))

        t0 = time()
        # --- Step 5: Update the DMP model ---
        # Pass flags indicating whether t or s changed to enable selective updates
        update_DQP!(B, t == B.params.t, s == B.s)
        append!(times["update DQP"], (time() - t0))

        t0 = time()
        # --- Step 6: Re-solve the DMP ---
        solve_DQP(B)
        append!(times["solve DQP"], (time() - t0))

        # --- Step 7: Extract the new search direction and compute derived quantities ---
        compute_direction(B)

        # --- Step 8: (Optional) Remove outdated bundle components ---
        t0 = time()
        if epoch >= B.params.remotionStep
            # remove_outdated(B)  # Disabled; uncomment to enable bundle compression
        end
        append!(times["remove outdated"], (time() - t0))

        # Record the current t value in the history
        append!(B.ts, B.params.t)

        # --- Step 9: Check stopping criterion (unless force_maxIt overrides it) ---
        if !(force_maxIt) && stopping_criteria(B)
            println("Satisfied stopping criteria")
            return true, times
        end

        push!(B.memorized["times"], time() - t1)
    end

    # Copy the per-iteration timing into the output dictionary
    times["times"] = B.memorized["times"]
    return false, times
end


"""
    t_strategy(B::DualBundle, i::Int, ts::abstract_t_strategy, unstable=false)

Decide whether to perform a Serious Step (SS) or a Null Step (NS), update the
stabilization point accordingly, and call the appropriate increment or decrement
of the regularization parameter `t`.

**Serious Step** (SS): triggered when the new trial point improves the stabilization
point objective by at least a fraction `m1` of the predicted improvement `vStar`:
    f(z_i) - f(zS) ≥ m1 * vStar
In this case, the stabilization point is moved to `z_i` and `increment_t` is called.

**Null Step** (NS): triggered when the improvement is insufficient.
The stabilization point is not moved and `decrement_t` is called.

Consecutive SS/NS counters (`B.CSS`, `B.CNS`) are maintained for use in long-term
t-strategies that react to trends in the step acceptance history.

# Arguments
- `B::DualBundle`: The bundle object.
- `i::Int`: Index of the newly added bundle component (typically `B.li`).
- `ts::abstract_t_strategy`: The t-strategy to apply.
- `unstable::Bool`: If `true`, always perform a Serious Step regardless of the
  objective improvement (default: `false`).
"""
function t_strategy(B::DualBundle, i::Int, ts::abstract_t_strategy, unstable::Bool = false)
    # --- Serious Step condition ---
    if B.obj[i] - objS(B) >= B.params.m1 * B.vStar || unstable
        # Update consecutive step counters
        B.CSS += 1
        B.CNS  = 0

        # Move the stabilization point to the new trial point
        B.s = i

        # Recompute all linearization errors w.r.t. the new stabilization point
        update_linearization_errors(B)

        # Ask the t-strategy whether to increase t after a Serious Step
        increment_t(B, ts)
    else
        # --- Null Step condition ---
        # Update consecutive step counters
        B.CNS += 1
        B.CSS  = 0

        # Ask the t-strategy whether to decrease t after a Null Step
        decrement_t(B, ts)
    end
end