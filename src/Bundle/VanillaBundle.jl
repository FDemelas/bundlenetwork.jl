"""
    initializeBundle(bt, ϕ, t, z; bp, max_bundle_size) -> Bundle

Construct and initialize a classical (Vanilla) `Bundle` for maximizing `ϕ`.

This is the standard aggregated proximal bundle method. It solves the Dual Master
Problem at each iteration to compute the search direction, and supports configurable
heuristic t-strategies for the regularization parameter.

The bundle is initialized at the starting point `z`: the oracle is called once to
compute the initial objective value and subgradient, all bundle matrices are
pre-allocated, the first DMP is built and solved, and the initial search direction
is extracted.

# Arguments
- `bt::VanillaBundleFactory`: Factory type used for dispatch.
- `ϕ::AbstractConcaveFunction`: The concave objective function to maximize.
- `t::Real`: Initial value of the regularization / step-size parameter.
- `z::AbstractArray`: Starting point for the optimization.

# Keyword Arguments
- `bp::BundleParameters`: Bundle hyperparameters (default: `BundleParameters()`).
- `max_bundle_size::Int`: Maximum number of bundle components to retain.
  Use `-1` (default) for unlimited (variable-size) storage.

# Returns
An initialized `Bundle` with the DMP solved and the initial search direction computed.

# Notes
- If `max_bundle_size > 0`, all matrices are pre-allocated to fixed size for efficiency.
  A small diagonal regularization `1e-5` is added to `Q[1,1]` to improve numerical stability.
- If `max_bundle_size <= 0`, matrices grow dynamically as new components are added.
- `B.sign` is set based on the sign of `ϕ`: if `sign(ϕ) != 1`, non-negativity constraints
  are added to the DMP (for Lagrangian relaxation with non-negative multipliers).
"""
function initializeBundle(bt::VanillaBundleFactory, ϕ::AbstractConcaveFunction, t::Real, z::AbstractArray; bp::BundleParameters = BundleParameters(), max_bundle_size = -1)

    # Allocate the bundle with placeholder values; all fields will be properly set below.
    # Notable defaults: objB = Inf (no bound yet), s = -1 (no stabilization point yet),
    # sign = false (non-negativity constraints disabled by default until checked below)
    B = Bundle(
        Float32[;;], Float32[;;], Float32[;;], [],
        -1, Model(Gurobi.Optimizer), [], [], Inf, [Inf], [Float32[]],
        bp, 0, 0, 1, Float32[], 1, [t],
        Dict("times" => []), 0.0, 0.0, 0.0, 0.0, false
    )

    # Oracle call: evaluate objective and subgradient at the starting point
    obj, g = value_gradient(ϕ, z)

    # The stabilization point starts at column 1 (the initialization point)
    B.s = 1

    # Flatten the subgradient to a 1D vector for storage
    g = reshape(g, :)

    # Store the maximum bundle size (controls whether fixed or variable storage is used)
    B.params.max_β_size = max_bundle_size

    if 0 < B.params.max_β_size < Inf
        # --- Fixed-size storage: pre-allocate all matrices to max_β_size ---

        # Visited trial points matrix (input_dim × max_β_size); column 1 = starting point
        B.z         = zeros(Float32, (length(z), B.params.max_β_size))
        B.z[:, 1]   = reshape(z, :)

        # Linearization errors (one per bundle slot); zero at initialization
        B.α         = zeros(Float32, B.params.max_β_size)

        # Subgradient matrix (input_dim × max_β_size); column 1 = subgradient at starting point
        B.G         = zeros(Float32, (length(g), B.params.max_β_size))
        B.G[:, 1]   = g

        # Objective values (one per bundle slot); entry 1 = objective at starting point
        B.obj       = zeros(Float32, B.params.max_β_size)
        B.obj[1]    = obj

        # Gram matrix Q = GᵀG (max_β_size × max_β_size); initialize with the first diagonal entry.
        # A small regularization term 1e-5 is added to prevent a singular Gram matrix
        # when the bundle has a single component.
        B.Q         = zeros(Float32, (B.params.max_β_size, B.params.max_β_size))
        B.Q[1, 1]   = g' * g + 1.0e-5
    else
        # --- Variable-size storage: start with a single component and grow dynamically ---
        B.α   = [0]                            # Single linearization error (zero at initialization)
        B.z   = reshape(z, (length(z), 1))     # Single column: starting point
        B.G   = reshape(g, (length(g), 1))     # Single column: subgradient at starting point
        B.Q   = Float32[g'g;;]                 # 1×1 Gram matrix
        B.obj = [obj]                           # Single entry: objective at starting point
    end

    # Store the initial step size parameter
    B.params.t = t

    # Record the initial t value in the step size history
    append!(B.ts, t)

    # Determine whether non-negativity constraints are needed in the DMP.
    # This is the case for sign-constrained dual problems (e.g., Lagrangian relaxation
    # with non-negative multipliers), detected via the sign of the objective function.
    B.sign = sign(ϕ) == 1 ? false : true

    # Build the initial Dual Quadratic Master Problem with the given t
    B.model = create_DQP(B, t)

    # Solve the DMP and extract the initial search direction
    solve_DQP(B)
    compute_direction(B)

    # Set the initial search direction as the weighted subgradient at the starting point
    # (θ from compute_direction weights the single initial bundle component)
    B.w = B.G[:, 1] .* B.θ

    # Initialize the cumulative θ history with a single-component simplex weight
    B.cumulative_θ = [Float32[1.0]]

    # Record the initial objective value in the full objective history
    push!(B.all_objs, obj)

    return B
end