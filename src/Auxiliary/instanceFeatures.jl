"""
Abstract type for all instance feature extraction strategies.
Subtypes define how problem-specific data (costs, right-hand sides, graph structure)
are encoded into feature vectors or matrices for use by the neural network model.
"""
abstract type abstractInstanceFeaturesType end


"""
Abstract type for graph-based instance feature extraction strategies.
Subtypes of this encode the problem instance as a graph (e.g., a bipartite constraint
graph) to be consumed by a Graph Neural Network (GNN).
"""
abstract type abstractGraphInstanceFeaturesType end


"""
Graph-based instance feature type for the Multi-Commodity Network Design (MCND) problem
that excludes commodity-resource (Cr) features from the graph encoding.
Used to select the `size_ins_features` and `featuresExtraction` implementations
that omit those features.
"""
struct noCrGraphInstanceFeaturesType <: abstractGraphInstanceFeaturesType end


"""
    size_ins_features(ϕ::LagrangianFunctionMCND, fmt::noCrGraphInstanceFeaturesType) -> Int

Return the number of feature rows per node type in the instance feature matrix
for the `noCrGraphInstanceFeaturesType` encoding.

Returns `1`, meaning each node type (row constraints, knapsack constraints,
continuous variables, integer variables) is described by a single scalar feature.
"""
function size_ins_features(ϕ::LagrangianFunctionMCND, fmt::noCrGraphInstanceFeaturesType)
    return 1
end


"""
    features_matrix(ϕ::LagrangianFunctionMCND, fmt::abstractInstanceFeaturesType) -> Matrix

Build the node feature matrix for the MCND Lagrangian instance `ϕ`.

Each column corresponds to one variable or constraint node in the bipartite graph;
each row group encodes a different type of problem data:
- Row 1: Right-hand side (RHS) of routing constraints (one entry per routing constraint node).
- Row 2: RHS of knapsack constraints (one entry per knapsack constraint node).
- Row 3: Cost of continuous (flow) variables (one entry per continuous variable node).
- Row 4: Cost of integer (design) variables (one entry per integer variable node).

The total number of columns is `s_cv + s_rc + s_kc + s_iv` (all variable and constraint types).
The total number of rows is `4 * size_ins_features(ϕ, fmt)`.

# Arguments
- `ϕ::LagrangianFunctionMCND`: The Lagrangian relaxation of an MCND instance.
- `fmt::abstractInstanceFeaturesType`: Feature extraction format (determines row count per type).

# Returns
A `(4 * size_ins_features) × (s_cv + s_rc + s_kc + s_iv)` Float64 matrix.
"""
function features_matrix(ϕ::LagrangianFunctionMCND, fmt::abstractInstanceFeaturesType)
    # Retrieve the number of each variable/constraint type in the MCND instance
    s_cv = get_number_cv(ϕ)   # Number of continuous (flow) variables
    s_iv = get_number_iv(ϕ)   # Number of integer (design) variables
    s_rc = get_number_rc(ϕ)   # Number of routing (flow conservation) constraints
    s_kc = get_number_kc(ϕ)   # Number of knapsack (capacity) constraints

    # Allocate the feature matrix: 4 feature groups × total number of nodes
    f   = zeros(4 * size_ins_features(ϕ, fmt), s_cv + s_rc + s_kc + s_iv)
    idx = 1   # Running column index across all node types

    # Row 1: RHS of routing constraints (one column per routing constraint node)
    for i in 1:s_rc
        f[1, idx] = get_rhs_rc(ϕ, i)
        idx += 1
    end

    # Row 2: RHS of knapsack (capacity) constraints (one column per knapsack constraint node)
    for i in 1:s_kc
        f[2, idx] = get_rhs_kc(ϕ, i)
        idx += 1
    end

    # Row 3: Costs of continuous (flow) variables (one column per flow variable node)
    for i in 1:s_cv
        f[3, idx] = get_costs_cv(ϕ, i)
        idx += 1
    end

    # Row 4: Costs of integer (design) variables (one column per design variable node)
    for i in 1:s_iv
        f[4, idx] = get_costs_iv(ϕ, i)
        idx += 1
    end

    return f
end


"""
    from_couple_to_idx(j::Int, i::Int, maxI::Int) -> Int64

Convert a (row, column) pair `(j, i)` to a flat (row-major) linear index,
assuming the inner dimension has size `maxI`.

The mapping is: `idx = (j - 1) * maxI + i`

Used to convert commodity-edge or commodity-node pairs into a single integer
index for accessing flattened problem data arrays.

# Arguments
- `j`: Outer index (e.g., commodity index).
- `i`: Inner index (e.g., edge or node index), ranging from `1` to `maxI`.
- `maxI`: Size of the inner dimension.
"""
function from_couple_to_idx(j::Int, i::Int, maxI::Int)
    return Int64((j - 1) * maxI + i)
end


"""
    from_idx_to_couple(ji::Int, maxI::Int) -> (Int64, Int64)

Convert a flat linear index `ji` back to a (row, column) pair `(j, i)`,
assuming the inner dimension has size `maxI`. Inverse of `from_couple_to_idx`.

The mapping is:
- `i = ji mod maxI` (with `i = maxI` when `ji` is a multiple of `maxI`)
- `j = (ji - i) / maxI + 1`

# Arguments
- `ji`: Flat linear index.
- `maxI`: Size of the inner dimension.

# Returns
A tuple `(j, i)` where `j` is the outer index and `i` is the inner index.
"""
function from_idx_to_couple(ji::Int, maxI::Int)
    i = (ji % maxI)
    i = i ≈ 0 ? maxI : i     # Handle the case where ji is an exact multiple of maxI
    j = (ji - i) / maxI + 1
    return Int64(j), Int64(i)
end


"""
    get_costs_cv(ϕ::LagrangianFunctionMCND, i::Int) -> Real

Return the cost coefficient of the `i`-th continuous (flow) variable in the MCND instance.

The continuous variables represent flow on each arc for each commodity.
The flat index `i` is converted to an (edge, commodity) pair `(e, k)` using
`from_idx_to_couple`, and the cost is read from `ϕ.inst.r[k, e]`.

# Arguments
- `ϕ`: The Lagrangian MCND function containing the problem instance.
- `i`: Flat index of the continuous variable (1-based, ranging over all edge-commodity pairs).
"""
function get_costs_cv(ϕ::LagrangianFunctionMCND, i::Int)
    # Convert flat index to (edge, commodity) pair using the number of commodities as inner dim
    e, k = from_idx_to_couple(i, sizeK(ϕ.inst))
    return ϕ.inst.r[k, e]   # Unit routing cost for commodity k on edge e
end


"""
    get_costs_iv(ϕ::LagrangianFunctionMCND, i::Int) -> Real

Return the fixed cost of the `i`-th integer (design/arc-opening) variable in the MCND instance.

The integer variables represent arc-opening decisions. Their costs are stored
in `ϕ.inst.f` indexed directly by arc.

# Arguments
- `ϕ`: The Lagrangian MCND function containing the problem instance.
- `i`: Index of the integer variable (1-based, one per arc).
"""
function get_costs_iv(ϕ::LagrangianFunctionMCND, i::Int)
    return ϕ.inst.f[i]   # Fixed arc-opening cost for arc i
end


"""
    get_rhs_rc(ϕ::LagrangianFunctionMCND, i::Int) -> Real

Return the right-hand side of the `i`-th routing (flow conservation) constraint
in the MCND instance.

Routing constraints enforce flow conservation at each node for each commodity.
The flat index `i` is converted to a (commodity, node) pair `(k, v)`, and the
RHS is retrieved via `b(ϕ.inst, k, v)` (the supply/demand of commodity `k` at node `v`).

# Arguments
- `ϕ`: The Lagrangian MCND function containing the problem instance.
- `i`: Flat index of the routing constraint (1-based, ranging over all commodity-node pairs).
"""
function get_rhs_rc(ϕ::LagrangianFunctionMCND, i::Int)
    # Convert flat index to (commodity, node) pair
    k, v = from_idx_to_couple(i, sizeK(ϕ.inst))
    return b(ϕ.inst, k, v)   # Supply/demand value for commodity k at node v
end


"""
    get_rhs_kc(ϕ::LagrangianFunctionMCND, i::Int) -> Real

Return the right-hand side of the `i`-th knapsack (arc capacity) constraint
in the MCND instance.

Knapsack constraints enforce that total flow on each arc does not exceed its capacity.
The RHS is the capacity of arc `i`, stored in `ϕ.inst.c`.

# Arguments
- `ϕ`: The Lagrangian MCND function containing the problem instance.
- `i`: Index of the knapsack constraint (1-based, one per arc).
"""
function get_rhs_kc(ϕ::LagrangianFunctionMCND, i::Int)
    return ϕ.inst.c[i]   # Capacity of arc i
end


"""
    get_number_cv(ϕ::LagrangianFunctionMCND) -> Int

Return the total number of continuous (flow) variables in the MCND instance.
This equals the number of arcs times the number of commodities: `|E| × |K|`.
"""
function get_number_cv(ϕ::LagrangianFunctionMCND)
    return sizeE(ϕ.inst) * sizeK(ϕ.inst)
end


"""
    get_number_iv(ϕ::LagrangianFunctionMCND) -> Int

Return the total number of integer (design/arc-opening) variables in the MCND instance.
This equals the number of arcs: `|E|`.
"""
function get_number_iv(ϕ::LagrangianFunctionMCND)
    return sizeE(ϕ.inst)
end


"""
    get_number_rc(ϕ::LagrangianFunctionMCND) -> Int

Return the total number of routing (flow conservation) constraints in the MCND instance.
This equals the number of commodities times the number of nodes: `|K| × |V|`.
"""
function get_number_rc(ϕ::LagrangianFunctionMCND)
    return sizeK(ϕ.inst) * sizeV(ϕ.inst)
end


"""
    get_number_kc(ϕ::LagrangianFunctionMCND) -> Int

Return the total number of knapsack (arc capacity) constraints in the MCND instance.
This equals the number of arcs: `|E|`.
"""
function get_number_kc(ϕ::LagrangianFunctionMCND)
    return sizeE(ϕ.inst)
end


"""
    get_number_c(ϕ::LagrangianFunctionMCND) -> Int

Return the total number of constraints in the MCND instance,
combining routing and knapsack constraints:
    `get_number_rc(ϕ) + get_number_kc(ϕ)` = `|K|×|V| + |E|`
"""
function get_number_c(ϕ::LagrangianFunctionMCND)
    return get_number_rc(ϕ) + get_number_kc(ϕ)
end


"""
    number_non_zeros_coefficients(ϕ::LagrangianFunctionMCND) -> Int

Return the number of non-zero coefficients in the MCND constraint matrix.

Each arc contributes:
- 1 non-zero in each knapsack constraint (the capacity row for that arc).
- 2 non-zeros per commodity in the routing constraints (one at the tail node,
  one at the head node of the arc), giving `2 × |K|` per arc.

Total: `|E| × (1 + 2 × |K|)`.

!!! note
    The formula in the code uses `3 * sizeK(ϕ.inst)` rather than `2 * sizeK + 1`;
    this may be a conservative overestimate or reflect a slightly different constraint
    structure in the specific MCND formulation used.
"""
function number_non_zeros_coefficients(ϕ::LagrangianFunctionMCND)
    return sizeE(ϕ.inst) * (1 + 3 * sizeK(ϕ.inst))
end


"""
    preprocess_weight(fmt::abstractInstanceFeaturesType, weight::Real) -> Real

Preprocess a constraint matrix coefficient before storing it as an edge weight
in the bipartite graph representation.

Applies an exponential transformation: `exp(weight)`. This maps signed coefficients
(which can be `-1`, `0`, or `+1` in network flow problems) to strictly positive
edge weights suitable for GNN message passing.

# Arguments
- `fmt`: The feature extraction format (currently unused; reserved for format-specific transforms).
- `weight`: The raw constraint matrix coefficient to transform.
"""
function preprocess_weight(fmt::abstractInstanceFeaturesType, weight::Real)
    return exp(weight)   # Map coefficient to a positive edge weight via exponentiation
end


"""
    get_coefficient(ϕ::LagrangianFunctionMCND, i_c::Int, i_v::Int) -> Float64

Return the coefficient of variable `i_v` in constraint `i_c` of the MCND problem.

The MCND constraint matrix has a block structure:
- Constraints are indexed as: first `get_number_rc(ϕ)` routing constraints,
  then `get_number_kc(ϕ)` knapsack constraints.
- Variables are indexed as: first `get_number_c(ϕ)` entries are reserved (not variables),
  then `get_number_cv(ϕ)` continuous variables, then `get_number_iv(ϕ)` integer variables.

Returns:
- For routing constraints × continuous variables: the arc-node incidence coefficient
  (`+1` if the arc's head is the constraint's node, `-1` if the tail, `0` otherwise),
  restricted to matching commodities.
- For knapsack constraints × continuous variables: `1.0` if the arc indices match, else `0.0`.
- For knapsack constraints × integer variables: `-c[e]` if arc indices match (capacity
  coefficient), else `0.0`.
- `0.0` for routing constraints × integer variables (no incidence).

Prints a warning and returns `0.0` for out-of-range indices.

# Arguments
- `ϕ`: The Lagrangian MCND function.
- `i_c::Int`: Constraint index (1-based, first routing then knapsack).
- `i_v::Int`: Variable index (1-based, offset past the constraint count).
"""
function get_coefficient(ϕ::LagrangianFunctionMCND, i_c::Int, i_v::Int)
    # Guard: constraint index must be within the valid range
    if i_c > get_number_c(ϕ)
        println(" Something wrong with indexes !")
        return 0.0
    else
        # Guard: variable index must be past the constraint-count offset
        if i_v <= get_number_c(ϕ)
            println(" Something wrong with indexes !")
            return 0.0
        else
            if i_v <= get_number_c(ϕ) + get_number_cv(ϕ)
                # --- Variable is a continuous (flow) variable ---
                if i_c <= get_number_rc(ϕ)
                    # Routing constraint × continuous variable:
                    # Check if the arc (e_v, k_v) appears in the flow conservation
                    # at node v_c for commodity k_c
                    v_c, k_c = from_idx_to_couple(i_c, sizeK(ϕ.inst))
                    idx_v    = i_v - get_number_c(ϕ)   # Re-index to continuous variable space
                    e_v, k_v = from_idx_to_couple(idx_v, sizeK(ϕ.inst))

                    # Incidence coefficient: -1 if arc e_v leaves node v_c, +1 if it enters
                    coeff = tail(ϕ.inst, e_v) == v_c ? -1.0 :
                            (head(ϕ.inst, e_v) == v_c ? 1.0 : 0.0)

                    # Non-zero only if the commodity indices match
                    return k_c == k_v ? coeff : 0.0
                else
                    # Knapsack constraint × continuous variable:
                    # The constraint for arc e_c involves flow variable (e_v, any commodity)
                    e_c   = i_c - get_number_rc(ϕ)   # Arc index of the knapsack constraint
                    idx_v = i_v - get_number_rc(ϕ) - get_number_kc(ϕ)   # Re-index
                    e_v, _ = from_idx_to_couple(idx_v, sizeInputSpace(ϕ)[1])

                    # Non-zero only if the arc indices match (coefficient is 1)
                    return e_v == e_c ? 1.0 : 0.0
                end
            else
                # --- Variable is an integer (design/arc-opening) variable ---
                if i_c <= get_number_rc(ϕ)
                    # Routing constraint × integer variable: always 0
                    # (integer variables do not appear in flow conservation constraints)
                    return 0.0
                else
                    # Knapsack constraint × integer variable:
                    # The coefficient is -c[e_c] if the arc indices match
                    idx_v = i_v - get_number_rc(ϕ) - get_number_kc(ϕ) - get_number_cv(ϕ)
                    e_v, _ = from_idx_to_couple(idx_v, sizeK(ϕ))
                    e_c    = i_c - get_number_rc(ϕ)   # Arc index of the knapsack constraint

                    # Capacity coefficient: -c[e_c] links arc capacity to the design variable
                    return e_v == e_c ? -ϕ.inst.c[e_c] : 0.0
                end
            end
        end
    end
end


"""
    featuresExtraction(ϕ::LagrangianFunctionMCND, fmt::abstractGraphInstanceFeaturesType) -> GNNGraph

Build a bipartite GNN graph representing the MCND constraint matrix for use as
input to a Graph Neural Network model.

The graph has two types of nodes:
- **Constraint nodes**: routing constraint nodes (`nodesRC`) and knapsack constraint
  nodes (`nodesKC`), indexed `1` to `s_rc + s_kc`.
- **Variable nodes**: continuous variable nodes (`nodesx`) and integer variable
  nodes (`nodesy`), indexed `s_rc + s_kc + 1` to `s_rc + s_kc + s_cv + s_iv`.

Edges connect each constraint node to each variable node for which the constraint
matrix coefficient is non-zero. Each edge is added in **both directions** (constraint→variable
and variable→constraint) to form an undirected bipartite graph suitable for GNN message passing.

Edge weights are the preprocessed constraint coefficients (via `preprocess_weight`).
Node features are provided by `features_matrix`.
Self-loops are added to all nodes at the end (standard GNN practice for node self-aggregation).

# Arguments
- `ϕ::LagrangianFunctionMCND`: The Lagrangian MCND function containing the problem instance.
- `fmt::abstractGraphInstanceFeaturesType`: The graph feature extraction format,
  controlling node feature dimensionality.

# Returns
A `GNNGraph` with:
- Node features from `features_matrix(ϕ, fmt)`.
- Bidirectional edges for all non-zero constraint matrix entries, with preprocessed weights.
- Global data: `[sizeInputSpace(ϕ)]` (the input space dimension).
- Self-loops added to all nodes.
"""
function featuresExtraction(ϕ::LagrangianFunctionMCND, fmt::abstractGraphInstanceFeaturesType)
    # Retrieve the number of each variable/constraint type
    s_cv = get_number_cv(ϕ)   # Number of continuous (flow) variables
    s_iv = get_number_iv(ϕ)   # Number of integer (design) variables
    s_rc = get_number_rc(ϕ)   # Number of routing constraints
    s_kc = get_number_kc(ϕ)   # Number of knapsack constraints

    # Compute the cumulative node index boundaries for each node type
    # Node indices are 1-based and contiguous across types
    lnfc = s_rc + 1              # First index of knapsack constraint nodes
    lnkc = lnfc + s_kc           # First index of continuous variable nodes
    lnx  = lnkc + s_cv           # First index of integer variable nodes
    lny  = lnx + s_iv            # One past the last node index

    # Collect node index ranges for each type
    nodesRC = collect(1:(lnfc - 1))       # Routing constraint node indices
    nodesKC = collect(lnfc:(lnkc - 1))   # Knapsack constraint node indices
    nodesx  = collect(lnkc:(lnx - 1))    # Continuous variable node indices
    nodesy  = collect(lnx:(lny - 1))     # Integer variable node indices

    # Pre-allocate edge arrays for the bidirectional bipartite graph.
    # Each non-zero coefficient contributes 2 directed edges (both directions).
    sizeBiArcs = 2 * number_non_zeros_coefficients(ϕ)
    tails   = zeros(Int64,   sizeBiArcs)   # Source node of each directed edge
    heads   = zeros(Int64,   sizeBiArcs)   # Target node of each directed edge
    weightsE = zeros(Float32, sizeBiArcs)  # Preprocessed weight of each directed edge

    tmp = 1   # Running index into the edge arrays

    # Iterate over all (constraint node, variable node) pairs and add edges
    # for non-zero constraint matrix coefficients
    for i in union(nodesRC, nodesKC)        # For each constraint node
        for j in union(nodesx, nodesy)       # For each variable node
            s, e   = i, j
            weight = get_coefficient(ϕ, i, j)   # Raw constraint matrix coefficient

            if !(weight ≈ 0)
                # Add the forward directed edge: constraint node → variable node
                tails[tmp]    = s
                heads[tmp]    = e
                weightsE[tmp] = preprocess_weight(fmt, weight)
                tmp += 1

                # Add the backward directed edge: variable node → constraint node
                # This makes the graph undirected (bidirectional), as required by most GNNs
                tails[tmp]    = e
                heads[tmp]    = s
                weightsE[tmp] = preprocess_weight(fmt, weight)
                tmp += 1
            end
        end
    end

    # Build the node feature matrix (one column per node, encoding problem data)
    f = features_matrix(ϕ, fmt)

    # Construct the GNN graph with node features and global instance-level data
    g = GNNGraph(
        tails, heads, weightsE,
        ndata = f,                         # Node feature matrix
        gdata = [sizeInputSpace(ϕ)]        # Global graph-level feature: input space size
    )

    # Add self-loops to all nodes (standard practice in GNNs to include self-information
    # in the aggregation step)
    return add_self_loops(g)
end