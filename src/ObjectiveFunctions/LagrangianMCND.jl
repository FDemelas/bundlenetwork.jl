"""
    LagrangianFunctionMCND <: AbstractLagrangianFunction

Lagrangian subproblem function for the Multi-Commodity Network Design (MCND) problem
using capacity-based Lagrangian relaxation.

# Problem Context
Multi-Commodity Network Design (MCND) involves:
- **Network**: Directed graph with nodes and arcs
- **Commodities**: K different commodities with origin-destination pairs
- **Flows**: Route each commodity through the network
- **Capacities**: Each arc has limited capacity (shared among commodities)
- **Costs**: Arc usage has associated costs
- **Objective**: Minimize total cost while satisfying all demands

# Lagrangian Relaxation Approach
The capacity constraints are relaxed using Lagrange multipliers:

1. **Original Problem** (hard):
   ```
   min  Σ cost × flow
   s.t. Flow conservation for each commodity
        Σₖ flow[k,ij] ≤ capacity[ij]  (capacity constraints)
        flow ≥ 0
   ```

2. **Lagrangian Relaxation**:
   - Dualize capacity constraints with multipliers z[k,i]
   - Decompose into K independent shortest path problems
   - Each commodity: Find min-cost path with modified arc costs

# Mathematical Formulation
For dual variables z ∈ ℝ^(K×N) (where K = commodities, N = nodes):

```
L(z) = Σₖ min { cost' x[k] + z[k]' (b[k] - Ax[k]) : x[k] ≥ 0 }
```

Where:
- x[k]: Flow variables for commodity k
- A: Node-arc incidence matrix (flow conservation)
- b[k]: Demand vector for commodity k (source/sink/transit)
- z[k,i]: Dual variable for node i, commodity k

# Decomposition Structure
The Lagrangian decomposes into **K independent shortest path problems**:

```
For each commodity k = 1, ..., K:
    Find shortest path from origin[k] to destination[k]
    with modified arc costs: cost[ij] + z[k,tail[ij]] - z[k,head[ij]]
```

This decomposition enables:
- **Parallel solving**: Each commodity solved independently
- **Efficiency**: Shortest path algorithms (Dijkstra, Bellman-Ford)
- **Scalability**: Linear in number of commodities

# Fields
- `inst::cpuInstanceMCND`: Problem instance (from Instances package)
  - Contains: K (commodities), N (nodes), E (arcs)
  - Network structure, costs, capacities, demands
- `rescaling_factor::Real`: Divisor for scaling objective values

# Subgradient Interpretation
The subgradient at z represents **flow conservation violations**:

```
∂L(z)/∂z[k,i] = inflow[k,i] - outflow[k,i] - demand[k,i]
```

Where:
- inflow: Total flow into node i for commodity k
- outflow: Total flow out of node i for commodity k  
- demand: Required net flow at node i for commodity k

At optimal dual variables:
- Flow conservation satisfied approximately
- Bounds primal optimal value

# Dual Problem
The dual problem provides an upper bound for the primal:

```
maximize  L(z)
   z

L(z*) ≥ optimal_primal_value
```

Solving via subgradient methods tightens this bound.

# Example Usage
```julia
# Load MCND instance
using Instances
inst = load_MCND_instance("data/mcnd_network.dat")

# Create Lagrangian function
lagrangian = constructFunction(inst, rescaling_factor=100.0)

# Evaluate at zero duals (LP relaxation bound)
z = zeros(Float32, inst.K, inst.N)
bound = lagrangian(z)

# Compute subgradient
obj, grad = value_gradient(lagrangian, z)

# Subgradient ascent step
learning_rate = 0.01
z += learning_rate * grad
```

# Network Structure Example
```
   1 ----→ 2
   |  ╲    ↓
   ↓    ╲  3
   4 ----→ 5

Commodity 1: 1 → 5 (demand = 10)
Commodity 2: 2 → 4 (demand = 5)

Arc capacities: shared across commodities
```

# Comparison with LagrangianFunctionGA
| Aspect | GA (Knapsack) | MCND (Network Flow) |
|--------|---------------|---------------------|
| Relaxation | Capacity | Capacity |
| Subproblems | J knapsacks | K shortest paths |
| Variables | Binary selection | Continuous flow |
| Algorithm | Dynamic programming | Dijkstra/Bellman-Ford |
| Complexity | Pseudo-polynomial | Polynomial |

# See Also
- `Instances` package: https://github.com/FDemelas/Instances
- `constructFunction`: Constructor
- `ChainRulesCore.rrule`: Gradient computation
- Shortest path algorithms
- Lagrangian relaxation for network design
"""
mutable struct LagrangianFunctionMCND <: AbstractLagrangianFunction
    inst::cpuInstanceMCND
    rescaling_factor::Real
end

"""
    (l::LagrangianFunctionMCND)(z::AbstractArray)

Forward pass: evaluates the Lagrangian subproblem at dual variables z.

# Arguments
- `z::AbstractArray`: Lagrange multipliers (dual variables)
  - Shape: (K, N) where K = commodities, N = nodes
  - z[k,i]: Dual variable for node i, commodity k

# Returns
- Lagrangian value: L(z) / rescaling_factor

# Process
1. **Convert to CPU**: Move z to CPU (network algorithms on CPU)
2. **Solve subproblems**: Call LR function (solves K shortest paths)
3. **Extract objective**: First element of LR return
4. **Scale**: Divide by rescaling_factor

# Mathematical Operation
```
L(z) = Σₖ min { Σᵢⱼ (cost[ij] + z[k,tail[ij]] - z[k,head[ij]]) × x[k,ij] }
       s.t. Flow conservation constraints
            x[k,ij] ≥ 0
```

For each commodity k:
- Modified arc costs: cost[ij] + z[k,tail[ij]] - z[k,head[ij]]
- Find shortest path with these costs
- Shortest path cost contributes to L(z)

# Implementation via LR Function
The `LR` function from Instances package:
- Solves all K shortest path problems
- Returns [objective_value, flow_solutions, ...]
- Uses efficient shortest path algorithms

# Computational Complexity
- **Time**: O(K × (E log N)) with Dijkstra's algorithm
  - K commodities, each solved independently
  - E arcs, N nodes
  - Dijkstra: O(E log N) per commodity
- **Space**: O(K × E) for flow solutions

# Example
```julia
# Create Lagrangian
inst = load_MCND_instance("network.dat")
lagrangian = constructFunction(inst, 50.0)

# Zero duals (relaxed problem)
K, N = inst.K, inst.N
z_zero = zeros(Float32, K, N)
bound_zero = lagrangian(z_zero)

# Optimized duals (tighter bound)
z_opt = optimize_duals(lagrangian)
bound_opt = lagrangian(z_opt)

println("LP bound: ", bound_zero)
println("Lagrangian bound: ", bound_opt)
# bound_opt ≥ bound_zero (tighter)
```

# Modified Arc Costs
For each arc (i,j) and commodity k:
```
modified_cost[k,ij] = original_cost[ij] + z[k,i] - z[k,j]
```

This represents:
- Original arc cost
- "Penalty" for leaving node i (+ z[k,i])
- "Reward" for entering node j (- z[k,j])

# Shortest Path Interpretation
Each subproblem finds the shortest path from source[k] to sink[k]
with modified costs. The path represents optimal flow for commodity k
given the current dual prices.

# CPU Requirement
⚠️ Network flow algorithms typically CPU-only:
- Graph algorithms not easily parallelizable on GPU
- CPU implementations highly optimized
- Transfer overhead negligible compared to solving time

# See Also
- `Instances.LR`: Core Lagrangian relaxation solver
- `ChainRulesCore.rrule`: Gradient computation
- Shortest path algorithms (Dijkstra, Bellman-Ford)
"""
function (l::LagrangianFunctionMCND)(z::AbstractArray)
    return LR(l.inst, cpu(z))[1] / l.rescaling_factor
end

# Declare as Flux layer for proper integration
Flux.@layer LagrangianFunctionMCND

"""
    constructFunction(inst::cpuInstanceMCND, rescaling_factor::Real=1.0)

Constructs a Lagrangian function from an MCND problem instance.

# Arguments
- `inst::cpuInstanceMCND`: MCND problem instance from Instances package
  - Must contain: K (commodities), N (nodes), E (arcs)
  - Network topology, costs, capacities, demands
- `rescaling_factor::Real`: Scaling factor for objective (default: 1.0)

# Returns
- `LagrangianFunctionMCND`: Callable Lagrangian function

# Instance Requirements
The `cpuInstanceMCND` instance must contain:
- `K::Int`: Number of commodities
- `N::Int`: Number of nodes in network
- `E::Int`: Number of arcs (directed edges)
- Network structure: head, tail functions for arcs
- `cost::Vector{Float32}`: Arc costs (E,)
- `capacity::Vector{Float32}`: Arc capacities (E,)
- Demand information: b function for node demands

# Rescaling Factor Guidelines
Used to normalize Lagrangian values:
- **Small networks** (K, N, E < 100): 1.0 - 10.0
- **Medium networks** (K, N, E < 1000): 10.0 - 100.0
- **Large networks** (K, N, E > 1000): 100.0 - 1000.0

Choose to keep L(z) values in range [0.1, 1000] for numerical stability.

# Example - Basic Usage
```julia
using Instances

# Load problem instance
inst = load_MCND_instance("data/mcnd_20c_50n_200e.dat")
# 20 commodities, 50 nodes, 200 arcs

# Create Lagrangian
lagrangian = constructFunction(inst, rescaling_factor=50.0)

# Verify dimensions
@assert sizeInputSpace(lagrangian) == (inst.K, inst.N)
@assert numberSP(lagrangian) == inst.E
```

# Example - Creating Custom Instance
```julia
# Define simple network
#   1 → 2 → 3
#   ↓       ↑
#   4 → 5 →┘

K = 2  # Two commodities
N = 5  # Five nodes
E = 6  # Six arcs

inst = cpuInstanceMCND(
    K = K,
    N = N,
    E = E,
    # Define arcs, costs, capacities, demands...
)

lagrangian = constructFunction(inst)
```

# Example - Complete Workflow
```julia
# Load instance
inst = load_MCND_instance("test_network.dat")
lagrangian = constructFunction(inst, 100.0)

# Initialize dual variables
K, N = sizeLM(inst)
z = zeros(Float32, K, N)

# Subgradient optimization
for iter in 1:1000
    # Evaluate and get subgradient
    obj, grad = value_gradient(lagrangian, z)
    
    # Subgradient step with decreasing step size
    step_size = 1.0 / sqrt(iter)
    z += step_size * grad
    
    # Project onto non-negative orthant (if needed)
    # z = max.(z, 0)
    
    if iter % 100 == 0
        println("Iteration $iter: Bound = $obj")
    end
end

println("Final Lagrangian bound: ", lagrangian(z))
```

# Network Topology Functions
The instance provides helper functions:
- `head(inst, arc)`: Head node of arc
- `tail(inst, arc)`: Tail node of arc
- `b(inst, node, commodity)`: Demand at node for commodity
- `sizeE(inst)`: Number of arcs
- `sizeLM(inst)`: Dual variable dimensions

# See Also
- `Instances` package documentation
- `LagrangianFunctionMCND`: Resulting structure
- `sizeInputSpace`: Verify dual dimensions
- `numberSP`: Verify arc count
"""
function constructFunction(inst::cpuInstanceMCND, rescaling_factor::Real=1.0)
    return LagrangianFunctionMCND(inst, rescaling_factor)    
end

"""
    ChainRulesCore.rrule(ϕ::LagrangianFunctionMCND, z::AbstractArray)

Custom backward pass for MCND Lagrangian gradient computation.

# Arguments
- `ϕ::LagrangianFunctionMCND`: The Lagrangian function
- `z::AbstractArray`: Dual variables (K × N matrix)

# Returns
- `obj`: Forward pass result (Lagrangian value)
- `loss_pullback`: Function for backward gradient computation

# Subgradient Formula
The subgradient of the Lagrangian at z is the **flow imbalance** at each node:

```
∂L(z)/∂z[k,i] = Σⱼ:(j→i) x*[k,ij] - Σⱼ:(i→j) x*[k,ij] - b[k,i]
```

Where:
- First sum: Total inflow to node i for commodity k
- Second sum: Total outflow from node i for commodity k
- b[k,i]: Required net supply at node i for commodity k (demand)
- x*[k,ij]: Optimal flow on arc (i,j) for commodity k

# Physical Interpretation
Each gradient component represents:
- **> 0**: Net inflow exceeds demand → node "oversupplied"
- **< 0**: Net outflow exceeds supply → node "undersupplied"  
- **= 0**: Flow conservation satisfied (optimality condition)

# Why Custom Implementation?
1. **Analytical gradient**: Exact subgradient from flow solutions
2. **Efficiency**: Solutions already computed in forward pass
3. **Network structure**: Exploit problem structure
4. **Exactness**: No numerical approximation

# Process

## Forward Pass
1. **Move to CPU**: z = cpu(z)
2. **Solve LR**: obj, x, _ = LR(inst, z)
   - obj: Lagrangian value
   - x: Flow solutions (K × E matrix)
3. **Initialize gradient**: grad = zeros(K, N)
4. **Compute flow imbalances**: For each commodity k, node i:
   ```
   grad[k,i] = (inflow - outflow - demand)
   ```

## Backward Pass (loss_pullback)
Computes gradient of loss w.r.t. dual variables:
```
∂loss/∂z = grad / rescaling_factor × dl
```

# Implementation Details

## Flow Imbalance Computation
```julia
for k in 1:K
    for i in 1:N
        # Inflow: arcs ending at i
        inflow = sum([x[k,ij] for ij in arcs where tail(ij) == i])
        
        # Outflow: arcs starting at i
        outflow = sum([x[k,ij] for ij in arcs where head(ij) == i])
        
        # Demand at node i for commodity k
        demand = b(inst, i, k)
        
        # Gradient component
        grad[k,i] = inflow - outflow - demand
    end
end
```

## Network Structure Functions
- `head(inst, ij)`: Returns head node of arc ij (destination)
- `tail(inst, ij)`: Returns tail node of arc ij (origin)
- `b(inst, i, k)`: Returns demand at node i for commodity k
  - Positive: Supply/source node
  - Negative: Demand/sink node
  - Zero: Transshipment node

# Mathematical Justification
By envelope theorem, for Lagrangian:
```
L(z) = min_x { c'x + z'(b - Ax) : x ≥ 0 }
```

The subgradient is:
```
∂L/∂z = b - Ax*
```

Which represents flow conservation violations in the optimal flow x*.

# Subgradient Properties
The computed subgradient satisfies:
- **Validity**: Always a valid subgradient at z
- **Direction**: Points toward improved dual bound
- **Sparsity**: Often sparse (many zero components)
- **Boundedness**: Bounded by network structure and demands

# Computational Complexity
- **Forward pass**: O(K × E log N) - Solve K shortest paths
- **Backward pass**: O(K × E × N) - Compute all imbalances
  - For each of K commodities
  - For each of N nodes
  - Check all E arcs (can be optimized with adjacency lists)
- **Memory**: O(K × E) - Store flow solutions

# Optimization Opportunity
Current implementation checks all arcs for each node.
Can optimize using adjacency lists:
```julia
# Precompute adjacency
incoming_arcs[i] = [ij for ij where tail(ij) == i]
outgoing_arcs[i] = [ij for ij where head(ij) == i]

# Faster computation
inflow = sum(x[k, ij] for ij in incoming_arcs[i])
outflow = sum(x[k, ij] for ij in outgoing_arcs[i])
```

# Example Usage
```julia
# Automatic during backpropagation
z = randn(Float32, K, N)

# Forward and backward
obj, pullback = ChainRulesCore.rrule(lagrangian, z)

# Compute gradient (called by autodiff)
dl = 1.0
grad_z = pullback(dl)[2]

# Interpret gradient
println("Max flow imbalance: ", maximum(abs.(grad_z)))
println("Nodes with imbalance: ", count(abs.(grad_z) .> 1e-6))

# Dual update
z_new = z + learning_rate * grad_z
```

# Convergence to Optimality
As dual optimization progresses:
- Subgradient magnitudes typically decrease
- Flow imbalances approach zero
- Lagrangian bound approaches primal optimum
- Flow solutions approach feasibility

# See Also
- Envelope theorem for Lagrangian gradients
- Network flow theory (conservation laws)
- Subgradient methods for non-smooth optimization
- `Instances.LR`: Network flow solver
"""
function ChainRulesCore.rrule(ϕ::LagrangianFunctionMCND, z::AbstractArray)
    # Move dual variables to CPU
    z = cpu(z)
    
    # Solve Lagrangian relaxation
    # obj: Lagrangian value
    # x: Flow solutions (K × E matrix)
    # _: Additional info (unused)
    obj, x, _ = LR(ϕ.inst, reshape(z, sizeInputSpace(ϕ)))
    
    # Initialize gradient matrix (K commodities × N nodes)
    grad = zeros(Float32, sizeInputSpace(ϕ))
    
    # Compute flow imbalance at each node for each commodity
    for k in 1:sizeInputSpace(ϕ)[1]  # For each commodity
        for i in 1:sizeInputSpace(ϕ)[2]  # For each node
            # Compute inflow: sum of flows on arcs ending at node i
            inflow = sum([x[k, ij] for ij in 1:sizeE(ϕ.inst) 
                         if tail(ϕ.inst, ij) == i])
            
            # Compute outflow: sum of flows on arcs starting at node i
            outflow = sum([x[k, ij] for ij in 1:sizeE(ϕ.inst) 
                          if head(ϕ.inst, ij) == i])
            
            # Demand at this node for this commodity
            demand = b(ϕ.inst, i, k)
            
            # Subgradient: flow imbalance = inflow - outflow - demand
            # Positive: oversupplied, Negative: undersupplied, Zero: balanced
            grad[k, i] = inflow - outflow - demand
        end
    end
    
    # Define pullback function for backward pass
    function loss_pullback(dl)
        # Scale gradient and reshape to match input
        # No negative sign (unlike GA) - direct gradient
        grad_z = device(reshape(grad, size(z))) / ϕ.rescaling_factor * dl
        
        return (NoTangent(), grad_z, NoTangent(), NoTangent())
    end
    
    # Return scaled objective and pullback
    return device(obj / ϕ.rescaling_factor), loss_pullback
end

"""
    sizeInputSpace(ϕ::LagrangianFunctionMCND)

Returns the dimensions of the dual variable space.

# Arguments
- `ϕ::LagrangianFunctionMCND`: The Lagrangian function

# Returns
- `(K, N)`: Tuple of dimensions
  - K: Number of commodities
  - N: Number of nodes in network

# Explanation
The Lagrangian has one dual variable per (commodity, node) pair:
- Each commodity k has N dual variables (one per node)
- Total: K × N dual variables

These correspond to the flow conservation constraints at each node
for each commodity.

# Relationship to Instance
```
sizeInputSpace(ϕ) == sizeLM(ϕ.inst) == (K, N)
```

# Example
```julia
# Problem: 10 commodities, 50 nodes, 200 arcs
inst = load_MCND_instance("mcnd_10_50_200.dat")
lagrangian = constructFunction(inst)

# Dual space dimensions
dims = sizeInputSpace(lagrangian)
@assert dims == (10, 50)  # (K, N)

# Total dual variables
total = prod(dims)
@assert total == 500  # 10 × 50

# Initialize dual variables
z = zeros(Float32, dims...)  # Shape (10, 50)
```

# Memory Requirements
```julia
dims = sizeInputSpace(lagrangian)
K, N = dims

# Float32: 4 bytes per number
bytes = K * N * 4
megabytes = bytes / (1024^2)

println("Dual variables: $K × $N = $(K*N)")
println("Memory: $(megabytes) MB")
```

# Comparison with Problem Size
```julia
K, N = sizeInputSpace(lagrangian)  # Dual dimensions
E = numberSP(lagrangian)            # Number of arcs

println("Network structure:")
println("  Commodities: $K")
println("  Nodes: $N")
println("  Arcs: $E")
println("  Dual vars: $(K*N)")
println("  Flow vars: $(K*E)")
```

# See Also
- `sizeLM`: Underlying function from Instances package
- `numberSP`: Number of arcs (subproblems)
- Network dimensions and complexity
"""
function sizeInputSpace(ϕ::LagrangianFunctionMCND)
    return sizeLM(ϕ.inst)
end

"""
    numberSP(ϕ::LagrangianFunctionMCND)

Returns the number of arcs in the network.

# Arguments
- `ϕ::LagrangianFunctionMCND`: The Lagrangian function

# Returns
- `Int`: Number of arcs (directed edges) in the network

# Explanation
While the Lagrangian decomposes into K subproblems (one per commodity),
this function returns the number of **arcs** (E) in the network, which
represents the number of primal decision variables per commodity.

# Relationship to Instance
```
numberSP(ϕ) == sizeE(ϕ.inst) == E
```

# Why "Number of Subproblems"?
The name is somewhat historical/generic. For MCND:
- **Actual subproblems**: K (one shortest path per commodity)
- **This function returns**: E (number of arcs)

The E arcs represent the structure of each subproblem.

# Example
```julia
# Load instance: 10 commodities, 50 nodes, 200 arcs
inst = load_MCND_instance("network.dat")
lagrangian = constructFunction(inst)

# Network size
K, N = sizeInputSpace(lagrangian)  # (10, 50)
E = numberSP(lagrangian)            # 200

println("Network has $E arcs")
println("Each of $K commodities uses these $E arcs")
println("Total flow variables: $(K * E)")
```

# Flow Variables
Total number of flow decision variables:
```
Total flow vars = K × E
```

Each commodity has one flow variable per arc.

# Network Density
```julia
K, N = sizeInputSpace(lagrangian)
E = numberSP(lagrangian)

# Maximum possible arcs in directed graph
max_arcs = N * (N - 1)

# Network density
density = E / max_arcs * 100

println("Network density: $(round(density, digits=2))%")
```

# Comparison Table
```julia
println("Problem Dimensions:")
println("  Commodities (K): $K")
println("  Nodes (N): $N")
println("  Arcs (E): $E")
println()
println("Variable Counts:")
println("  Dual vars: $(K * N)")
println("  Flow vars: $(K * E)")
println("  Total: $(K * N + K * E)")
```

# Computational Implications
- **K commodities**: Number of shortest path problems to solve
- **E arcs**: Size of each subproblem (network structure)
- **Complexity**: O(K × E log N) for all subproblems

# See Also
- `sizeE`: Underlying function from Instances package
- `sizeInputSpace`: Dual variable dimensions (K, N)
- Network structure and flow formulations
"""
function numberSP(ϕ::LagrangianFunctionMCND)
    return sizeE(ϕ.inst)
end