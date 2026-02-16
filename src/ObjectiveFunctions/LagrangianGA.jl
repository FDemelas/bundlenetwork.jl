"""
    LagrangianFunctionGA <: AbstractLagrangianFunction

Lagrangian subproblem function for the Knapsack-Relaxation of the 
Multi-Commodity Network Design (MCND) Problem.

# Problem Context
Multi-Commodity Network Design involves:
- Designing a network to route multiple commodities
- Each commodity has origin-destination demand
- Network arcs have capacity and cost constraints
- Goal: Minimize total cost while satisfying demands

# Lagrangian Relaxation
The Knapsack-Relaxation decomposes the MCND problem by:
1. **Relaxing** coupling constraints (e.g., capacity constraints)
2. **Dualizing** with Lagrange multipliers z
3. **Decomposing** into independent knapsack subproblems per commodity

# Mathematical Formulation
For dual variables z ∈ ℝᴵ (Lagrange multipliers):
```
L(z) = Σⱼ min_xⱼ { (p[:,j] - z)' xⱼ : w[:,j]' xⱼ ≤ c[j] }
```

Where:
- I: Number of items (arcs in network)
- J: Number of commodities
- p[i,j]: Profit/cost of item i for commodity j
- w[i,j]: Weight/capacity of item i for commodity j
- c[j]: Capacity constraint for commodity j
- xⱼ: Binary selection vector for commodity j

Each subproblem is a **0-1 knapsack problem**.

# Fields
- `inst::cpuInstanceGA`: Problem instance (from Instances package)
  Contains: I (items), J (commodities), p (profits), w (weights), c (capacities)
- `rescaling_factor::Real`: Divisor for scaling objective values

# Decomposition Structure
The Lagrangian decomposes into J independent knapsack problems:
```
For each commodity j = 1, ..., J:
    max  (p[:,j] - z)' x[j]
    s.t.  w[:,j]' x[j] ≤ c[j]
          x[j] ∈ {0,1}ᴵ
```

This decomposition enables:
- **Parallel solving**: Each knapsack solved independently
- **Scalability**: Linear complexity in J
- **Efficiency**: Knapsack has pseudo-polynomial algorithms

# Sign Convention
⚠️ **Important**: This Lagrangian uses **minimization** convention (sign = -1),
unlike most other AbstractConcaveFunction instances which use maximization (sign = 1).

The negative sign in the forward pass converts maximization to minimization.

# Dual Problem
The dual problem (solved by outer optimization) is:
```
maximize  L(z)
   z
```

Provides an **upper bound** for the primal minimization problem.

# Example Usage
```julia
# Load problem instance
using Instances
inst = load_instance("MCND_instance.dat")

# Create Lagrangian function
lagrangian = constructFunction(inst, rescaling_factor=100.0)

# Evaluate at dual point
z = zeros(Float32, inst.I)  # Initial dual variables
obj_value = lagrangian(z)    # Lagrangian bound

# Compute gradient (subgradient)
obj, grad = value_gradient(lagrangian, z)

# Dual ascent step
z += 0.01 * grad
```

# Subgradient Structure
The subgradient at z is:
```
∂L(z) = Σⱼ x*[j] - d
```

Where:
- x*[j]: Optimal solution to knapsack j
- d: Target demand (typically 1 for network design)

# See Also
- `Instances` package: https://github.com/FDemelas/Instances
- `constructFunction`: Constructor for LagrangianFunctionGA
- `ChainRulesCore.rrule`: Custom backward pass implementation
- Knapsack problem algorithms
"""
mutable struct LagrangianFunctionGA <: AbstractLagrangianFunction
    inst::cpuInstanceGA
    rescaling_factor::Real
end

"""
    (l::LagrangianFunctionGA)(z::AbstractArray)

Forward pass: evaluates the Lagrangian subproblem at dual variables z.

# Arguments
- `z::AbstractArray`: Lagrange multipliers (dual variables), size I

# Returns
- Lagrangian value: L(z) / rescaling_factor

# Process
1. **Convert to CPU**: Move z to CPU (knapsack solver requires CPU data)
2. **Negate**: Apply -z (sign convention for the relaxation)
3. **Solve**: Call LR function from Instances package
4. **Scale**: Divide by rescaling_factor
5. **Negate again**: Convert to minimization convention

# Mathematical Operation
```
L(z) = Σⱼ max { (p[:,j] - z)' x : w[:,j]' x ≤ c[j], x ∈ {0,1}ᴵ }
```

Implemented via:
```julia
-LR(inst, -z)[1] / rescaling_factor
```

# Double Negation Explained
- First `-z`: Adjusts for Instances package convention
- Second `-`: Converts max to min (this function minimizes)
- Net effect: Proper Lagrangian evaluation for minimization

# Implementation Note
The `LR` function from the Instances package:
- Solves all J knapsack subproblems
- Returns [objective_value, solutions, ...]
- Operates on CPU data only

# Computational Complexity
- **Time**: O(J × I × C) where C is max capacity
  - Each knapsack: O(I × c[j]) via dynamic programming
  - J knapsacks solved sequentially (or parallel if implemented)
- **Space**: O(J × I) for solutions

# Example
```julia
# Create Lagrangian
lagrangian = constructFunction(instance, 1.0)

# Evaluate at zero duals (original problem bound)
z_zero = zeros(Float32, instance.I)
bound_zero = lagrangian(z_zero)

# Evaluate at non-zero duals
z_optimal = optimize_dual_variables(lagrangian)
bound_optimal = lagrangian(z_optimal)

# bound_optimal ≥ bound_zero (tighter bound)
```

# GPU Considerations
⚠️ Data is moved to CPU because:
- Knapsack solver is CPU-only
- Dynamic programming not efficiently parallelizable on GPU
- Transfer overhead is small compared to solving time

# See Also
- `Instances.LR`: Core Lagrangian relaxation solver
- `ChainRulesCore.rrule`: Gradient computation
- Dynamic programming for knapsack
"""
function (l::LagrangianFunctionGA)(z::AbstractArray)
    return -LR(l.inst, -cpu(z))[1] / l.rescaling_factor
end

# Declare as Flux layer for proper integration
Flux.@layer LagrangianFunctionGA

"""
    constructFunction(inst::cpuInstanceGA, rescaling_factor::Real=1.0)

Constructs a Lagrangian function from a problem instance.

# Arguments
- `inst::cpuInstanceGA`: Problem instance from Instances package
  - Contains: I (items), J (commodities), p, w, c arrays
- `rescaling_factor::Real`: Scaling factor for objective (default: 1.0)

# Returns
- `LagrangianFunctionGA`: Callable Lagrangian function

# Instance Requirements
The `cpuInstanceGA` instance must contain:
- `I::Int`: Number of items (arcs in network)
- `J::Int`: Number of commodities
- `p::Matrix{Float32}`: Profit matrix (I × J)
- `w::Matrix{Float32}`: Weight matrix (I × J)
- `c::Vector{Float32}`: Capacity vector (J,)

# Rescaling Factor
Used to normalize objective values:
- **Too small**: May cause numerical overflow
- **Too large**: May cause underflow or slow convergence
- **Recommended**: Scale to keep L(z) in range [0.1, 1000]

Typical values:
- Small instances (I, J < 100): 1.0 - 10.0
- Medium instances (I, J < 1000): 10.0 - 100.0
- Large instances (I, J > 1000): 100.0 - 1000.0

# Example
```julia
using Instances

# Load problem instance
inst = load_GA_instance("data/mcnd_50_20.dat")
# 50 items, 20 commodities

# Create Lagrangian with scaling
lagrangian = constructFunction(inst, rescaling_factor=50.0)

# Verify dimensions
@assert sizeInputSpace(lagrangian) == inst.I
@assert numberSP(lagrangian) == inst.J
```

# Loading Instances
```julia
# From Instances package
inst = cpuInstanceGA(
    I = 50,           # Number of items
    J = 20,           # Number of commodities
    p = rand(50, 20), # Profit matrix
    w = rand(50, 20), # Weight matrix
    c = rand(20)      # Capacity constraints
)

lagrangian = constructFunction(inst)
```

# See Also
- `Instances` package: Problem instance management
- `LagrangianFunctionGA`: Resulting structure
- `sizeInputSpace`: Verify dimension
"""
function constructFunction(inst::cpuInstanceGA, rescaling_factor::Real=1.0)
    return LagrangianFunctionGA(inst, rescaling_factor)    
end

"""
    ChainRulesCore.rrule(ϕ::LagrangianFunctionGA, z::AbstractArray)

Custom backward pass for Lagrangian function gradient computation.

# Arguments
- `ϕ::LagrangianFunctionGA`: The Lagrangian function
- `z::AbstractArray`: Dual variables (input point)

# Returns
- `obj`: Forward pass result (Lagrangian value)
- `loss_pullback`: Function for backward gradient computation

# Gradient Computation
The subgradient of the Lagrangian at z is:
```
∂L(z)/∂z = Σⱼ x*[j] - 1
```

Where x*[j] is the optimal solution to knapsack j at dual point z.

# Why Custom Implementation?
1. **Analytical gradient**: We can compute exact subgradient from knapsack solutions
2. **Efficiency**: Avoid numerical differentiation
3. **Exactness**: Subgradient is exact, not approximated
4. **Integration**: Solutions already computed in forward pass

# Process

## Forward Pass
1. **Initialize**: Create gradient vector (ones), solution matrix (zeros)
2. **Add dual term**: obj = Σᵢ zᵢ
3. **Solve knapsacks**: For each commodity j:
   - Solve: max (p[:,j] - z)' x s.t. w[:,j]' x ≤ c[j]
   - Store solution in x[:,j]
   - Add objective contribution

## Backward Pass (loss_pullback)
Computes gradient of loss w.r.t. dual variables:
```
∂loss/∂z = -(Σⱼ x*[:,j]) / rescaling_factor × dl
```

The negative sign accounts for:
- Minimization convention (sign = -1)
- Subgradient direction

# Mathematical Justification
By envelope theorem, the derivative of the Lagrangian is:
```
dL/dz = -Σⱼ x*[j]
```

Since each x*[j] is the argmax of knapsack j, and:
```
L(z) = Σⱼ max { (p[:,j] - z)' x : constraints }
```

The subgradient is the sum of active constraints.

# Implementation Details

## Gradient Initialization
```julia
grad = ones(Float32, sizeInputSpace(ϕ))
```
Starts with 1 because obj starts with Σᵢ zᵢ.

## Solution Accumulation
```julia
for j in 1:J
    # Solve knapsack j
    xp = solve_knapsack(I, p[:,j] - z, w[:,j], c[j])
    x[:,j] = xp
    
    # Accumulate objective
    obj += knapsack_value
end
```

## Gradient Finalization
```julia
grad -= sum(x, dims=2)'  # Subtract solution sums
```

Each item i contributes to gradient based on how many times it's selected.

# NoTangent Explanation
- First `NoTangent()`: No gradient w.r.t. the function itself
- Second `NoTangent()`: No gradient w.r.t. dual variables in pullback context
- Third/Fourth `NoTangent()`: No gradient w.r.t. auxiliary arguments

Only the actual input `z` receives meaningful gradients.

# Example Usage
```julia
# This is called automatically during backpropagation
z = randn(Float32, inst.I)

# Forward and backward pass
obj, pullback = ChainRulesCore.rrule(lagrangian, z)

# Compute gradient (called automatically by autodiff)
dl = 1.0  # Gradient from loss
grad_z = pullback(dl)[2]  # Extract gradient w.r.t. z

# Dual ascent update
z_new = z + learning_rate * grad_z
```

# Subgradient Properties
The computed subgradient satisfies:
- **Validity**: ∂L(z) is a valid subgradient at z
- **Direction**: Points in ascent direction for maximization
- **Sparsity**: Often sparse (many zeros if few items selected)
- **Boundedness**: Bounded by ±J (max selections per item)

# Computational Complexity
- **Forward**: O(J × I × C) - Solve J knapsacks
- **Backward**: O(I × J) - Sum solutions
- **Memory**: O(I × J) - Store solution matrix

# See Also
- Subgradient methods for non-smooth optimization
- Lagrangian relaxation theory
- Dynamic programming for knapsack
- `Instances.solve_knapsack`: Core solver
"""
function ChainRulesCore.rrule(ϕ::LagrangianFunctionGA, z::AbstractArray)
    # Store original shape and flatten
    sz = size(z)
    z = reshape(cpu(z), :)
    
    # Initialize gradient vector (starts with ones for dual term)
    grad = ones(Float32, sizeInputSpace(ϕ))
    
    # Initialize solution matrix (I items × J commodities)
    x = zeros(Float32, ϕ.inst.I, ϕ.inst.J)
    
    # Initialize objective with dual term: Σᵢ zᵢ
    obj = sum(z)
    
    # Variable for knapsack objective
    obj1 = 0
    
    # Solve each knapsack subproblem
    for j in 1:ϕ.inst.J
        # Temporary solution vector for commodity j
        xp = zeros(Float32, ϕ.inst.I)
        
        # Solve knapsack j with modified profits: p[:,j] - z
        # Constraints: w[:,j]' x ≤ c[j], x ∈ {0,1}ᴵ
        obj1 = Instances.solve_knapsack(
            ϕ.inst.I,           # Number of items
            ϕ.inst.p[:, j] - z, # Modified profits (Lagrangian term)
            ϕ.inst.w[:, j],     # Weights
            ϕ.inst.c[j],        # Capacity
            xp                  # Output: optimal selection
        )
        
        # Store solution
        x[:, j] = xp
        
        # Accumulate objective value
        obj += obj1   
    end
    
    # Compute subgradient: 1 - Σⱼ x[:,j]
    # Each item contributes based on selection frequency
    grad -= sum(x, dims=2)'
    
    # Define pullback function for backward pass
    function loss_pullback(dl)
        # Compute gradient contribution
        # Negative sign for minimization convention
        # Reshape to match original input shape
        grad_z = device(reshape(-grad, sz)) / ϕ.rescaling_factor * dl
        
        return (NoTangent(), grad_z, NoTangent(), NoTangent())
    end
    
    # Return scaled objective and pullback
    return -device(obj / ϕ.rescaling_factor), loss_pullback
end

"""
    sizeInputSpace(ϕ::LagrangianFunctionGA)

Returns the dimension of the dual variable space.

# Arguments
- `ϕ::LagrangianFunctionGA`: The Lagrangian function

# Returns
- `Int`: Dimension of dual variables (equals number of items I)

# Explanation
The Lagrangian has one dual variable per relaxed constraint.
For the knapsack relaxation of MCND:
- Each item (arc) has one dual variable
- Total: I dual variables

# Relationship to Instance
```
sizeInputSpace(ϕ) == ϕ.inst.I == sizeLM(ϕ.inst)
```

Where:
- `I`: Number of items in the problem
- `sizeLM`: Size of Lagrange multiplier vector

# Example
```julia
# Problem with 50 items, 20 commodities
inst = load_instance("mcnd_50_20.dat")
lagrangian = constructFunction(inst)

# Dual space dimension
dim = sizeInputSpace(lagrangian)
@assert dim == 50  # One dual per item

# Initialize dual variables
z = zeros(Float32, dim)
```

# Usage
```julia
# Allocate dual variable vector
dim = sizeInputSpace(lagrangian)
z = randn(Float32, dim) * 0.1

# Verify compatibility
@assert length(z) == sizeInputSpace(lagrangian)
```

# See Also
- `sizeLM`: Underlying function from Instances package
- `numberSP`: Returns number of subproblems (J)
"""
function sizeInputSpace(ϕ::LagrangianFunctionGA)
    return sizeLM(ϕ.inst)
end

"""
    numberSP(ϕ::LagrangianFunctionGA)

Returns the number of subproblems in the Lagrangian decomposition.

# Arguments
- `ϕ::LagrangianFunctionGA`: The Lagrangian function

# Returns
- `Int`: Number of subproblems (equals number of commodities J)

# Explanation
The Lagrangian decomposes into independent knapsack subproblems:
- One subproblem per commodity
- Total: J subproblems
- Each solvable independently (parallelizable)

# Relationship to Instance
```
numberSP(ϕ) == ϕ.inst.J
```

# Example
```julia
# Problem with 50 items, 20 commodities
inst = load_instance("mcnd_50_20.dat")
lagrangian = constructFunction(inst)

# Number of subproblems
num_sp = numberSP(lagrangian)
@assert num_sp == 20  # One per commodity

# Can solve in parallel
results = pmap(1:num_sp) do j
    solve_knapsack_j(lagrangian, j, z)
end
```

# Parallelization Potential
```julia
num_sp = numberSP(lagrangian)

# Sequential (current implementation)
for j in 1:num_sp
    solve_subproblem(j)
end

# Parallel (potential optimization)
Threads.@threads for j in 1:num_sp
    solve_subproblem(j)
end
```

# See Also
- `sizeInputSpace`: Dimension of dual space (I)
- Knapsack problem decomposition
- Parallel solving strategies
"""
function numberSP(ϕ::LagrangianFunctionGA)
    return ϕ.inst.J
end

"""
    sign(ϕ::LagrangianFunctionGA)

Returns the optimization direction for this Lagrangian function.

# Returns
- `-1`: Indicates this is a **minimization** problem

# Why -1?
Unlike most `AbstractConcaveFunction` instances (which return +1 for maximization),
this Lagrangian represents a **minimization** problem:

- **Original problem**: Minimize network design cost
- **Dual problem**: Maximize Lagrangian bound
- **Convention**: sign = -1 indicates minimization

# Relationship to Duality
```
Primal (min): minimize c'x subject to Ax ≥ b
Dual (max):   maximize L(z) = min { c'x + z'(b - Ax) }
              
This function computes the dual bound (which maximizes),
but represents the primal (which minimizes).
```

# Usage in Optimization
```julia
direction = sign(lagrangian)

if direction == 1
    # Maximize: gradient ascent
    z_new = z + lr * gradient
elseif direction == -1
    # Minimize: gradient descent
    z_new = z - lr * gradient
end
```

# Duality Gap
The Lagrangian provides bounds:
```
L(z) ≥ optimal_primal_value  (for any z)
```

Optimizing z maximizes the bound, tightening it.

# Example
```julia
lagrangian = constructFunction(inst)

# Check optimization direction
@assert sign(lagrangian) == -1  # Minimization

# Dual ascent (maximize bound)
for iter in 1:max_iterations
    obj, grad = value_gradient(lagrangian, z)
    z += learning_rate * grad  # Ascent for dual maximization
end
```

# Comparison with Other Functions
| Function Type | sign() | Objective |
|---------------|--------|-----------|
| InnerLoss | +1 | Maximize accuracy |
| LagrangianFunctionGA | **-1** | Minimize cost |
| Most Lagrangians | +1 | Maximize bound |

# See Also
- Lagrangian duality theory
- Primal-dual relationships
- Subgradient methods for dual optimization
"""
function sign(ϕ::LagrangianFunctionGA)
    return -1
end