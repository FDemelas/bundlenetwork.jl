"""
    LagrangianFunctionUC <: AbstractLagrangianFunction

Lagrangian subproblem function for the Unit Commitment (UC) problem
using Lagrangian relaxation.

# Problem Context
Unit Commitment involves scheduling power generation units to meet electricity demand:
- **Generators**: I thermal power units with different characteristics
- **Time periods**: T discrete time periods (typically hours)
- **Demand**: Electricity demand D[t] that must be met at each time t
- **Costs**: Startup, shutdown, and generation costs for each unit
- **Constraints**: 
  - Minimum up/down times
  - Ramping limits
  - Generation capacity bounds
  - Demand satisfaction

# Lagrangian Relaxation Approach
The demand satisfaction constraints are relaxed using Lagrange multipliers:

1. **Original Problem** (hard):
   ```
   min  Σᵢ Σₜ (startup_cost + generation_cost + shutdown_cost)
   s.t. Σᵢ generation[i,t] ≥ demand[t]  ∀t  (demand constraints)
        Unit commitment constraints (up/down times, ramping, etc.)
   ```

2. **Lagrangian Relaxation**:
   - Dualize demand constraints with multipliers z[t]
   - Decompose into I independent unit subproblems
   - Each unit: Optimize its own schedule with modified costs

# Mathematical Formulation
For dual variables z ∈ ℝᵀ (where T = number of time periods):

```
L(z) = Σᵢ min { cost[i] - z' × generation[i] : UC constraints for unit i }
       + z' × demand
```

Where:
- cost[i]: Total cost for unit i (startup + generation + shutdown)
- generation[i]: Power generation profile for unit i (vector of length T)
- z: Lagrange multipliers (electricity prices at each time period)
- demand: Required demand at each time period

# Decomposition Structure
The Lagrangian decomposes into **I independent unit commitment subproblems**:

```
For each generator i = 1, ..., I:
    Optimize unit i schedule independently
    with modified generation cost: original_cost - z[t] × generation[t]
```

Each subproblem is a **dynamic programming problem** considering:
- Minimum up/down time constraints
- Ramping constraints
- Generation bounds
- Startup/shutdown costs

This decomposition enables:
- **Parallel solving**: Each unit solved independently
- **Efficiency**: Dynamic programming for each unit
- **Scalability**: Linear complexity in number of units

# Fields
- `inst::Instances.TUC`: Unit commitment problem instance
  - Contains: I (generators), T (time periods), costs, capacities, demand
- `rescaling_factor::Real`: Divisor for scaling objective values

# Subgradient Interpretation
The subgradient at z represents **demand surplus/deficit**:

```
∂L(z)/∂z[t] = demand[t] - Σᵢ generation*[i,t]
```

Where:
- demand[t]: Required generation at time t
- Σᵢ generation*[i,t]: Total scheduled generation at time t
- Positive: Demand exceeds generation (undersupply)
- Negative: Generation exceeds demand (oversupply)
- Zero: Supply meets demand exactly

At optimal dual variables:
- Generation approximately matches demand
- Bounds primal optimal value

# Dual Problem
The dual problem provides a lower bound for the primal minimization:

```
maximize  L(z)
   z

L(z*) ≤ optimal_primal_value
```

Solving via subgradient methods tightens this bound.


# Economic Interpretation
The dual variables z[t] can be interpreted as:
- **Electricity prices** at each time period
- **Shadow prices** of demand constraints
- **Marginal value** of additional demand

Positive z[t] indicates high-value electricity at time t.

# Subproblem Structure
Each unit's subproblem determines:
- When to start up
- When to shut down
- How much to generate when online
- Balancing costs vs. revenue (from prices z)

# Comparison with Other Lagrangians
| Aspect | UC | MCND | GA |
|--------|----|----|-----|
| Application | Power systems | Networks | Network design |
| Subproblems | I units | K paths | J knapsacks |
| Variables | Binary + continuous | Continuous | Binary |
| Algorithm | Dynamic programming | Shortest path | DP knapsack |
| Constraints | Temporal | Spatial | Capacity |

# See Also
- `Instances` package: https://github.com/FDemelas/Instances
- `constructFunction`: Constructor
- `ChainRulesCore.rrule`: Gradient computation
- Unit commitment problem theory
- Dynamic programming for UC
"""
mutable struct LagrangianFunctionUC <: AbstractLagrangianFunction
    inst::Instances.TUC
    rescaling_factor::Real
end

"""
    (l::LagrangianFunctionUC)(z::AbstractArray)

Forward pass: evaluates the UC Lagrangian subproblem at dual variables z.

# Arguments
- `z::AbstractArray`: Lagrange multipliers (electricity prices)
  - Shape: (T,) where T = number of time periods
  - z[t]: Price/multiplier for demand constraint at time t

# Returns
- Lagrangian value: L(z) / rescaling_factor

# Process
1. **Convert to CPU**: Move z to CPU (DP solver requires CPU)
2. **Flatten**: Reshape to vector if needed
3. **Solve subproblems**: Call solve_SP (solves I unit subproblems)
4. **Extract objective**: First element of solve_SP return
5. **Scale**: Divide by rescaling_factor

# Mathematical Operation
```
L(z) = Σᵢ min { cost[i,t] - z[t] × generation[i,t] : UC constraints }
       + Σₜ z[t] × demand[t]
```

For each generator i:
- Modified generation revenue: z[t] × generation[i,t]
- Net cost: startup + generation + shutdown - revenue
- Find optimal schedule minimizing net cost

# Implementation via solve_SP Function
The `solve_SP` function from Instances package:
- Solves all I unit commitment subproblems via dynamic programming
- Returns [objective_value, generation_schedules, ...]
- Each unit solved independently (can be parallelized)

# Computational Complexity
- **Time**: O(I × T²) with dynamic programming
  - I units, each with O(T²) DP complexity
  - Depends on state space size and transitions
- **Space**: O(I × T) for generation schedules

# Modified Unit Costs
For each unit i at time t with price z[t]:
```
modified_cost[i,t] = generation_cost[i,t] - z[t] × generation[i,t]
```

- **High z[t]**: Incentivizes generation (high revenue)
- **Low z[t]**: Discourages generation (low revenue)
- Units respond by adjusting their schedules

# Dynamic Programming Structure
Each unit's subproblem solved via DP considering:
- **States**: (time, on/off status, time in current state)
- **Transitions**: Remain on, remain off, startup, shutdown
- **Costs**: Adjusted by revenue from prices z

# CPU Requirement
⚠️ Dynamic programming typically CPU-only:
- Sequential state transitions
- Complex branching logic
- CPU implementations highly optimized
- Transfer overhead negligible

# See Also
- `Instances.solve_SP`: Core UC subproblem solver
- `ChainRulesCore.rrule`: Gradient computation
- Dynamic programming for unit commitment
- Power system economics
"""
function (l::LagrangianFunctionUC)(z::AbstractArray)
    return Instances.solve_SP(l.inst, cpu(reshape(z, :)))[1] / l.rescaling_factor
end

# Declare as Flux layer for proper integration
Flux.@layer LagrangianFunctionUC

"""
    constructFunction(inst::Instances.TUC, rescaling_factor::Real=1.0)

Constructs a Lagrangian function from a unit commitment problem instance.

# Arguments
- `inst::Instances.TUC`: UC problem instance from Instances package
  - Must contain: I (units), T (periods), D (demand), costs, constraints
- `rescaling_factor::Real`: Scaling factor for objective (default: 1.0)

# Returns
- `LagrangianFunctionUC`: Callable Lagrangian function

# Instance Requirements
The `Instances.TUC` instance must contain:
- `I::Int`: Number of generating units
- `T::Int`: Number of time periods (horizon length)
- `D::Vector{Float32}`: Demand profile (T,)
- Unit characteristics for each generator:
  - Capacity limits (min/max generation)
  - Startup/shutdown costs
  - Generation cost curves
  - Minimum up/down times
  - Ramping rates

# Rescaling Factor Guidelines
Used to normalize Lagrangian values for numerical stability:

**Small systems** (I, T < 10): 1.0 - 100.0
**Medium systems** (I, T < 100): 100.0 - 1000.0
**Large systems** (I, T > 100): 1000.0 - 10000.0

Choose to keep L(z) values in range [0.1, 10000] for stable gradients.

# See Also
- `Instances.TUC`: Instance structure documentation
- `LagrangianFunctionUC`: Resulting structure
- `sizeInputSpace`: Number of time periods (T)
- `numberSP`: Number of units (I)
- Unit commitment problem formulation
"""
function constructFunction(inst::Instances.TUC, rescaling_factor::Real=1.0)
    return LagrangianFunctionUC(inst, rescaling_factor)    
end

"""
    ChainRulesCore.rrule(ϕ::LagrangianFunctionUC, z::AbstractArray)

Custom backward pass for UC Lagrangian gradient computation.

# Arguments
- `ϕ::LagrangianFunctionUC`: The Lagrangian function
- `z::AbstractArray`: Dual variables (electricity prices), shape (T,)

# Returns
- `obj`: Forward pass result (Lagrangian value)
- `loss_pullback`: Function for backward gradient computation

# Subgradient Formula
The subgradient of the Lagrangian at z is the **demand-supply mismatch**:

```
∂L(z)/∂z[t] = demand[t] - Σᵢ generation*[i,t]
```

Where:
- demand[t]: Required electricity demand at time t (inst.D[t])
- Σᵢ generation*[i,t]: Total generation scheduled across all units at time t
- generation*[i,t]: Optimal generation for unit i at time t (from subproblem solution)

# Physical Interpretation
Each gradient component represents supply-demand balance:

- **> 0**: **Shortage** - Demand exceeds generation
  - Need more generation at time t
  - Should increase price z[t] to incentivize generation
  
- **< 0**: **Surplus** - Generation exceeds demand
  - Too much generation at time t
  - Should decrease price z[t] to discourage generation
  
- **= 0**: **Balance** - Generation exactly meets demand
  - Optimal coordination achieved
  - No price adjustment needed

# Why Custom Implementation?
1. **Analytical gradient**: Exact subgradient from generation schedules
2. **Efficiency**: Schedules already computed in forward pass
3. **Physical meaning**: Gradient has direct economic interpretation
4. **Exactness**: No numerical approximation needed

# Process

## Forward Pass
1. **Move to CPU**: z = cpu(reshape(z, :))
2. **Solve UC subproblems**: obj, p, _ = solve_SP(inst, z)
   - obj: Lagrangian objective value
   - p: Generation schedules (I × T matrix)
     - p[i,t]: Generation of unit i at time t
   - _: Additional info (unused)
3. **Compute gradient**: grad = D - Σᵢ p[i,:]
   - D: Demand vector (T,)
   - sum(p, dims=1)': Total generation at each time (T,)
   - grad[t]: Demand surplus/deficit at time t

## Backward Pass (loss_pullback)
Computes gradient of loss w.r.t. prices:
```
∂loss/∂z = grad / rescaling_factor × dl
```


# Mathematical Justification
By envelope theorem, for the Lagrangian:
```
L(z) = min_x { cost(x) - z' × generation(x) : UC constraints } + z' × demand
```

The subgradient is:
```
∂L/∂z = demand - generation*(z)
```

Where generation*(z) is the optimal generation schedule at prices z.

# Economic Interpretation
The gradient drives price adjustment:
- **Shortage (grad > 0)**: Raise prices to attract generation
- **Surplus (grad < 0)**: Lower prices to reduce generation
- **Balance (grad = 0)**: Prices are market-clearing

This mimics electricity market dynamics where prices adjust to balance supply and demand.

# Subgradient Properties
- **Validity**: Always a valid subgradient at z
- **Direction**: Points toward improved (tighter) bound
- **Sparsity**: Typically not sparse (most times have imbalance)
- **Boundedness**: |grad[t]| ≤ max(D[t], total_capacity)

# Computational Complexity
- **Forward pass**: O(I × T²) - Solve I DP problems
- **Backward pass**: O(I × T) - Sum generation schedules
- **Memory**: O(I × T) - Store generation schedules

# Convergence Behavior
As optimization progresses:
- Subgradient magnitudes typically decrease
- Supply-demand imbalances approach zero
- Lagrangian bound approaches primal optimum
- Generation schedules approach feasibility

# Relationship to Market Clearing
The gradient represents the same imbalance computed in:
- **Day-ahead markets**: Balancing supply bids and demand
- **Real-time markets**: Maintaining frequency through generation control
- **Capacity markets**: Ensuring adequate reserve margins

# See Also
- Envelope theorem for Lagrangian duality
- Unit commitment theory and solution methods
- Electricity market economics
- Subgradient methods for non-smooth optimization
- `Instances.solve_SP`: Core UC solver
"""
function ChainRulesCore.rrule(ϕ::LagrangianFunctionUC, z::AbstractArray)
    # Move prices to CPU and flatten to vector
    z = cpu(reshape(z, :))
    
    # Solve all unit commitment subproblems
    # obj: Lagrangian objective value
    # p: Generation schedules (I × T matrix)
    #    p[i,t] = power generation of unit i at time t
    # _: Additional info (unused)
    obj, p, _ = Instances.solve_SP(ϕ.inst, reshape(z, sizeInputSpace(ϕ)))
    
    # Compute subgradient: demand - total generation
    # sum(p, dims=1)' sums over units (dim 1) to get total generation at each time
    # Shape: (T,) representing demand-supply mismatch at each time period
    grad = ϕ.inst.D - sum(p, dims=1)'
    
    # Define pullback function for backward pass
    function loss_pullback(dl)
        # Scale gradient and reshape to match input
        # Positive grad: shortage (demand > supply) → increase price
        # Negative grad: surplus (demand < supply) → decrease price
        grad_z = device(reshape(grad, size(z))) / ϕ.rescaling_factor * dl
        
        return (NoTangent(), grad_z, NoTangent(), NoTangent())
    end
    
    # Return scaled objective and pullback
    return device(obj / ϕ.rescaling_factor), loss_pullback
end

"""
    sizeInputSpace(ϕ::LagrangianFunctionUC)

Returns the dimension of the dual variable space (number of time periods).

# Arguments
- `ϕ::LagrangianFunctionUC`: The Lagrangian function

# Returns
- `Int`: Number of time periods T in the planning horizon

# Explanation
The Lagrangian has one dual variable (price) per time period:
- Each time period t has one demand constraint
- Each constraint has one Lagrange multiplier z[t]
- Total: T dual variables

These represent electricity prices at each time period.

# Relationship to Instance
```
sizeInputSpace(ϕ) == Instances.nT(ϕ.inst) == T
```

# See Also
- `Instances.nT`: Underlying function from Instances package
- `numberSP`: Number of generators (I)
- UC problem horizon selection
- Temporal resolution trade-offs
"""
function sizeInputSpace(ϕ::LagrangianFunctionUC)
    return Instances.nT(ϕ.inst)
end

"""
    numberSP(ϕ::LagrangianFunctionUC)

Returns the number of subproblems (generating units).

# Arguments
- `ϕ::LagrangianFunctionUC`: The Lagrangian function

# Returns
- `Int`: Number of generating units I in the power system

# Explanation
The Lagrangian decomposes into independent unit subproblems:
- One subproblem per generating unit
- Each unit optimizes its own schedule independently
- Total: I subproblems that can be solved in parallel

# Relationship to Instance
```
numberSP(ϕ) == ϕ.inst.I
```


# See Also
- `sizeInputSpace`: Number of time periods (T)
- Dynamic programming for unit commitment
- Parallel computing for decomposed problems
- Power system structure and composition
"""
function numberSP(ϕ::LagrangianFunctionUC)
    return ϕ.inst.I
end