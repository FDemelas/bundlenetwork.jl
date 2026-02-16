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

# Example Usage
```julia
# Load unit commitment instance
using Instances
inst = TUC(
    I = 10,          # 10 generators
    T = 24,          # 24-hour horizon
    D = demand_profile,  # Hourly demand
    # ... other parameters
)

# Create Lagrangian function
lagrangian = constructFunction(inst, rescaling_factor=1000.0)

# Evaluate at zero prices (no coordination)
z = zeros(Float32, 24)
bound = lagrangian(z)

# Compute subgradient
obj, grad = value_gradient(lagrangian, z)

# Subgradient ascent (maximize lower bound)
learning_rate = 0.1
z += learning_rate * grad
```

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

# Example
```julia
# Create Lagrangian for 10 units, 24-hour horizon
inst = TUC(I=10, T=24, D=demand_24h)
lagrangian = constructFunction(inst, 100.0)

# Zero prices (uncoupled units)
z_zero = zeros(Float32, 24)
bound_zero = lagrangian(z_zero)

# Market-clearing prices
z_market = [30.0, 35.0, 40.0, ...]  # /MWh for each hour
bound_market = lagrangian(z_market)

# Optimal prices (from subgradient method)
z_opt = optimize_prices(lagrangian)
bound_opt = lagrangian(z_opt)

println("Uncoupled bound: ", bound_zero)
println("Market bound: ", bound_market)
println("Optimal bound: ", bound_opt)
# bound_opt ≥ bound_market ≥ bound_zero (tighter bounds)
```

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

# Example - Basic Usage
```julia
using Instances

# Create 5-unit, 24-hour UC problem
inst = TUC(
    I = 5,                      # 5 generators
    T = 24,                     # 24-hour horizon
    D = [100, 120, 140, ...],  # Hourly demand (MW)
    # ... unit parameters
)

# Create Lagrangian
lagrangian = constructFunction(inst, rescaling_factor=500.0)

# Verify dimensions
@assert sizeInputSpace(lagrangian) == 24  # T time periods
@assert numberSP(lagrangian) == 5          # I units
```

# Example - Standard UC Instance
```julia
# Load from file
inst = load_UC_instance("data/uc_10unit_24h.dat")
lagrangian = constructFunction(inst, 1000.0)

println("System size:")
println("  Units: $(inst.I)")
println("  Horizon: $(inst.T) hours")
println("  Peak demand: $(maximum(inst.D)) MW")
```

# Example - Complete Optimization Workflow
```julia
# Create instance
inst = TUC(I=10, T=24, D=demand_curve)
lagrangian = constructFunction(inst, 1000.0)

# Initialize prices (dual variables)
T = sizeInputSpace(lagrangian)
z = ones(Float32, T) * 50.0  # Start at 50/MWh

# Subgradient optimization
best_bound = -Inf
for iter in 1:1000
    # Evaluate Lagrangian and get subgradient
    obj, grad = value_gradient(lagrangian, z)
    
    # Track best bound
    best_bound = max(best_bound, obj)
    
    # Subgradient step with decreasing step size
    step_size = 100.0 / sqrt(iter)
    z += step_size * grad
    
    # Project prices to non-negative (if desired)
    z = max.(z, 0.0)
    
    # Log progress
    if iter % 100 == 0
        avg_shortage = mean(grad)
        println("Iter $iter: Bound=$obj, Avg shortage=$avg_shortage")
    end
end

println("Best Lagrangian bound: $best_bound")
```

# Example - Price Analysis
```julia
lagrangian = constructFunction(inst, 1000.0)
z_optimal = optimize_lagrangian(lagrangian)

# Analyze optimal prices
for t in 1:inst.T
    println("Hour $t: Price = \$$(z_optimal[t])/MWh, " *
            "Demand = $(inst.D[t]) MW")
end

# Peak hours typically have higher prices
peak_hours = findall(inst.D .> quantile(inst.D, 0.75))
println("Peak hour prices: ", z_optimal[peak_hours])
```

# Instance Creation Tips
```julia
# Define unit parameters
unit_params = [
    (Pmin=50, Pmax=200, startup=500, shutdown=200, ...),  # Unit 1
    (Pmin=30, Pmax=150, startup=400, shutdown=150, ...),  # Unit 2
    # ... more units
]

# Define demand profile (e.g., typical daily pattern)
demand = [
    100, 95, 90, 85, 90, 100,      # Night (low)
    120, 140, 160, 170, 175, 180,  # Morning ramp-up
    185, 180, 175, 180, 185, 180,  # Daytime (high)
    170, 150, 130, 120, 110, 105   # Evening ramp-down
]

inst = TUC(I=length(unit_params), T=length(demand), D=demand, 
           units=unit_params)
```

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

# Implementation Details

## Generation Aggregation
```julia
# p is (I × T) matrix of generation schedules
# p[i,t] = generation of unit i at time t

# Sum over units (dim 1) to get total generation at each time
total_generation = sum(p, dims=1)'  # Shape: (T,)

# Compute supply-demand mismatch
grad = inst.D - total_generation    # Shape: (T,)
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

# Example Usage
```julia
# Automatic during backpropagation
T = 24
z = ones(Float32, T) * 50.0  # 50/MWh initial prices

# Forward and backward
obj, pullback = ChainRulesCore.rrule(lagrangian, z)

# Compute gradient (called automatically by autodiff)
dl = 1.0
grad_z = pullback(dl)[2]

# Interpret gradient
println("Demand-supply mismatch:")
for t in 1:T
    if grad_z[t] > 1.0
        println("  Hour $t: Shortage $(grad_z[t]) MW")
    elseif grad_z[t] < -1.0
        println("  Hour $t: Surplus $(abs(grad_z[t])) MW")
    end
end

# Price update based on imbalance
step_size = 0.1
z_new = z + step_size * grad_z
```

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

# Example
```julia
# 24-hour unit commitment problem
inst = TUC(I=10, T=24, D=demand_profile)
lagrangian = constructFunction(inst)

# Price dimension
T = sizeInputSpace(lagrangian)
@assert T == 24  # 24 hours

# Initialize hourly prices
z = ones(Float32, T) * 40.0  # 40/MWh for all hours
```

# Typical Horizons
Different UC problem horizons:
- **Real-time**: 1-4 hours (high resolution)
- **Day-ahead**: 24-48 hours (hourly)
- **Week-ahead**: 168 hours (7 days)
- **Multi-week**: 336+ hours (2+ weeks)

```julia
# Day-ahead market
T_day = 24
z_day = zeros(Float32, T_day)

# Week-ahead planning
T_week = 168  # 24 * 7
z_week = zeros(Float32, T_week)
```

# Memory Requirements
```julia
T = sizeInputSpace(lagrangian)

# Float32: 4 bytes per price
bytes = T * 4
println("Dual variables: $T")
println("Memory: $bytes bytes")
```

# Time Resolution
The time period length determines resolution:
```julia
T = sizeInputSpace(lagrangian)

# If hourly resolution
hours = T
println("Planning horizon: $hours hours")

# If 15-minute resolution
minutes = T * 15
println("Planning horizon: $minutes minutes = $(minutes/60) hours")
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

# Example
```julia
# Power system with 10 generators
inst = TUC(I=10, T=24, D=demand)
lagrangian = constructFunction(inst)

# Number of units
I = numberSP(lagrangian)
@assert I == 10

println("System has $I generating units")
println("Each solves independent UC subproblem")
```

# Parallelization Potential
```julia
I = numberSP(lagrangian)

# Sequential (current implementation)
for i in 1:I
    solve_unit_subproblem(i, z)
end

# Parallel (potential optimization)
Threads.@threads for i in 1:I
    solve_unit_subproblem(i, z)
end

# Speedup: Up to I-fold with sufficient cores
```

# System Size Categories
```julia
I = numberSP(lagrangian)
T = sizeInputSpace(lagrangian)

if I < 10
    println("Small system: $I units")
elseif I < 100
    println("Medium system: $I units")
else
    println("Large system: $I units")
end

println("Problem size: $(I*T) decision variables (unit-time pairs)")
```

# Unit Types
Typical power system mix:
```
Nuclear:     2-5 units (base load)
Coal:        5-20 units (base/intermediate)
Gas turbine: 10-50 units (intermediate/peak)
Hydro:       Variable (dispatchable)
```

# Computational Implications
```julia
I = numberSP(lagrangian)
T = sizeInputSpace(lagrangian)

# Complexity per subproblem: O(T²) (dynamic programming)
# Total complexity: O(I × T²)

complexity = I * T^2
println("Computational complexity: O($complexity)")

# Each unit independently optimizes over T time periods
println("Each unit: $T-period optimization problem")
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