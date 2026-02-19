"""
    AbstractConcaveFunction

Abstract base type for concave functions in the optimization framework.

# Purpose
This abstract type provides a unified interface for functions that need to:
1. Compute function values
2. Compute subgradients (or gradients for differentiable functions)

It is used throughout the package to ensure consistent function handling
and to enable automatic differentiation with proper gradient computation.

# Mathematical Context
A function f: ℝⁿ → ℝ is **concave** if for all x, y ∈ ℝⁿ and λ ∈ [0,1]:
```
f(λx + (1-λ)y) ≥ λf(x) + (1-λ)f(y)
```

Equivalently, -f is convex. This package focuses on **maximization** of
concave functions, which is equivalent to **minimization** of convex functions.

# Key Operations
Concrete subtypes must support:
- Function evaluation: `value = ϕ(z)`
- Value and gradient computation: `value, grad = value_gradient(ϕ, z)`

# Subtype Hierarchy
```
AbstractConcaveFunction
├── AbstractLagrangianFunction
│   └── (Lagrangian subproblem functions)
└── (Other concave functions)
    └── InnerLoss (quasi-concave approximation)
```

# Concrete Examples
Several concrete implementations are provided in separate files:
- **Lagrangian functions** (`LagrangianMCND.jl`): Compute values and subgradients
  for Lagrangian relaxation subproblems
- **InnerLoss** (`InnerLoss.jl`): A special case that may not be strictly concave
  but is treated similarly in the optimization framework

# Non-Concave Functions
⚠️ **Important Note**: While this package is designed for concave functions,
it may work on non-concave functions, though unexpected behaviors can occur.
The package assumes the objective is to **maximize** the function.

# Convex to Concave Conversion
If you have a convex function g(x) that you want to minimize:
1. Define the concave function: ϕ(x) = -g(x)
2. Use the package to maximize ϕ(x)
3. Convert results back: min g(x) = -max ϕ(x)


# Automatic Differentiation
The package uses Flux.jl's automatic differentiation by default.
Custom gradient computation can be provided by overriding `value_gradient`.


# See Also
- `value_gradient`: Compute value and gradient together
- `AbstractLagrangianFunction`: Specialized for Lagrangian subproblems
- `ChainRulesCore.rrule`: Custom backward pass for gradients
"""
abstract type AbstractConcaveFunction end

"""
    value_gradient(ϕ::AbstractConcaveFunction, z::AbstractArray)

Computes both the function value and subgradient for a concave function.

# Arguments
- `ϕ::AbstractConcaveFunction`: The concave function to evaluate
- `z::AbstractArray`: The point at which to evaluate the function

# Returns
- `value`: The function value ϕ(z)
- `gradient`: A subgradient ∂ϕ(z) (or gradient if ϕ is differentiable)

# Mathematical Background
For a concave function ϕ, a vector g is a **subgradient** at point z if:
```
ϕ(y) ≤ ϕ(z) + g'(y - z)  for all y
```

For differentiable concave functions, the gradient is the unique subgradient.

# Implementation Details
1. **CPU Transfer**: Input is moved to CPU for computation
2. **Automatic Differentiation**: Uses Flux.withgradient for gradient computation
3. **Device Transfer**: Results are moved back to appropriate device (CPU/GPU)

# Why CPU Transfer?
Some operations may be more stable or only supported on CPU. The function
ensures compatibility by:
- Moving input to CPU before computation
- Computing on CPU
- Moving results back to the original device


# Numerical Considerations
- Gradients are computed using automatic differentiation
- For non-differentiable points, returns one valid subgradient
- Ensures type consistency (Float32) for GPU compatibility

# See Also
- `ChainRulesCore.rrule`: Custom backward pass implementation
- `AbstractConcaveFunction`: Base type requiring this interface
"""
function value_gradient(ϕ::AbstractConcaveFunction, z::AbstractArray)
    # Move to CPU for computation stability
    z = cpu(z)
    
    # Compute value and gradient using automatic differentiation
    obj, grad = Flux.withgradient((x) -> ϕ(x), z)
    
    # Move results back to appropriate device (CPU/GPU)
    return device(obj), device(grad[1])
end

"""
    ChainRulesCore.rrule(f::typeof(value_gradient), ϕ::AbstractConcaveFunction, z::AbstractArray)

Defines the backward pass (reverse-mode differentiation) for `value_gradient`.

# Purpose
This custom reverse rule enables proper gradient propagation through the
`value_gradient` function during backpropagation in neural network training.

# Arguments
- `f::typeof(value_gradient)`: The forward function (value_gradient)
- `ϕ::AbstractConcaveFunction`: The concave function being evaluated
- `z::AbstractArray`: The input point

# Returns
- `(value, grad)`: Forward pass results (primal)
- `loss_pullback`: Function to compute gradients in backward pass (tangent)

# Mathematical Background
The reverse rule computes how gradients flow backward through the computation.
For the composition:
```
loss = f(value_gradient(ϕ, z))
```

We need:
```
∂loss/∂z = (∂loss/∂value) × (∂value/∂z) + (∂loss/∂grad) × (∂grad/∂z)
```

# Second-Order Derivatives
**Important**: This implementation **IGNORES second-order derivatives**.

Specifically, we ignore:
- ∂²ϕ/∂z²: Second derivatives of ϕ
- ∂grad/∂z: How the gradient changes with z

This is a common simplification in many optimization algorithms where:
1. Second-order information is expensive to compute
2. First-order methods are sufficient
3. Computational efficiency is prioritized

# When Second-Order Matters
If your application requires second-order derivatives:
1. Override this `rrule` for your specific function type
2. Implement computation of the full Jacobian ∂grad/∂z
3. Test thoroughly (this is rarely needed and never tested in this package)

# Design Rationale
This implementation provides:
1. **Simplicity**: Avoids complex second-order computations
2. **Efficiency**: Faster backward pass
3. **Flexibility**: Easy to override for specific needs
4. **Compatibility**: Works with Flux.jl's autodiff system


# See Also
- `value_gradient`: The forward function
- `ChainRulesCore`: Package for custom differentiation rules
- Flux.jl documentation on custom gradients
"""
function ChainRulesCore.rrule(
    f::typeof(value_gradient),
    ϕ::AbstractConcaveFunction,
    z::AbstractArray
)
    # Forward pass: compute value and gradient
    value, grad = Flux.withgradient((x) -> ϕ(x), z)
    
    # Ensure gradient is on correct device and has correct shape
    grad = device(Float32.(reshape(grad[1], size(z))))
    
    # Define pullback function for backward pass
    function loss_pullback(dl)
        # Split incoming gradient into two components
        # dl1: gradient w.r.t. value
        # dl2: gradient w.r.t. gradient (ignored in first-order approximation)
        dl1, dl2 = Flux.MLUtils.chunk(dl, 2)
        
        # Compute gradient w.r.t. input z
        # Note: Only uses first-order term (grad * dl1)
        # Second-order term (involving ∂grad/∂z) is ignored
        return (NoTangent(), NoTangent(), grad * device(Float32.(dl1))')
    end
    
    return (value, grad), loss_pullback
end

"""
    AbstractLagrangianFunction <: AbstractConcaveFunction

Abstract type for Lagrangian subproblem functions.

# Purpose
Specialized subtype of `AbstractConcaveFunction` for functions that arise
in Lagrangian relaxation methods. These functions compute the value of
Lagrangian subproblems given dual variables.

# Mathematical Background
In Lagrangian relaxation, we decompose a constrained optimization problem:
```
minimize  c'x
subject to  Ax ≥ b
            x ∈ X
```

Into a Lagrangian function:
```
L(x, λ) = c'x + λ'(b - Ax)
```

The Lagrangian subproblem for fixed λ is:
```
maximize_x  L(x, λ)  subject to x ∈ X
```

This subproblem is often easier to solve than the original problem.

# Usage Pattern
Concrete implementations represent specific Lagrangian subproblems:
1. Take dual variables (λ) as input
2. Solve the subproblem: max_x L(x, λ)
3. Return both the optimal value and a subgradient w.r.t. λ

# Concrete Examples
See `LagrangianMCND.jl` for implementations such as:
- Multicommodity Network Design (MCND) Lagrangian subproblems
- Network flow problems with Lagrangian relaxation
- Other decomposable optimization problems

# Advantages of Lagrangian Relaxation
1. **Decomposition**: Large problems become many small problems
2. **Parallelization**: Subproblems can be solved independently
3. **Bounds**: Provides upper bounds for minimization problems
4. **Structure**: Exploits problem structure naturally

# See Also
- `AbstractConcaveFunction`: Parent type
- `LagrangianMCND.jl`: Concrete implementations
- Subgradient methods for dual optimization
"""
abstract type AbstractLagrangianFunction <: AbstractConcaveFunction end

"""
    sign(ϕ::AbstractConcaveFunction)

Returns the sign convention for the function (+1 for maximization).

# Returns
- `1`: Indicates this is a maximization problem

# Purpose
This function provides a consistent way to query whether a function
should be maximized (+1) or minimized (-1). For concave functions,
we always maximize, hence the return value is always 1.

# Usage
This is useful when building generic optimization algorithms that
need to know the direction of optimization:


# Design Note
While currently all `AbstractConcaveFunction` instances return 1,
this method could be overridden in derived types if needed for
special cases or for symmetry with a potential `AbstractConvexFunction`
type that would return -1.

# See Also
- Optimization direction in gradient-based methods
- Convention for maximization vs minimization
"""
function sign(ϕ::AbstractConcaveFunction)
    return 1
end