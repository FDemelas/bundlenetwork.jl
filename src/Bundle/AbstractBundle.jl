"""
Abstract type to model all the bundles of this package.
All concrete bundle types must subtype this. A bundle, in the context of 
bundle methods for nonsmooth optimization, stores the cutting-plane model 
(subgradient information) used to build the next iterate.
"""
abstract type AbstractBundle end

"""
Abstract type for bundles that solve a Dual Master Problem to compute 
the new search direction.
In bundle methods, the Dual Master Problem determines the next trial point 
by combining subgradient cuts. All dual-based bundle variants should subtype this.
"""
abstract type DualBundle <: AbstractBundle end

"""
Abstract type for Bundle Factories.
Factories are responsible solely for the construction and initialization 
of bundle objects. This separation of concerns keeps bundle logic independent 
from instantiation details, following the Factory design pattern.
"""
abstract type AbstractBundleFactory end

"""
Abstract type for Soft Bundles, where both the t-strategy and the 
Dual Master Problem solver are replaced by a neural network.
In a soft bundle, the neural network acts as an end-to-end policy:
it receives the current bundle state and directly outputs the next 
iterate, bypassing the classical optimization-based inner loop entirely.
"""
abstract type AbstractSoftBundle <: AbstractBundle end

"""
Factory for creating `SoftBundle` instances.
A `SoftBundle` replaces the classical Dual Master Problem and t-strategy 
with a neural network that jointly handles both roles.
"""
struct SoftBundleFactory <: AbstractBundleFactory end

"""
Factory for creating `tLearningBundle` instances.
A `tLearningBundle` keeps the classical Dual Master Problem structure 
but replaces only the t-strategy with a neural network (`nn_t_strategy`),
learning to adaptively tune the regularization/step-size parameter `t`.
"""
struct tLearningBundleFactory <: AbstractBundleFactory end

"""
Factory for creating `BatchedSoftBundle` instances.
A `BatchedSoftBundle` is a batched variant of the `SoftBundle`,
designed to handle multiple bundle instances in parallel (e.g., for 
training the neural network policy across several problem instances simultaneously).
"""
struct BatchedSoftBundleFactory <: AbstractBundleFactory end

"""
Factory for creating `VanillaBundle` instances.
A `VanillaBundle` is the classical bundle method implementation,
using a standard Dual Master Problem and a conventional (non-learned) t-strategy.
Use this as the baseline against which learned variants are compared.
"""
struct VanillaBundleFactory <: AbstractBundleFactory end

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
Abstract type for Machine Learning-based Bundle methods that specialize `DualBundle`.

Subtypes of this type still solve the Dual Master Problem to obtain the search
direction, but incorporate a neural network somewhere in the algorithm
(e.g., to predict the step size `t` instead of using a hand-crafted t-strategy).
"""
abstract type AbstractMLBundle <: DualBundle end


"""
DeepBundle structure.

A bundle method variant that solves the classical Dual Master Problem for the
search direction, but replaces the t-strategy with a neural network. At each
iteration, the network receives the current bundle state as features and
directly outputs the regularization/step-size parameter `t`.

This structure also supports an exact gradient computation for the backward pass
via a custom `ChainRulesCore.rrule`, exploiting the KKT conditions of the DMP.

# Fields
- `Q::Matrix{Float32}`: Gram matrix of subgradients, `Q = GᵀG` (size: max_β_size × max_β_size).
- `G::Matrix{Float32}`: Subgradient matrix, one column per bundle component (size: input_dim × max_β_size).
- `z::Matrix{Float32}`: Visited trial points matrix, one column per component (size: input_dim × max_β_size).
- `α::Vector{Float32}`: Linearization error vector, one entry per bundle component.
- `s::Int64`: Index of the current stabilization point column in `G`, `z`, and `obj`.
- `model::Model`: JuMP model representing the Dual Quadratic Master Problem.
- `w::Vector{Any}`: Current search direction (convex combination of bundle subgradients).
- `θ::Vector{Float32}`: Current DMP solution (simplex weights over bundle components).
- `objB::Float32`: Scaled DMP objective value (used in SS/NS test and t-strategies).
- `obj::Vector{Float32}`: Objective function values at stored bundle points.
- `cumulative_θ::Vector{Vector{Float32}}`: History of DMP solutions (used in `remove_outdated`).
- `params::BundleParameters`: Hyperparameters controlling the bundle behavior (e.g., `t`, `maxIt`, `m1`).
- `nn`: Neural network model used to predict the step size `t`.
- `lt::AbstractTModelFactory`: Factory used to create features for the neural network.
- `back_prop_idx::Any`: Indices of iterations selected for backpropagation (Serious Steps or all).
- `ws::Any`: History of search directions across iterations (used in the backward pass).
- `θ2s::Any`: History of the KKT-based correction vectors `θ2` (used for exact gradient computation).
- `features::Any`: History of input feature vectors fed to the neural network at each iteration.
- `ts::Any`: History of predicted step sizes `t` across iterations.
- `CSS::Int64`: Counter for consecutive Serious Steps.
- `CNS::Int64`: Counter for consecutive Null Steps.
- `ϕ0::AbstractVector`: Instance-level features used as global context for the neural network.
- `size::Int`: Current number of active bundle components.
- `all_objs::Vector{Float32}`: Full history of objective values at all visited points.
- `exactGrad::Bool`: If `true`, use the KKT-based correction in the backward pass for a better ∂w/∂t.
- `li::Int`: Index of the last inserted bundle component.
- `memorized::Dict`: Dictionary for storing per-iteration diagnostic data (e.g., iteration times).
- `vStar::Float32`: Current DMP objective value (used in SS/NS decision).
- `ϵ::Float32`: DMP objective at the reference step size `t_star` (used in stopping criterion).
- `linear_part::Float32`: Linear part `αᵀθ` of the DMP objective.
- `quadratic_part::Float32`: Quadratic part `‖w‖²` of the DMP objective.
"""
mutable struct DeepBundle <: AbstractMLBundle
    Q::Matrix{Float32}
    G::Matrix{Float32}
    z::Matrix{Float32}
    α::Vector{Float32}
    s::Int64
    model::Model
    w::Vector{Any}
    θ::Vector{Float32}
    objB::Float32
    obj::Vector{Float32}
    cumulative_θ::Vector{Vector{Float32}}
    params::BundleParameters
    nn::Any
    lt::AbstractTModelFactory
    back_prop_idx::Any
    ws::Any
    θ2s::Any
    features::Any
    ts::Any
    CSS::Int64
    CNS::Int64
    ϕ0::AbstractVector
    size::Int
    all_objs::Vector{Float32}
    exactGrad::Bool
    li::Int
    memorized::Dict
    vStar::Float32
    ϵ::Float32
    linear_part::Float32
    quadratic_part::Float32
end

"""
Bundle structure for the classical (Vanilla) proximal bundle method.

Extends `DualBundle` for the case where the t-strategy is a hand-crafted heuristic
(not a neural network). This is the standard aggregated proximal bundle method:
at each iteration it solves the Dual Master Problem to obtain the search direction
and uses a configurable heuristic t-strategy to update the regularization parameter.

# Fields
- `G::Matrix{Float32}`: Subgradient matrix, one column per bundle component (size: input_dim × max_β_size).
- `Q::Matrix{Float32}`: Gram matrix of subgradients, `Q = GᵀG` (size: max_β_size × max_β_size).
- `z::Matrix{Float32}`: Visited trial points matrix, one column per component (size: input_dim × max_β_size).
- `α::Vector{Float32}`: Linearization error vector, one entry per bundle component.
- `s::Int64`: Index of the current stabilization point column in `G`, `z`, and `obj`.
- `model::Model`: JuMP model representing the Dual Quadratic Master Problem.
- `w::Vector{Any}`: Current search direction (weighted combination of bundle subgradients).
- `θ::Vector{Float32}`: Current DMP solution (simplex weights over bundle components).
- `objB::Float32`: Scaled DMP objective value (used in the Serious Step / Null Step test).
- `obj::Vector{Float32}`: Objective function values at stored bundle points.
- `cumulative_θ::Vector{Vector{Float32}}`: History of DMP solutions across iterations (used by `remove_outdated`).
- `params::BundleParameters`: Hyperparameters controlling the bundle (e.g., `t`, `maxIt`, `m1`, `max_β_size`).
- `CSS::Int64`: Counter for consecutive Serious Steps.
- `CNS::Int64`: Counter for consecutive Null Steps.
- `size::Int64`: Current number of active bundle components.
- `all_objs::Vector{Float32}`: Full history of objective values at all visited trial points.
- `li::Int64`: Index of the last inserted bundle component.
- `ts::Vector{Float32}`: History of regularization parameter values `t` across iterations.
- `memorized::Dict`: Dictionary for per-iteration diagnostic data (e.g., iteration times).
- `vStar::Float32`: Current DMP objective value (used in SS/NS decision and t-strategies).
- `ϵ::Float32`: DMP objective at the reference step size `t_star` (used in the stopping criterion).
- `linear_part::Float32`: Linear part `αᵀθ` of the DMP objective.
- `quadratic_part::Float32`: Quadratic part `‖w‖²` of the DMP objective.
- `sign::Bool`: If `true`, the DMP includes non-negativity constraints on the dual variables
  (used for sign-constrained Lagrangian relaxation problems).
"""
mutable struct Bundle <: DualBundle
    G::Matrix{Float32}
    Q::Matrix{Float32}
    z::Matrix{Float32}
    α::Vector{Float32}
    s::Int64
    model::Model
    w::Vector{Any}
    θ::Vector{Float32}
    objB::Float32
    obj::Vector{Float32}
    cumulative_θ::Vector{Vector{Float32}}
    params::BundleParameters
    CSS::Int64
    CNS::Int64
    size::Int64
    all_objs::Vector{Float32}
    li::Int64
    ts::Vector{Float32}
    memorized::Dict
    vStar::Float32
    ϵ::Float32
    linear_part::Float32
    quadratic_part::Float32
    sign::Bool
end


