"""
Abstract Class for Concave functions.
It is used to provide correct inputs for the majority of the functions handled in this package.
Each structure that herits from this abstract type should provide a possibility to compute the value and a sub-gradient of the function.
In particular we provide several examples in other files.
Some of those are Lagrangian Functions (that given a point compute the value of the associated Lagrangian Sub-Problem and herits from the class `AbstractLagrangianFunction`) as the ones contained in the files `LagrangianMCND.jl`.
Another example here presented it is not actually really convex, the one presented in `InnerLoss.jl`, see this file for more details about that.
Morally we can say that this package `could` works on non-concave function (even if inexpected behaviours could occur), in each case we should be interested in maximizing the function.
We just recall that maximizing a concave function is equivalent to minimizing a convex function and so if you are interested to minimizing a function you just have to define the concave function as the oppposite of the convex function and just use again the opposite of the function to obtain the real value outside the package computations . 
"""
abstract type AbstractConcaveFunction end


"""
Given an `AbstractConcaveFunction` `ϕ` and an input vector `z` for the former, this function computes the value and a sub-gradient for `ϕ(z)`.
"""
function value_gradient(ϕ::AbstractConcaveFunction,z::AbstractArray)
    z=cpu(z)
    obj,grad=Flux.withgradient((x)->ϕ(x),z)
    return device(obj), device(grad[1])
end

"""
Function that computes the backward pass for the `value_gradient` function.
In general we IGNORE second-order derivatives.
A generalization that considers second order derivatives could be possible (even if never tested!) if the functions allow to compute that information.
In this case you will have to override this backward pass for `value_gradient`, otherwhise this function is been created exactly with the aim to make easier the construction of further personalized concave functions.
"""
function ChainRulesCore.rrule(f::typeof(value_gradient),ϕ::AbstractConcaveFunction,z::AbstractArray)
    value, grad = Flux.withgradient((x)->ϕ(x),z)
    grad=device(Float32.(reshape(grad[1],size(z))))
	function loss_pullback(dl)
        dl1,dl2=Flux.MLUtils.chunk(dl,2)
        return (NoTangent(), NoTangent(),  grad * device(Float32.(dl1))')
    end
    return (value,grad),loss_pullback
end

"""
Sub-Type of `AbstractConcaveFunction` to modelize the Lagrangian Sub-Problem Functions  (that given a point compute the value of the associated Lagrangian Sub-Problem and herits from the class `AbstractLagrangianFunction`) as the ones contained in the files `LagrangianMCND.jl`.
"""
abstract type AbstractLagrangianFunction <: AbstractConcaveFunction end
