"""
Lagrangian Sub-Problem function for the Knapsack-Relaxation of the Multi-Commodity Network Design Problem.
See the package `https://github.com/FDemelas/Instances` for more details on that relaxation and implementation of the functions.
"""
mutable struct LagrangianFunctionUC <: AbstractLagrangianFunction
    inst::Instances.TUC
    rescaling_factor::Real
end


"""
Forward Pass for the Lagrangian Sub-Problem function for the Knapsack-Relaxation of the Multi-Commodity Network Design Problem.
See the package `https://github.com/FDemelas/Instances` for more informations.
"""
function (l::LagrangianFunctionUC)(z::AbstractArray)
    return Instances.solve_SP(l.inst,cpu(reshape(z,:)))[1]/l.rescaling_factor
end

Flux.@layer LagrangianFunctionUC

"""
Given a MCND instance from the package `https://github.com/FDemelas/Instances` it construct a Lagrangian Function
"""
function constructFunction(inst::Instances.TUC,rescaling_factor::Real=1.0)
    return LagrangianFunctionUC(inst,rescaling_factor)    
end

"""
Backward Pass for the Lagrangian Sub-Problem function for the Knapsack-Relaxation of the Multi-Commodity Network Design Problem.
See the package `https://github.com/FDemelas/Instances` for more informations.
"""
function ChainRulesCore.rrule(ϕ::LagrangianFunctionUC, z::AbstractArray)
    z=cpu(reshape(z,:))
    obj, p, _ = Instances.solve_SP(ϕ.inst, reshape(z,sizeInputSpace(ϕ)))
    grad = ϕ.inst.D - sum(p,dims=1)'
	loss_pullback(dl) = (NoTangent(), device(grad)/ϕ.rescaling_factor * dl, NoTangent(), NoTangent())
	return device(obj/ϕ.rescaling_factor), loss_pullback
end


"""
Size of the inputs of the function `ϕ` of type `LagrangianFunctionUC`.
"""
function sizeInputSpace(ϕ::LagrangianFunctionUC)
    return Instances.nT(ϕ.inst)
end

function numberSP(ϕ::LagrangianFunctionUC)
    return ϕ.inst.I
end