"""
Lagrangian Sub-Problem function for the Knapsack-Relaxation of the Multi-Commodity Network Design Problem.
See the package `https://github.com/FDemelas/Instances` for more details on that relaxation and implementation of the functions.
"""
mutable struct LagrangianFunctionMCND <: AbstractLagrangianFunction
    inst::cpuInstanceMCND
    rescaling_factor::Real
end


"""
Forward Pass for the Lagrangian Sub-Problem function for the Knapsack-Relaxation of the Multi-Commodity Network Design Problem.
See the package `https://github.com/FDemelas/Instances` for more informations.
"""
function (l::LagrangianFunctionMCND)(z::AbstractArray)
    return LR(l.inst,cpu(z))[1]/l.rescaling_factor
end

Flux.@layer LagrangianFunctionMCND

"""
Given a MCND instance from the package `https://github.com/FDemelas/Instances` it construct a Lagrangian Function
"""
function constructFunction(inst::cpuInstanceMCND,rescaling_factor::Real=1.0)
    return LagrangianFunctionMCND(inst,rescaling_factor)    
end

"""
Backward Pass for the Lagrangian Sub-Problem function for the Knapsack-Relaxation of the Multi-Commodity Network Design Problem.
See the package `https://github.com/FDemelas/Instances` for more informations.
"""
function ChainRulesCore.rrule(ϕ::LagrangianFunctionMCND, z::AbstractArray)
    z=cpu(z)
    obj, x, _ = LR(ϕ.inst, reshape(z,sizeInputSpace(ϕ)))
    grad = zeros(Float32, sizeInputSpace(ϕ))
    for k in 1:sizeInputSpace(ϕ)[1]
        for i in 1:sizeInputSpace(ϕ)[2]
            grad[k, i] = sum([x[k, ij] for ij in 1:sizeE(ϕ.inst) if tail(ϕ.inst, ij) == i]) - sum([x[k, ij] for ij in 1:sizeE(ϕ.inst) if head(ϕ.inst, ij) == i]) - b(ϕ.inst, i, k)
        end
    end
	loss_pullback(dl) = (NoTangent(), device(reshape(grad,size(z)))/ϕ.rescaling_factor * dl, NoTangent(), NoTangent())
	return device(obj/ϕ.rescaling_factor), loss_pullback
end


"""
Size of the inputs of the function `ϕ` of type `LagrangianFunctionMCND`.
"""
function sizeInputSpace(ϕ::LagrangianFunctionMCND)
    return sizeLM(ϕ.inst)
end


function numberSP(ϕ::LagrangianFunctionMCND)
    return sizeE(ϕ.inst)
end
