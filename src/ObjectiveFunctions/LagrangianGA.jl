"""
Lagrangian Sub-Problem function for the Knapsack-Relaxation of the Multi-Commodity Network Design Problem.
See the package `https://github.com/FDemelas/Instances` for more details on that relaxation and implementation of the functions.
"""
mutable struct LagrangianFunctionGA <: AbstractLagrangianFunction
    inst::cpuInstanceGA
    rescaling_factor::Real
end


"""
Forward Pass for the Lagrangian Sub-Problem function for the Knapsack-Relaxation of the Multi-Commodity Network Design Problem.
See the package `https://github.com/FDemelas/Instances` for more informations.
"""
function (l::LagrangianFunctionGA)(z::AbstractArray)
    return -LR(l.inst,-cpu(z))[1]/l.rescaling_factor
end

Flux.@layer LagrangianFunctionGA

"""
Given a GA instance from the package `https://github.com/FDemelas/Instances` it construct a Lagrangian Function
"""
function constructFunction(inst::cpuInstanceGA,rescaling_factor::Real=1.0)
    return LagrangianFunctionGA(inst,rescaling_factor)    
end

"""
Backward Pass for the Lagrangian Sub-Problem function for the Knapsack-Relaxation of the Multi-Commodity Network Design Problem.
See the package `https://github.com/FDemelas/Instances` for more informations.
"""
function ChainRulesCore.rrule(ϕ::LagrangianFunctionGA, z::AbstractArray)
    sz=size(z)
    z=reshape(cpu(z),:)
    grad = ones(Float32, sizeInputSpace(ϕ))
    
    #z=Vector{Float64}(cpu(z)[1,:])
	x = zeros(Float32, ϕ.inst.I, ϕ.inst.J)

	obj = sum(z)
	obj1 = 0

	for j in 1:ϕ.inst.J
		xp = zeros(Float32, ϕ.inst.I)
		obj1 = Instances.solve_knapsack(ϕ.inst.I, ϕ.inst.p[:, j] - z, ϕ.inst.w[:, j], ϕ.inst.c[j], xp)
		x[:,j] = xp
        obj+=obj1   
	end
    
    grad -= sum(x,dims=2)'
    
    #grad=reshape(grad,:)
    loss_pullback(dl) = (NoTangent(), device(reshape(-grad,sz))/ϕ.rescaling_factor * dl, NoTangent(), NoTangent())
	return -device(obj/ϕ.rescaling_factor), loss_pullback
end


"""
Size of the inputs of the function `ϕ` of type `LagrangianFunctionGA`.
"""
function sizeInputSpace(ϕ::LagrangianFunctionGA)
    return sizeLM(ϕ.inst)
end

function numberSP(ϕ::LagrangianFunctionGA)
    return ϕ.inst.I
end

function sign(ϕ::LagrangianFunctionGA)
    return -1
end
