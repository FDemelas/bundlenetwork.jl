"""
Bundle structure that extends a `DualBundle` in the case in which the t-strategy is not based on a neural-network model.
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

"""
construct and initialize a `Bundle`.
This version is also refered to 'classic' or 'vanilla' Bundle and corresponds to an aggregated proximal bundle method.
It solve the Dual Master Problem to obtain a search direction and can use different (heuristic) t-strategies.
"""
function initializeBundle(bt::VanillaBundleFactory,ϕ::AbstractConcaveFunction, t::Real, z::AbstractArray; bp::BundleParameters = BundleParameters(), max_bundle_size = -1)
	B = Bundle(Float32[;;], Float32[;;], Float32[;;], [], -1, Model(Gurobi.Optimizer), [], [], Inf, [Inf], [Float32[]], bp, 0, 0, 1, Float32[], 1, [t],Dict("times"=>[]),0.0,0.0,0.0,0.0,true)
	obj, g = value_gradient(ϕ, z)
	B.s = 1
	g = reshape(g, :)
	B.params.max_β_size = max_bundle_size
	if 0 < B.params.max_β_size < Inf
		B.z = zeros(Float32, (length(z), B.params.max_β_size))
		B.z[:, 1] = reshape(z, :)
		B.α = zeros(Float32, B.params.max_β_size)
		B.G = zeros(Float32, (length(g), B.params.max_β_size))
		B.G[:, 1] = g
		B.obj = zeros(Float32, B.params.max_β_size)
		B.obj[1] = obj
		B.Q = zeros(Float32, (B.params.max_β_size, B.params.max_β_size))
		B.Q[1, 1] = g'g
	else
		B.α = [0]
		B.z = reshape(z, (length(z), 1))
		B.G = reshape(g, (length(g), 1))
		B.Q = Float32[g'g;;]
		B.obj = [obj]
	end
	B.params.t = t
	append!(B.ts, t)
	B.model = create_DQP(B, t)
	B.sign = sign(ϕ) == 1 ? false : true
	solve_DQP(B)

	compute_direction(B)

	B.w = B.G[:, 1] .* B.θ

	B.cumulative_θ = [Float32[1.0]]
	push!(B.all_objs, obj)
	return B
end