"""
SoftBundle structure.
It works as SoftBundle structure, but allows to handle multiple bundle execution in parallel in order to perform batching.
This variant does not computes the search direction as solution of the dual master problem, but it needs to use another model (generally based on neural networks).
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
Construct and initialize a SoftBundle structure.
Considering the function `ϕ` and the staring point `z`.
`lt` will be the factory of the model that we want consider for the prediction at the place of the Dual Master Problem.
The Bundle will be initialized to perform `maxIt` iterations (by default `10`).
"""
function initializeBundle(bt::SoftBundleFactory, ϕ::AbstractConcaveFunction, z::AbstractArray, lt, maxIt::Int = 10, reduced_components::Bool = false)
	# Construct Bundle structure
	B = SoftBundle([], [], [], -1, [], [], [], Inf, [Inf], [Float32[]], lt, [], 0, 0, [], 1, Dict(), maxIt, [1.0], 1, [1], reduced_components)
	# Compute objective and sub-gradient in z
	obj, g = value_gradient(ϕ, reshape(z, sizeInputSpace(ϕ)))
	# reshape gradient to a vector
	g = reshape(cpu(g), :)
	# initialize the stabilization point to the unique Bundle component
	B.s = 1
	# Construct a matrix to store the visited points
	B.z = zeros(Float32, length(z), B.maxIt + 1)
	# Initialize the first column of the former matrix to the initialization point
	B.z[:, 1] = reshape(z, :)
	# Construct a matrix to store the linearization errors (as the stabilization point is the only component) all the entries are zero
	B.α = zeros(Float32, 1, B.maxIt + 1)
	# Construct a matrix to store the gradients of the visited points
	B.G = zeros(Float32, length(g), B.maxIt + 1)
	# Initialize the first column of the former matrix to the sub-gradient of the initialization point
	B.G[:, 1] = g
	# Construct a matrix to store the objective functions
	B.obj = zeros(Float32, 1, B.maxIt + 1)
	# Initialize the first column of the former matrix to the objective value of the initialization point
	B.obj[:, 1] = [obj]
	# The Dual Master Problem objective value 
	B.objB = g'g
	# The trial direction is simply g
	B.w = g
	# The DMP solution is straightforward to compute
	B.θ = ones(1, 1)
	# The last inspired component is the only component in the bundle
	B.li = 1
	return B
end

"""
Reinitialize the Bundle before the execution.
This function allows to reuse the same Bundle multiple times without re-creating it.
It is particularly usefull to train models and it should be called before each Bundle execution (even without Backward pass).
"""
function reinitialize_Bundle!(B::SoftBundle)
	# if reinitialize completely the bundle keeping only th e initialization point
	B.li = 1
	B.s = 1
	B.size = 1
	B.lis = [1]
	# reinitialize the gradient matrix, the visited point matrix, the linearization error matrix and the objective value matrix
	B.G = Zygote.bufferfrom(device(hcat([B.G[:, 1], zeros(size(B.G, 1), B.maxIt)]...)))
	B.z = Zygote.bufferfrom(device(hcat([B.z[:, 1], zeros(size(B.z, 1), B.maxIt)]...)))
	B.α = Zygote.bufferfrom(cpu(hcat(B.α[:, 1], zeros(size(B.α, 1), B.maxIt))))
	B.obj = Zygote.bufferfrom(cpu(hcat(B.obj[:, 1], zeros(size(B.obj, 1), B.maxIt))))
	# The Dual Master Problem Solution and the new trial direction are straightforwad to compunte (B.w can be actually keeped all zero as unused before new prediction)
	B.θ = ones(1, 1)
	B.w = device(B.G[:, 1])
end

"""
Function to perform the execution of the SoftBundle `B` to optimize (maximize) the function `Φ`.
If will use the model `m` to provide the search direction and the step size at the place of the resolution of the Dual Master Problem and the t-strategy.
It is throughed to be differentiable using the Automatic Differentiation and so it can be used demanding for an automatic computation of the backward gradient.
"""
function bundle_execution(
	B::SoftBundle,
	ϕ::AbstractConcaveFunction,
	m::AbstractModel;
	soft_updates = true,
	λ = 0.0,
	γ = 0.0,
	δ = 0.0,
	distribution_function = softmax,
	verbose::Int = 0,
	max_inst = Inf,
	metalearning = false,
	unstable = false,
	inference = false,
	z_bar = Zygote.bufferfrom(Float32.(vcat([B.z[:, B.s]]...))),
	z_new = Zygote.bufferfrom(B.z[:, B.li]),
	act=identity)
	# Some global data initialized into inner blocks (as ignore_derivatives() ) should be defined in a more global scope
	let xt, xγ, z_copy, LR_vec, Baseline, obj_new, obj_bar, g, t0, t1, times, maxIt, t, γs, θ, w, comps
		# Initialize a dictionary to store times and the maximum iteration number
		ignore_derivatives() do
			times = Dict("init" => 0.0, "iters" => [], "model" => [], "distribution" => [], "features" => [], "stab_point" => [], "update_bundle" => [], "update_direction" => [], "update_point" => [], "lsp" => [])
			maxIt = B.maxIt
			t0 = time()
		end
		# initialize the global features (for standard model AttentionModel) it is unused and initialized as zero 
		featG = 0.0 #function_features(B, B.lt)
		# initilize some values
		ignore_derivatives() do
			# sub-gradient in the current trial-point
			g = Zygote.bufferfrom(device(zeros(size(B.w))))
			# objective value in the new trial point
			obj_new = Zygote.bufferfrom(cpu(B.obj[B.li.*ones(Int64, 1)]))
			# objective value in the stabilization point
			obj_bar = Zygote.bufferfrom(cpu(B.obj[B.s.*ones(Int64, 1)]))
			# vector containing the trial directions at each iterations
			w = Zygote.bufferfrom([device(zeros(length(B.w))) for it in 1:maxIt])
			# output of the model to compute the DMP solution, before applying a distribution function (and so they are real values not still in the simples)
			# it has the same size of the current bundle components (i.e. number of iterations)
			γs = Zygote.bufferfrom([device(zeros(1, it)) for it in 1:maxIt])
			# approximation of the DMP solution, after applying a distribution function (values between 0-1 in the simplex)
			# it has the same size of the current bundle components (i.e. number of iterations)
			θ = Zygote.bufferfrom([device(zeros(1, it)) for it in 1:maxIt])

			xt = Zygote.bufferfrom([device(zeros(it,1,size_features(B.lt))) for it in 1:maxIt])
			xγ = Zygote.bufferfrom([device(zeros(it,1,size_comp_features(B.lt))) for it in 1:maxIt])
			
			# value of the Dual Master Problem
			B.objB = 0
			# initialization time
			times["init"] = time() - t0
			comps = Zygote.bufferfrom([ones(Int64, it) for it in 1:maxIt+1])
		end
		for it in 1:maxIt
			ignore_derivatives() do
				t0 = time()
			end

			# no Back-Propagation here
			# create features and resize them properly
			ignore_derivatives() do
				xt[it], xγ[it] = create_features(B, m; auxiliary = featG)
				#if size(xt)[1] == length(xt)
				#	xt[it] = reshape(xt, (length(xt), 1))
				#end
				#if size(xγ)[1] == length(xγ)
				#	xγ[it] = reshape(xγ, (length(xγ), 1))
				#end
			end

			#features extraction time
			ignore_derivatives() do
				append!(times["features"], time() - t0)
			end

			ignore_derivatives() do
				t1 = time()
			end
			# output of the model: step-size t and one value for each bundle component, i.e. γs[it] that has it components
			t, γs[it] = m(xt[it], xγ[it], B.li, comps[it])

			# store B.t for features extraction (no derivative is needed as we did not differentate through features extraction)
			ignore_derivatives() do
				B.t = reshape(t, :)
			end

			# and index that allows to consider all the bundle components (by default as max_inst = + Inf) or just a fixed amount (max_inst < +Inf)
			min_idx = Int64(max(1, length(comps[it]) - max_inst))

			# model computing time
			ignore_derivatives() do
				append!(times["model"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end

			# Compute a simplex vector using the output γs[it] of the model
			θ[it] = distribution_function(γs[it][:, :]; dims = 2)

			# store it for featurers extraction and store also the time used for computing this simplex vector
			ignore_derivatives() do
				B.θ = θ[it]
				append!(times["distribution"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end

			# Compute the new trial direction as convex combination of the gradients in the Bundle
			w[it] = B.G[:, comps[it]] * θ[it][1, :]

			# Store this direction for features extraction and store the time to compute that convex combination
			ignore_derivatives() do
				B.w = w[it]
				append!(times["update_direction"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end

			# Compute the new trial point considering a step of size `t` from the stabilization point `z_bar` considering the new search direction `w[it]`
			z_new[:] = act(z_bar[:] .+ sum(t) .* w[it])

			# Store the time for compute the new trial point
			ignore_derivatives() do
				append!(times["update_point"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end

			# Compute the value and the sub-gradient associated to the new trial point
			v, g_tmp = value_gradient(ϕ, z_new[:])
			g[:] = device(g_tmp)
			obj_new[1] = v

			# Store the time for value, gradient computation
			# note: in the case of Lagrangian relaxation this corresponds to the Lagrangian-sub-problem resolution
			ignore_derivatives() do
				append!(times["lsp"], time() - t1)
			end


			B.size += 1
			B.li = B.size
			if B.reduced_components
				already_in = false
				j = []
				for i in comps[it]
					if sum(B.G[:, i] - g[:]) < 1.0e-6
						already_in = true
						ignore_derivatives() do
							push!(j, i)
						end
					end
				end
				if already_in
					comps[it+1] = vcat([k for k in comps[it] if !(k in j)], B.size)
				else
					comps[it+1] = vcat(comps[it], B.size)
				end
			else
				comps[it+1] = vcat(comps[it], B.size)
			end


			ignore_derivatives() do
				append!(B.lis, B.li)
			end


			ignore_derivatives() do
				t1 = time()
			#	@show B.θ
			end

			# update the stabilization point
			# different possibilities are possible
			if unstable
				# if we consider the unstable version, then the stabilization vector will be the last visited vector
				z_bar = z_new
				obj_bar = obj_new
				B.s = B.li .* ones(Int64, length(B.s))
			else
				# otherwhise
				if !soft_updates
					# if no soft updates are consider, then we update the stabilization point if the new trial point improves the objective value
					z_bar[:] = obj_new[:] > obj_bar[:] ? (z_new[:]) : (z_bar[:])
					obj_bar[:] = (obj_new[:] > obj_bar[:] ? obj_new[:] : obj_bar[:])
				else
					# if soft updates are consider, then we update the stabilization point 
					# using the softmax of the objective value (of new trial point and stabilization point)
					sm = softmax([obj_new[1], obj_bar[1]])
					obj_bar[1] = sm' * [obj_new[1], obj_bar[1]]
					z_bar[:] = device(sm' * cpu.([z_new[:], z_bar[:]]))
				end
				# update the stabilization point index (used only for features extraction)
				B.s = sum(obj_new[:] > obj_bar[:] ? B.li : B.s)
			end
			# store the time for stabilization point updates
			ignore_derivatives() do
				append!(times["stab_point"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end
			# update the gradient matrix
			B.G[:, B.li] = device(g[:])
			# update the trial points matrix
			B.z[:, B.li] = z_new[:]
			# update the objective value matrix
			B.obj[:, B.li] = obj_new[:]
			# update the linearization errors matrix
			B.α[1, :] = (B.obj[1, :] .- obj_bar[1, :]) .- cpu(sum(B.G[:, :] .* (B.z[:, :] .- z_bar[:]); dims = 1))'

			# store the time for bundle update and the one for the whole iteration
			ignore_derivatives() do
				append!(times["update_bundle"], time() - t1)
				append!(times["iters"], time() - t0)
			end
		end


		if inference
			# at inference time just return the objective value in the final stabilization point and the times dictionary
			return ϕ(reshape(z_bar, sizeInputSpace(ϕ))), times
		else
			# at training time return the loss value
			# first a telescopic sum of all the visited points
			vγ = (γ > 0 ? γ * mean([γ^(maxIt + 1 - i) for i in (maxIt+1):-1:1] .* [ϕ(reshape(z, sizeInputSpace(ϕ))) for z in eachcol(B.z[:, 1:(maxIt+1)])]) : 0)
			# then also a convex combination of final trial point and final stabilization point
			vλ = (1 - λ) * ϕ(reshape(z_bar[:], sizeInputSpace(ϕ))) + (λ) * ϕ(reshape(z_new[:], sizeInputSpace(ϕ)))
			# the loss will be a sum of the two
			return vγ + vλ
		end
	end
end
