"""
SoftBundle structure.
It works as SoftBundle structure, but allows to handle multiple bundle execution in parallel in order to perform batching.
This variant does not computes the search direction as solution of the dual master problem, but it needs to use another model (generally based on neural networks).
"""
mutable struct SoftBundle <: AbstractSoftBundle
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
	t::Any
	idxComp::AbstractArray
	size::Int64
	lis::Vector{Int64}
	reduced_components::Bool
end

"""
Construct and initialize a SoftBundle structure.
Considering the functions contained in the vector `ϕs` and the staring points contained in the vector (of vectors) `z`.
`ϕs` and `z` must have the same number of components.
`lt` will be the factory of the model that we want consider for the prediction at the place of the Dual Master Problem.
The Bundle will be initialized to perform `maxIt` iterations (by default `10`).
"""
function initializeBundle(bt::SoftBundleFactory, ϕs::Vector{<:AbstractConcaveFunction}, z::Vector{<:AbstractArray}, lt, maxIt::Int = 10, reduced_components::Bool = false)
	# Construct Bundle structure
	B = SoftBundle([], [], [], [-1], [], [], [], Inf, [Inf], [Float32[]], lt, [], 0, 0, [], 1, Dict(), maxIt, zeros(length(ϕs)), [], 1, [1], reduced_components)
	# the batch size is equal to the number of input functions
	batch_size = length(ϕs)
	
	# Construct a matrix to store the visited points
	B.z = zeros(Float32, maximum([prod(sizeInputSpace(ϕ)) for ϕ in ϕs]), batch_size,  1)
	# Construct a matrix to store the linearization errors (as the stabilization point is the only component) all the entries are zero
	B.α = zeros(Float32, 1, batch_size, 1)
	# Construct a matrix to store the objective functions
	B.obj =  zeros(Float32, 1, batch_size, 1)
	# Construct a matrix to store the gradients of the visited points
	B.G = zeros(Float32, maximum([prod(sizeInputSpace(ϕ)) for ϕ in ϕs]), batch_size, 1)
	B.idxComp =	 [ 1:prod(sizeInputSpace(ϕ)) for (idx, ϕ) in enumerate(ϕs)]
	# Compute objective and sub-gradient in z for all the objectives
	# reshape gradient to a vector and store all the first and last components
	# to be able to access at each of those
	for (idx, ϕ) in enumerate(ϕs)
		lLM = prod(sizeInputSpace(ϕ))
		obj, g = value_gradient(ϕ, reshape(z[idx], sizeInputSpace(ϕ)))
		B.G[1:lLM,idx,1] = reshape(cpu(g), :)
		B.obj[1,idx,1] = obj
		B.α[1,idx,1] = 0.0
		B.z[1:lLM,idx,1] = reshape(z[idx], :)
	end

	
	# The trial direction is simply the first gradient
	B.w = reshape(B.G,:)
	# Initialize the first column of the former matrix to the sub-gradient of the initialization point
	# Construct a matrix to store the objective functions
	B.objB = B.w'B.w
	# The DMP solution is straightforward to compute
	B.θ = ones(batch_size, 1)
	
	# The last inspired component is the only component in the bundle
	B.li = 1
	# initialize the stabilization points to the unique Bundle component
	# for each objective function
	B.s = ones(length(B.idxComp))
	return B
end


"""
Construct and initialize a SoftBundle structure.
Considering the function `ϕ` and the staring point `z`.
`lt` will be the factory of the model that we want consider for the prediction at the place of the Dual Master Problem.
The Bundle will be initialized to perform `maxIt` iterations (by default `10`).
"""
function initializeBundle(bt::SoftBundleFactory, ϕ::AbstractConcaveFunction, z::AbstractArray, lt, maxIt::Int = 10, reduced_components::Bool = false)
	return initializeBundle(bt, [ϕ], [z], lt, maxIt, reduced_components)
end



"""
Reinitialize the Bundle before the execution.
This function allows to reuse the same Bundle multiple times without re-creating it.
It is particularly usefull to train models and it should be called before each Bundle execution (even without Backward pass).
"""
function reinitialize_Bundle!(B::SoftBundle)
	# if reinitialize completely the bundle keeping only th e initialization point
	B.li = 1
	B.size = 1
	B.lis = [1]
	B.s = ones(length(B.idxComp))
	# reinitialize the gradient matrix, the visited point matrix, the linearization error matrix and the objective value matrix
	B.G = device(B.G[:, :, 1:1])
	B.z = device(B.z[:, :, 1:1])
	B.α = device(B.α[:,:,1:1])
	B.obj = device(B.obj[:, :, 1:1])
	# The Dual Master Problem Solution and the new trial direction are straightforwad to compunte (B.w can be actually keeped all zero as unused before new prediction)
	B.θ = ones(length(B.idxComp), 1)
	B.w = device(B.G[:,:, 1])
	B.t = device(zeros(1,length(B.idxComp)))
end

"""
Function to perform the execution of the SoftBundle `B` to optimize (maximize) the functions contained in the vector `Φ`.
If will use the model `m` to provide the search direction and the step size at the place of the resolution of the Dual Master Problem and the t-strategy.
It is throughed to be differentiable using the Automatic Differentiation and so it can be used demanding for an automatic computation of the backward gradient.
"""
function bundle_execution(
	B::SoftBundle,
	ϕ::Vector{<:AbstractConcaveFunction},
	m::AbstractModel;
	soft_updates = false,
	λ = 0.0,
	γ = 0.0,
	δ = 0.0,
	distribution_function = softmax,
	verbose::Int = 0,
	max_inst = Inf,
	metalearning = false,
	unstable = false,
	inference = false,
	z_new = device(B.z[:,:,B.li]),
	z_bar = device(B.z[:,:,B.li]),
	act=identity	)
	# Some global data initialized into inner blocks (as ignore_derivatives() ) should be defined in a more global scope
	let xt, xγ, z_copy, LR_vec, Baseline, obj_new, obj_bar, g, t0, t1, times, maxIt, t, γs, θ, comps
		# Initialize a dictionary to store times and the maximum iteration number
		ignore_derivatives() do
			times = Dict("init" => 0.0, "iters" => [], "model" => [], "distribution" => [], "features" => [], "stab_point" => [], "update_bundle" => [], "update_direction" => [], "update_point" => [], "lsp" => [])
			maxIt = B.maxIt
			t0 = time()
		end
		# initialize the global features (for standard model AttentionModel) it is unused and initialized as zero 
		featG = 0.0 # function_features(B, B.lt)
		# initilize some values
		ignore_derivatives() do
			# sub-gradient in the current trial-point
			g = device(zeros(size(B.w)))
			# objective value in the new trial point
			obj_new = cpu(B.obj[1,:,B.li])
			# objective value in the stabilization point
			obj_bar = obj_new
			# output of the model to compute the DMP solution, before applying a distribution function (and so they are real values not still in the simples)
			# it has the same size of the current bundle components (i.e. number of iterations)
			#γs = Zygote.bufferfrom([device(zeros(length(B.idxComp), it)) for it in 1:maxIt])
			xt = Zygote.bufferfrom([device(zeros(length(B.idxComp),it,size_features(B.lt))) for it in 1:maxIt])
			xγ = Zygote.bufferfrom([device(zeros(length(B.idxComp),it,size_comp_features(B.lt))) for it in 1:maxIt])
		

			# approximation of the DMP solution, after applying a distribution function (values between 0-1 in the simplex)
			# it has the same size of the current bundle components (i.e. number of iterations)
			#θ = Zygote.bufferfrom([device(zeros(length(B.idxComp), it)) for it in 1:maxIt])
			# value of the Dual Master Problem
			B.objB = 0
			# initialization time
			times["init"] = time() - t0
			comps=Zygote.bufferfrom([ones(Int64, it) for it in 1:maxIt+1])
		end
		for it in 1:maxIt
			ignore_derivatives() do
				t0 = time()
			end

			# no Back-Propagation here
			# create features and resize them properly
			ignore_derivatives() do
				xt[it], xγ[it] = create_features( B, m; auxiliary = featG)
			end

			#features extraction time
			ignore_derivatives() do
				append!(times["features"], time() - t0)
			end

			ignore_derivatives() do
				t1 = time()
			end

			# output of the model: step-size t and one value for each bundle component, i.e. γs[it] that has it component
			B.t, γs = m(xt[it], xγ[it], B.li, comps[it])
			

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
			θ = distribution_function(γs[:, :]; dims = 2)
			
			# store it for featurers extraction and store also the time used for computing this simplex vector
			ignore_derivatives() do
				B.θ = θ
				append!(times["distribution"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end
			
			
			# Compute the new trial direction as convex combination of the gradients in the Bundle
			B.w = hcat([B.G[:,i, comps[it]] * θ[i, :] for (i, j) in enumerate(B.idxComp)]...)

			# Store the time to compute that convex combination
			ignore_derivatives() do
				append!(times["update_direction"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end

			# Compute the new trial point considering a step of size `t` from the stabilization point `z_bar` considering the new search direction `w[it]`
			z_new = act(z_bar + B.w .* B.t)
			ignore_derivatives() do
				append!(times["update_point"], time() - t1)
			end

			# Store the time for compute the new trial point
			ignore_derivatives() do
				t1 = time()
			end

			# Compute the value and the sub-gradient associated to the new trial point
			#for (i, j) in enumerate(B.idxComp)
			#	v, g_tmp = value_gradient(ϕ[i], z_new[j,i])
			#	g[j,i] = reshape(cpu(g_tmp),:)
			#	obj_new[i] = v
			#end
			obj_new = vcat([value_gradient(ϕ[i],z_new[:,i])[1] for i in 1:length(B.idxComp)]...)
			ignore_derivatives() do
				g = hcat([value_gradient(ϕ[i],z_new[:,i])[2] for i in 1:length(B.idxComp)]...)
			end
			B.size+=1
			B.li=B.size
			if B.reduced_components
				already_in = false
				j=[]
				for i in comps[it]
					if sum(B.G[:, i] - g[:]) < 1.0e-6
						already_in = true
						ignore_derivatives() do
							push!(j,i)
						end
					end
				end
				if already_in
					comps[it+1] = vcat([k for k in comps[it] if !(k in j)],B.size)
				else
					comps[it+1] = vcat(comps[it],B.size)
				end
			else
				comps[it+1] = vcat(comps[it],B.size)
			end

			


			# Store the time for value, gradient computation
			# note: in the case of Lagrangian relaxation this corresponds to the Lagrangian-sub-problem resolution
			ignore_derivatives() do
				append!(times["lsp"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
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
					isgeq = obj_bar .>= obj_new
					obj_bar = ifelse.(isgeq,obj_bar,obj_new)
					z_bar = ifelse.(repeat(device(isgeq'),(size(z_bar,1))),z_bar,z_new)
				else
					# if soft updates are consider, then we update the stabilization point 
					# using the softmax of thecobjective value (of new trial point and stabilization point)
						sj=cat(obj_new,obj_bar;dims=2)
						sm=softmax(sj;dims=2)
						obj_bar = sum(sm .* sj ;dims=2)
						bm = permutedims(cat(z_bar,z_new;dims=3),(2,3,1))
						z_bar = permutedims(dropdims(sum(device(sm) .* device(bm);dims=1),dims=1),(2,1))
				end
				# update the stabilization point index (used only for features extraction)
				B.s = ifelse.(reshape(cpu(obj_bar .>= obj_new),:),B.s,B.li*ones(length(B.s)))
			end
			# store the time for stabilization point updates
			ignore_derivatives() do
				append!(times["stab_point"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end

			# update the gradient matrix
			B.G = cat(B.G,g;dims=3)
			# update the trial points matrix
			B.z = cat(B.z,z_new;dims=3)
			# update the objective value matrix
			B.obj = cat(B.obj,(obj_new)';dims=3)
			# update the linearization errors matrix
			B.α = (B.obj .- device(obj_bar)) .-  (batched_mul(permutedims(B.z .- z_bar,(2,1,3)),B.G))
			

			# store the time for bundle update and the one for the whole iteration
			ignore_derivatives() do
				append!(times["update_bundle"], time() - t1)
				append!(times["iters"], time() - t0)
			end
		end

		if inference
			# at inference time just return the objective value in the final stabilization point and the times dictionary
			return mean(ϕ[iϕ](reshape(z_bar[s:e], sizeInputSpace(ϕ[iϕ]))) for (iϕ, (s, e)) in enumerate(B.idxComp)), times
		else
			# at training time return the loss value
			# first a telescopic sum of all the visited points
			vγ = (γ > 0 ? mean(γ * mean([γ^(maxIt+1 - i) for i in (maxIt+1):-1:1] .* [ϕ[iϕ](reshape(z, sizeInputSpace(ϕ[iϕ]))) for z in eachcol(B.z[j, iϕ, 1:(maxIt+1)])]) for (iϕ, j) in enumerate(B.idxComp)) : 0)
			# then also a convex combination of final trial point and final stabilization point
			vλ = mean((1 - λ) * ϕ[iϕ](reshape(z_bar[j,iϕ], sizeInputSpace(ϕ[iϕ]))) + (λ) * ϕ[iϕ](reshape(z_new[j,iϕ], sizeInputSpace(ϕ[iϕ]))) for (iϕ, j) in enumerate(B.idxComp))
			# the loss will be the sum of the two
			return vγ + vλ
		end
	end
end
