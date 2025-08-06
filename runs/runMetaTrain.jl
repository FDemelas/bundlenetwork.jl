using BundleNetworks
using MLDatasets
using BSON: @save
using Instances
using Flux, Zygote
using Plots
using Random
using JSON
using LinearAlgebra
using Statistics
using TensorBoardLogger, Logging
using ParameterSchedulers

gap(a, b) = abs(a - b) / max(a, b) * 100


function main(layers_vector=[(28 * 28,784/4), (784/4, 10)])
	data = MNIST(:train)

	factory = BundleNetworks.AttentionModelFactory()
	nn = BundleNetworks.create_NN(factory)
	maxIt = 10
	opt = Flux.OptimiserChain(Flux.Optimise.Adam(0.001), ClipNorm(5))
	opt_st = Flux.setup(opt, nn)

	dataset = []
	bs = 1
	mI = (10 * bs)
	
	for idx in 0:bs:(mI-bs)
		ϕ = BundleNetworks.constructFunction(data[idx+1:idx+bs], 1.0, layers_vector)
		factory = BundleNetworks.AttentionModelFactory()
		nn_inner = Chain( [Dense(i => o, sigmoid) for (i,o) in layers_vector]...)
		p = Flux.params(nn_inner).params
		z = vcat([reshape(ip, :) for ip in p]...)
		B = BundleNetworks.initializeBundle(SoftBundleFactory(), ϕ, z, factory, maxIt + 1)
		B.maxIt = maxIt
		push!(dataset, (ϕ, B))
	end

	v = []
	losses = []
	nn_best = deepcopy(nn)
	for it in 1:100
		ls = []
		vs = []
		GC.gc()
		CUDA.free_memory()
		shuffle!(dataset)
		for (ϕ, B) in dataset
			BundleNetworks.reinitialize_Bundle!(B)
			BundleNetworks.reset!(nn)
			B.maxIt = 10
			vv, g = Flux.withgradient((m) -> -BundleNetworks.bundle_execution(B, [ϕ], m; δ = 0.0, γ = 0.999, metalearning = true, distribution_function = BundleNetworks.sparsemax), nn)
			_, nn = Flux.Optimisers.update!(opt_st, nn, g[1])
			vv = -vv
			append!(ls, vv)
			append!(vs, maximum(B.obj))
		end
		if mean(vs) > maximum(v; init = -Inf)
			nn_best = deepcopy(nn)
		end
		append!(v, mean(vs))
		append!(losses, mean(ls))
		println(it, " ", v[end])
	end

	@save "nn_best_0312.bson" nn_best
	@save "nn_end_0312.bson" nn
	return nn
end

#nn_ext = main()
