using BundleNetworks
using Instances
using Flux, Zygote
using Plots
using Random
using JSON
using LinearAlgebra
using Statistics
using TensorBoardLogger, Logging

include("../runs/readingFunctions.jl")

gap(a, b) = abs(a - b) / max(a, b) * 100

using CUDA
using BSON: @save
using ArgParse
#include("./runs/testBundleNetworks.jl")
using Logging
using TensorBoardLogger
using MLUtils
using ParameterSchedulers
using ChainRules, ChainRulesCore


function main(args)
	s = ArgParseSettings("Training an unrolling model" *
						 "version info, default values, " *
						 "options with types, variable " *
						 "number of arguments.",
		version = "Version 1.0", # version info
		add_version = true)      # audo-add version option

	@add_arg_table! s begin
		"--data"
		arg_type = String
		default = "./data/MCNDforTest/"
		help = "optimizer for the training"
		"--lr"
		required = true
		arg_type = Float64
		help = "learning rate"
		"--decay"
		default = 0.9
		arg_type = Float64
		help = "learning rate"
		"--lambda"
		default = 0.0
		arg_type = Float32
		help = "the final loss will be lambda*(phi(last_trial_point))+(1-lambda)*(phi(last_stabilization_trial_point)) + ..."
		"--gamma"
		default = 0.0
		arg_type = Float32
		help = "parameter that controll the weights in the telescopic sum of all the objective values of the trial_points founded during the execution"
		"--delta"
		default = 0.0
		arg_type = Float32
		help = "learning rate"
		"--cn"
		default = 5
		arg_type = Int64
		help = "Clip Norm"
		"--mti"
		required = true
		arg_type = Int64
		help = "Maximum number of training instances"
		"--mvi"
		required = true
		arg_type = Int64
		help = "Maximum number of validation instances"
		"--seed"
		required = true
		arg_type = Int64
		help = "Random seed"
		"--maxItBack"
		required = true
		arg_type = Int64
		help = "Maximum number of Unrolled iteration for Backward."
		"--maxIt"
		default = -1
		arg_type = Int64
		help = "Maximum number of Unrolled iteration."
		"--maxItVal"
		default = 100
		arg_type = Int64
		help = "Maximum number of Unrolled iteration for the validation, by default 100."
		"--maxEP"
		required = true
		arg_type = Int64
		help = "Maximum number of training epochs."
		"--soft_updates"
		arg_type = Bool
		default = true
		help = "If true we use soft updates (with softmax) for the stabilization point"
		"--h_representation"
		arg_type = Int64
		default = 64
		help = "Size of the hidden representation for the NN model."
		"--use_softmax"
		arg_type = Bool
		default = true
		help = "If true use the softmax to compute the distribution used to provide the convex combination of the gradients, otherwhise use the sparsemax."
		"--use_graph"
		arg_type = Bool
		default = true
		help = "If true use the bipartite graph representation to compute additional features"
		"--batch_size"
		arg_type = Int64
		default = 1
		help = "Batch size. By default 1."
		"--incremental"
		arg_type = Bool
		default = false
		help = "If false it reduce to a 'standard' learning task with maxEP epochs and maxIt unrolling iterations of the bundle. If true at the first (maxEP)/2 epochs the number of Bundle iterations will be incremental. More precisely we start from 2*maxIt/maxEp and at each epoch we increment of 2*maxIt/maxEp, in such a way that, after maxEp/2 epochs we use maxIt iterations . By default false."
		"--always_batch"
		arg_type = Bool
		default = false
		help = "If true use the batch size implementation even if the batch size is equal to one."
		"--h_act"
		arg_type = String
		default = "softplus"
		help = "Activation function to use in the model."
		"--sampling_gamma"
		arg_type = Bool
		default = false
		help = "sample in the latent space to predicts keys and queries used to compute theta and then the coefficients thetas of the convex combination."
		 "--sampling_t"
                arg_type = Bool
                default = true
                help = "sample in the parameter t"
		"--reduced_components"
		arg_type = Bool
		default = false
		help = "If true add a component to the bundle only if it was not visited a point with the same sub-gradient. Otherwhise not add the component, but update the associated values."
		"--scheduling_ss"
                arg_type = Int64
                default = 10
                help = ""
	end

	# take the input parameters and construct a Dictionary
	parsed_args = parse_args(args, s)
	
	folder = parsed_args["data"]
	lr = parsed_args["lr"]
	decay = parsed_args["decay"]
	cn = parsed_args["cn"]
	mti = parsed_args["mti"]
	mvi = parsed_args["mvi"]
	seed = parsed_args["seed"]
	maxIt = parsed_args["maxIt"]
	maxItVal = parsed_args["maxItVal"]
	maxEp = parsed_args["maxEP"]
	soft_updates = parsed_args["soft_updates"]
	h_representation = parsed_args["h_representation"]
	use_softmax = parsed_args["use_softmax"]
	gamma = parsed_args["gamma"]
	lambda = parsed_args["lambda"]
	delta = parsed_args["delta"]
	maxBaIt = parsed_args["maxItBack"]
	maxBaIt = maxBaIt < 0 ? maxIt : maxBaIt
	bgr = parsed_args["use_graph"]
	batch_size = parsed_args["batch_size"]
	incremental = parsed_args["incremental"]
	a_b = parsed_args["always_batch"]
	scheduling_ss = parsed_args["scheduling_ss"]
	h_act = (parsed_args["h_act"] == "softplus") ? softplus : (parsed_args["h_act"] == "tanh" ? tanh : (parsed_args["h_act"] == "gelu" ? gelu : relu))
	sampling_θ = parsed_args["sampling_gamma"]
	sampling_t = parsed_args["sampling_t"]
	act= contains(folder,"GA") ? relu : identity

	reduced_components = parsed_args["reduced_components"]
	distribution_function = use_softmax ? softmax : (BundleNetworks.sparsemax)
	rng = Random.MersenneTwister(seed)
	directory = shuffle(rng, readdir(folder))
	format = split(directory[1], ".")[end]

	factory = bgr ? BundleNetworks.AttentionModelFactory() : BundleNetworks.AttentionModelFactory()
	global nn = BundleNetworks.create_NN(factory; h_representation, h_act, sampling_θ,sampling_t)
	nn.h_representation = BundleNetworks.h_representation(nn)
	BundleNetworks.reset!(nn, batch_size)
	use_gpu = true
	device = CUDA.functional() && use_gpu ? gpu : cpu
	dataset = []
	gold = Dict()
	println(format)
	
	if format == "dat"
		tmp_idx = 0
		for f in directory[1:(mti+mvi)]
			ins = my_read_dat(folder * f)
			ϕ = BundleNetworks.constructFunction(ins, 1.0)
			_, g = BundleNetworks.value_gradient(ϕ, zeros(sizeLM(ins)))
			ϕ = BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))
			push!(dataset, (f, ϕ))
			tmp_idx += 1
			if tmp_idx % 100 == 0
				GC.gc()
				CUDA.reclaim()
				CUDA.free_memory()
			end
		end

		f = JSON.open("./golds/" * split(folder, "/")[end-1] * "/gold.json", "r")
		gold = JSON.parse(f)
		close(f)

	else
		if contains(directory[1],"uc")
			tmp_idx = 0
			for f in directory[1:(mti+mvi)]
                                ins = Instances.read_json(folder*f)
				Ld = 0.0
				ϕ = BundleNetworks.constructFunction(ins, 1.0)
                                _, g = BundleNetworks.value_gradient(ϕ, zeros(sizeInputSpace(ϕ)))
                                ϕ = BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))
                                push!(dataset, (f, ϕ))
                                gold[f] = Ld
                                tmp_idx += 1
                                if tmp_idx % 100 == 0
                                	GC.gc()
                               		CUDA.reclaim()
                                	CUDA.free_memory()
				end
			end
		else
			tmp_idx = 0
			for f in directory[1:(mti+mvi)]
				ins, Ld = contains(folder,"GA") ? my_read_ga_json(folder * f) : my_read_dat_json(folder * f)
				ϕ = BundleNetworks.constructFunction(ins, 1.0)
				_, g = BundleNetworks.value_gradient(ϕ, zeros(sizeLM(ins)))
				ϕ = BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))
				push!(dataset, (f, ϕ))
				gold[f] = Ld
				tmp_idx += 1
				if tmp_idx % 100 == 0
					GC.gc()
					CUDA.reclaim()
					CUDA.free_memory()
				end
			end
		end

	end

	opt = Flux.OptimiserChain(Flux.Optimise.Adam(lr), ClipNorm(cn))
	opt_st = Flux.setup(opt, nn)
	scheduler = ParameterSchedulers.Stateful(Exp(start = lr, decay = decay))

	idxs = collect(1:mti)
	idxs_v = collect((mti+1):(mti+mvi))
	idxs_test = collect((mti+mvi+1):length(directory))
	global nn_best = deepcopy(nn)
	global nn_val = deepcopy(nn)
	nn_val.sample_t = false
	nn_val.sample_γ = false
	results = Dict("training" => Dict(), "validation" => Dict(), "test" => Dict())
	v, losses, gaps = [], [], []
	v_v, losses_v, gaps_v = [], [], []
	v_v_ti, losses_v_ti, gaps_v_ti = [], [], []
	times_val = []
	cum_grads = []
	ciwi = 0
	last_loss = Inf
	last_train_loss = Inf
	threshold = 1.0e-7
	#idxs=idxs[1:10];

	res_folder =
		"BatchVersion_bs_" * string(batch_size) * "_seed"*string(seed)*"_" * string(a_b) * "_" * string(split(folder, "/")[end-1]) * "_" * string(lr) * "_" * string(decay) * "_" * string(cn) * "_" * string(mti) * "_" * string(mvi) * "_" * string(seed) * "_" * string(maxIt) *
		"_" * string(maxEp) * "_" * string(soft_updates) * "_" *
		string(h_representation) * "_" * string(sampling_θ) * string(sampling_t)* "_" * string(h_act) * "_" * string(use_softmax) * "_" * string(gamma) * "_" * string(lambda) * "_" * string(delta) * "_" * string(distribution_function) * "_" * string(bgr) * "_" *
		string(incremental)*"_rc"*string(reduced_components)*"_ss"*string(scheduling_ss)
	sN = sum([1 for j in readdir("resLogs") if contains(j, res_folder)]; init = 0.0)
	res_folder = "resLogs/" * res_folder * "_" * string(sN + 1)
	mkdir(res_folder)
	f = JSON.open(res_folder * "/dataset.json", "w")
	JSON.print(f, Dict("training" => directory[idxs], "validation" => directory[idxs_v], "test" => directory[idxs_test]))
	close(f)

	batched_trainset = dataset[idxs]
	batched_valset = dataset[idxs_v]
	iterations_without_improvement = 0
	last_loss = Inf
	lg = TBLogger(res_folder, min_level = Logging.Info)
	with_logger(lg) do
		for it in 1:maxEp
			ls, ls_v, ls_v_tI = [], [], []
			gs, gs_v, gs_v_tI = [], [], []
			vs, vs_v, vs_v_tI = [], [], []
			time_val = 0.0
			shuffle!(rng, batched_trainset)
			first = 1
			last = batch_size
			for it_idx in 1:ceil(mti / batch_size)
				sample = batched_trainset[first:last]
				idx = batch_size > 1 || a_b ? [s[1] for s in sample] : sample[1][1]
				ϕ = batch_size > 1 || a_b ? [s[2] for s in sample] : sample[1][2]
				z = batch_size == 1 && !a_b ? zeros(prod(sizeInputSpace(ϕ))) : [zeros(prod(sizeInputSpace(ϕi))) for ϕi in ϕ]
				bt = batch_size == 1 && !a_b ? SoftBundleFactory() : BatchedSoftBundleFactory()
				B = BundleNetworks.initializeBundle(bt, ϕ, z, factory, maxIt + 1,reduced_components)
				B.maxIt = incremental ? min(2 * it * maxIt / maxEp, maxIt) : maxIt

				BundleNetworks.reset!(nn, max(1, last - first + 1))

				BundleNetworks.reinitialize_Bundle!(B)
				mv = 0.0
				B.maxIt = incremental ? min(2 * it * maxIt / maxEp, maxIt) : maxIt
				r_f = (batch_size > 1 || a_b ? sum(numberSP(f) for f in ϕ) : numberSP(ϕ)) * maxIt / (last - first + 1)
				vv, grads = Flux.withgradient((m) -> .- BundleNetworks.bundle_execution(B, ϕ, m; soft_updates = soft_updates, λ = lambda, γ = gamma, δ = delta, distribution_function, verbose = 0,inference=false,act) / r_f, nn)
				vv = -vv

				_, nn = Flux.Optimisers.update!(opt_st, nn, grads[1])

				mv += vv / round(maxIt / maxBaIt)
				#end
				if batch_size == 1 && !a_b
					append!(ls, mv)
					append!(vs, maximum(B.obj) * ϕ.rescaling_factor)
					append!(gs, gap(maximum(B.obj) * ϕ.rescaling_factor, gold[idx]))
				else
					append!(ls, mv)
					append!(vs, mean(maximum(B.obj[j, :]) * ϕ[j].rescaling_factor for j in eachindex(idx)))
					append!(gs, mean([gap(maximum(B.obj[j, :]) * ϕ[j].rescaling_factor, gold[idx[j]]) for j in eachindex(idx)]))
				end
				#				nn = B.nn
				first += batch_size
				last += batch_size
				last = min(last, mti)

			end


			GC.gc()
			CUDA.reclaim()
			CUDA.free_memory()

			@info "Train" GAP_percentage = mean(gs) log_step_increment = 0
			@info "Train" LSP_value = mean(vs) log_step_increment = 0
			@info "Train" Loss_value = mean(ls) log_step_increment = 0

			append!(v, mean(vs))
			append!(losses, mean(ls))
			append!(gaps, mean(gs))


			if it % scheduling_ss == 0
				#if last_train_loss < losses[end]
				Flux.adjust!(opt_st, ParameterSchedulers.next!(scheduler))
			end
			
			nn_val.encoder=nn.encoder
			nn_val.decoder_t=nn.decoder_t
			nn_val.decoder_γk=nn.decoder_γk
			nn_val.decoder_γq=nn.decoder_γq


			first = 1
			last = batch_size
			for it_idx in 1:ceil(mvi / batch_size)
				sample = batched_valset[first:last]
				idx = batch_size > 1 || a_b ? [s[1] for s in sample] : sample[1][1]
				ϕ = batch_size > 1 || a_b ? [s[2] for s in sample] : sample[1][2]
				z = batch_size == 1 && !a_b ? zeros(prod(sizeInputSpace(ϕ))) : [zeros(prod(sizeInputSpace(ϕi))) for ϕi in ϕ]
				bt = batch_size == 1 && !a_b ? SoftBundleFactory() : BatchedSoftBundleFactory()
				B = BundleNetworks.initializeBundle(bt, ϕ, z, factory, max(maxIt, maxItVal) + 1,reduced_components)
				B.maxIt = max(maxIt, maxItVal)
				BundleNetworks.reset!(nn_val, max(1, last - first + 1))

				BundleNetworks.reinitialize_Bundle!(B)
				t0 = time()
				r_f = (batch_size > 1 || a_b ? sum(sizeE(f.inst) for f in ϕ) : sizeE(ϕ.inst)) * maxIt / (last - first + 1)
				val = BundleNetworks.bundle_execution(B, ϕ, nn_val; soft_updates = soft_updates, λ = lambda, γ = gamma, δ = delta, distribution_function, verbose = 0,inference=false,act) / r_f
				
				time_val = time() - t0
				append!(ls_v, val)

				if batch_size == 1 && !a_b
					append!(vs_v, maximum(B.obj) * ϕ.rescaling_factor)
					append!(gs_v, gap(maximum(B.obj) * ϕ.rescaling_factor, gold[idx]))
					append!(vs_v_tI, maximum(B.obj[ 1:min(maxIt, maxItVal)]))
					append!(gs_v_tI, gap(maximum(B.obj[1:min(maxIt, maxItVal)]) * ϕ.rescaling_factor, gold[idx]))
				else
					append!(vs_v, mean(maximum(B.obj[j, :]) * ϕ[j].rescaling_factor for j in eachindex(idx)))
					append!(gs_v, mean([gap(maximum(B.obj[j, :]) * ϕ[j].rescaling_factor, gold[idx[j]]) for j in eachindex(idx)]))
					append!(vs_v_tI, mean(maximum(B.obj[j, 1:min(maxIt, maxItVal)]) * ϕ[j].rescaling_factor for j in eachindex(idx)))
					append!(gs_v_tI, mean([gap(maximum(B.obj[j, 1:min(maxIt, maxItVal)]) * ϕ[j].rescaling_factor, gold[idx[j]]) for j in eachindex(idx)]))
					first += batch_size
					last += batch_size
					last = min(last, mvi)

				end

			end
			if mean(gs_v_tI) > minimum(gaps_v; init = Inf)
				#mean(vs_v) > maximum(v_v; init = -Inf)
				nn_best = deepcopy(nn)
			end
			append!(v_v, mean(vs_v))
			append!(losses_v, mean(ls_v))
			append!(gaps_v, mean(gs_v))


			append!(v_v, mean(vs_v_tI))
			append!(gaps_v, mean(gs_v_tI))
			append!(times_val, time_val)


			if abs(last_loss - losses_v[end]) < threshold
				ciwi += 1
			else
				ciwi = 0
			end

			last_loss = losses_v[end]
			if ciwi > 3
				#break
			end

			@info "Validation" GAP_percentage = mean(gs_v) log_step_increment = 0
			@info "Validation" GAP_percentage_li = mean(gs_v_tI) log_step_increment = 0
			@info "Validation" LSP_value = mean(vs_v) log_step_increment = 0
			@info "Validation" Loss_value = mean(ls_v) log_step_increment = 1


			println(it, " Training - lsp: ", v[end], "  gap: ", gaps[end])
			println(it, " Validation - lsp: ", v_v[end], "  gap: ", gaps_v[end])

			GC.gc()
			CUDA.reclaim()
			CUDA.free_memory()

		end
	end
	nn = cpu(nn)
	nn_best = cpu(nn_best)




	@save res_folder * "/nn.bson" nn
	@save res_folder * "/nn_best.bson" nn_best

	f = open(res_folder * "/loss.json", "w")
	JSON.print(f, losses)
	close(f)

	f = open(res_folder * "/obj.json", "w")
	JSON.print(f, v)
	close(f)

	f = open(res_folder * "/gaps.json", "w")
	JSON.print(f, gaps)
	close(f)

	f = open(res_folder * "/obj_val.json", "w")
	JSON.print(f, Dict("maxIt" => v_v_ti, "maxItVal" => v_v))
	close(f)

	f = open(res_folder * "/gaps_val.json", "w")
	JSON.print(f, Dict("maxIt" => gaps_v_ti, "maxItVal" => gaps_v))
	close(f)

	f = open(res_folder * "/val_times.json", "w")
	JSON.print(f, times_val)
	close(f)


end

main(ARGS)
