using BundleNetworks
using Instances
using Flux, Zygote
using Plots
using Random
using JSON
using LinearAlgebra
using Statistics
using TensorBoardLogger, Logging
using BSON: @load
using ArgParse

include("../runs/readingFunctions.jl")

function main(args)
	s = ArgParseSettings("Training an unrolling model" *
						 "version info, default values, " *
						 "options with types, variable " *
						 "number of arguments.",
		version = "Version 1.0", # version info
		add_version = true)      # audo-add version option

	@add_arg_table! s begin
		"--folder"
		arg_type = String
		default = "/data1/demelas/MCNDsmallCom40/"
		help = "optimizer for the training"
		"--model_folder"
		arg_type = String
		default = "BatchVersion_bs_1_true_MCNDsmallCom40_1.0e-6_0.9_5_200_200_1_10_49_true_128_false_softplus_false_0.999_0.0_0.0_sparsemax_false_false_3.0"
		help = "optimizer for the training"
                "--name"
                arg_type = String
                default = "1"
                help = "optimizer for the training"
                arg_type = String
                default = "BatchVersion_bs_1_true_MCNDsmallCom40_1.0e-6_0.9_5_200_200_1_10_49_true_128_false_softplus_false_0.999_0.0_0.0_sparsemax_false_false_3.0"
                "--dataset_folder"
		help = "optimizer for the training"
                arg_type = String
                default = "-1"
                help = "optimizer for the training"
		"--iterations"
		arg_type = Int64
		help = "Maximum number of optimizer iterations"
		default = 250
	end

	# take the input parameters and construct a Dictionary
	parsed_args = parse_args(args, s)

	folder = parsed_args["folder"]
	model_folder = parsed_args["model_folder"]
	name = parsed_args["name"]
	dataset_folder = parsed_args["dataset_folder"] == "-1" ? parsed_args["model_folder"] : parsed_args["dataset_folder"]


	maxIt = parsed_args["iterations"]

	res = Dict("times" => Dict(), "objs" => Dict())
	@load "res/$(model_folder)/nn_best.bson" nn_best

	global nn = (nn_best)
	nn.encoder = Chain(x -> identity(x), nn.encoder[2:end]...)
	nn.encoder = gpu(nn.encoder)
	nn.decoder_t = gpu(nn.decoder_t)
	nn.decoder_γk = gpu(nn.decoder_γk)
	nn.decoder_γq = gpu(nn.decoder_γq)
	nn=gpu(nn)
	nn.sample_γ=false
	nn.sample_t=false
	dataset_path = "res/$(dataset_folder)/dataset.json"
	f = JSON.open(dataset_path, "r")
	dataset = JSON.parse(f)
	close(f)

	factory = BundleNetworks.AttentionModelFactory()

	directory = readdir(folder)
	format = split(directory[1], ".")[end]
	soft_updates = true
	for idx in dataset["test"]
			# dataset["training"]                
			ins = []
			if format == "json"
				ins, _ = my_read_dat_json(folder * idx)
			else
				ins = my_read_dat(folder * idx)
			end
			ϕ = BundleNetworks.constructFunction(ins, 1.0)
			_, g = BundleNetworks.value_gradient(ϕ, zeros(sizeInputSpace(ϕ)))
			ϕ = [BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))]
			B = BundleNetworks.initializeBundle(BundleNetworks.SoftBundleFactory(), ϕ, [zeros(sizeK(ins) * sizeV(ins))], factory, maxIt + 1)

			B.maxIt = maxIt+1
			BundleNetworks.reset!(nn, 1, maxIt+1)
			mv = 0.0
			BundleNetworks.reinitialize_Bundle!(B)
			B.maxIt = maxIt
			v, times = BundleNetworks.bundle_execution(B, ϕ, nn; soft_updates = soft_updates, λ = 0.0, γ = 0.0, δ = 0.0, distribution_function = BundleNetworks.sparsemax, verbose = 0, inference = true)
			res["times"][idx] = times
			res["objs"][idx] = reshape(B.obj[1:B.li], :) .* ϕ[1].rescaling_factor
			println(v, " ", maximum(reshape(B.obj[1:B.li], :)))
	end


	f = open("Results_$(name)_$(split(folder,"/")[end-1]).json", "w")
	JSON.print(f, cpu(res))
	close(f)
end

main(ARGS)
