using BundleNetworks, Instances, Statistics, Flux, Zygote, LinearAlgebra, JSON;
import BundleNetworks: sizeLM, gap;
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
		default = "./data/MCNDforTest/"
		help = "data path"
		"--maxIterBundle"
		arg_type = Int64
		default = 100
		help = "Maximum Number of Iterations for the Bundle Variants."
		"--maxIterDescentType"
		arg_type = Int64
		default = 1000
		help = "Maximum Number of Iterations for the Descent Type Variants (Adam and Descent)."
		"--TS"
		arg_type = Float64
		nargs = '*'
		help = "Vector of initialization for the step-size/regularization-parameter that should be tested"
	end

	parsed_args = parse_args(args, s)

	TS = parsed_args["TS"]#[10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1]
	folder = parsed_args["folder"]
	mI = parsed_args["maxIterBundle"]
	mI_DT = parsed_args["maxIterDescentType"]

	directory = readdir(folder)

	format = split(directory[1], ".")[end]
	act = contains(folder,"GA") ? relu : identity
	batches = collect(1:length(directory))
	opts = [
		("Descent", Descent),
		("Adam", Adam),
		("Bundle constant", BundleNetworks.constant_t_strategy()),
		("Bundle soft", BundleNetworks.soft_long_term_t_strategy(BundleNetworks.heuristic_t_strategy_1())),
		("Bundle hard", BundleNetworks.hard_long_term_t_strategy(BundleNetworks.heuristic_t_strategy_1())),
		("Bundle balancing", BundleNetworks.balancing_long_term_t_strategy(BundleNetworks.heuristic_t_strategy_1())),
	]

	res = Dict(m[1] => Dict(t => Dict(i => Dict() for i in directory) for t in TS) for m in opts)
	

	f = JSON.open("./golds/MCNDforTest/gold.json", "r")
	golds = JSON.parse(f)
	close(f)

	BundleNetworks.device=cpu
	device=cpu

	for i in 1:length(directory)
		for t in TS
			for method in opts
				v_b = []
				t0 = time()

				#                   			ins,gold=my_read_dat_json(folder*directory[i]);
				ins, gold = [], []
				if format == "json"
					ins, gold = contains(folder,"GA") ? my_read_ga_json(folder * directory[i]) : my_read_dat_json(folder * directory[i])
#					ins, gold = my_read_dat_json(folder * directory[i])
				else
					ins, gold = my_read_dat(folder * directory[i]), golds[directory[i]]
				end
				tInit = time()
				if !(contains(method[1], "Bundle"))
					opt = method[2](t)
					ϕ = BundleNetworks.constructFunction(ins, 1.0)
					v, g = BundleNetworks.value_gradient(ϕ, zeros(sizeInputSpace(ϕ)))
					ϕ = BundleNetworks.constructFunction(ins,sqrt(sum(g .* g)))
					z = zeros(BundleNetworks.numberSP(ϕ))
					bz = copy(z)
					updated = false
					best_ϕ = -Inf
					l_ep = 0
					ss = t
					timesD = Dict("lsp" => [], "update" => [], "times" => [])
					state = Flux.setup(opt, z)
					for ep in 1:(mI_DT)
						t0 = time()
						vi, ∂L = value_gradient(ϕ, z)
						append!(timesD["lsp"], time() - t0)
						t1 = time()
						Flux.update!(state, z, -∂L)
						z = act(z)
						append!(v_b, vi .* ϕ.rescaling_factor)
						if vi > best_ϕ
							bz = z
							best_ϕ = vi
							updated = true
							l_ep = ep
						else
							updated = false
						end
						ss = l_ep - ep > 2 ? ss / 2 : ss
						opt = method[2](ss)
						append!(timesD["update"], time() - t1)
						append!(timesD["times"], time() - t0)

					end
					res[method[1]][t][directory[i]]["times"] = timesD

				else
					ϕ = BundleNetworks.constructFunction(ins,1.0)
					v, g = BundleNetworks.value_gradient(ϕ, zeros(sizeInputSpace(ϕ)))
					ϕ = BundleNetworks.constructFunction(ins,sqrt(sum(g .* g)))
					factory = BundleNetworks.RnnTModelfactory()
					B = BundleNetworks.initializeBundle(VanillaBundleFactory(),ϕ, t, zeros(BundleNetworks.numberSP(ϕ)))
					B.params.maxIt = mI
					oc,tD=BundleNetworks.solve!(B, ϕ; t_strat = method[2])
					v_b = B.all_objs[2:end] .* ϕ.rescaling_factor
					res[method[1]][t][directory[i]]["times"] = tD

				end
				res[method[1]][t][directory[i]]["time"] = time() - tInit
				res[method[1]][t][directory[i]]["obj"] = v_b

			end
		end
	end


	f = open("Results_" * split(folder, "/")[end-1] * ".json", "w")
	JSON.print(f, res)
	close(f)
end



main(ARGS)
