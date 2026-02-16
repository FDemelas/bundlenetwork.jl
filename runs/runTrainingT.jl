using BundleNetworks, Instances, Statistics, Flux, LinearAlgebra, JuMP, JSON
using ArgParse
using BundleNetworks
using Instances
using Flux
using Random
using JSON
using Statistics
using BSON: @save, @load
using TensorBoardLogger, Logging
#using CUDA

include("../runs/readingFunctions.jl")

"""
    ep_train_and_val(folder, directory, dataset, gold, idxs_train, idxs_val, opt; kwargs...)

Execute episodic training and validation for neural network-guided bundle methods.

# Arguments
- `folder::String`: Path to the instance folder
- `directory::Vector{String}`: List of all instance files
- `dataset::Vector{Tuple}`: Dataset containing (filename, objective_function) pairs
- `gold::Dict`: Dictionary mapping instance names to optimal objective values
- `idxs_train::Vector{Int}`: Indices of training instances in the dataset
- `idxs_val::Vector{Int}`: Indices of validation instances in the dataset
- `opt`: Flux optimizer for training

# Keyword Arguments
- `maxEP::Int=10`: Maximum number of training epochs
- `maxIT::Int=50`: Maximum number of bundle method iterations per training instance
- `lt`: Learning model factory (default: RnnTModelfactory())
- `t_strat`: Strategy for proximity parameter t (default: constant_t_strategy())
- `t::Float64=0.000001`: Initial proximity parameter for bundle method
- `nn`: Neural network model (default: created from lt factory)
- `test_mode::Bool=false`: If true, run only one epoch (testing mode)
- `retrain_mode::Bool=false`: If true, enable retraining mode
- `location::String="./results/"`: Directory to save results and models
- `cr_init::Bool=false`: If true, initialize from cutting-plane relaxation dual variables; otherwise from zero
- `exactGrad::Bool=true`: If true, use exact gradient formula for master problem solution w.r.t. t
- `telescopic::Bool=false`: If true, loss includes all visited points; if false, only final point
- `γ::Float64=0.1`: Weight decay factor for telescopic sum (γ^iteration)
- `use_gold::Bool=true`: If true, compute and log optimality gaps using gold solutions
- `instance_features::Bool=false`: If true, include instance-specific features as NN input
- `seed::Int=1`: Random seed for shuffling
- `single_prediction::Bool=false`: If true, use constant t parameter throughout training

# Description
This function implements an episodic training procedure where:
1. The model is trained on each training instance sequentially
2. After each epoch, the model is validated on the validation set
3. Metrics (objective values, times, optimality gaps) are tracked and logged
4. The best model (by validation objective) is saved

# Training vs Validation Differences
- Training: Uses maxIT iterations with gradient updates
- Validation: Uses 5*maxIT iterations without gradient updates (inference only)

# Returns
Nothing (results are saved to disk)

# Output Files
- `nn.bson`: Final trained neural network
- `nn_bestLV.bson`: Best neural network by validation objective
- `res_train.json`: Detailed training results per epoch and instance
- `res_val.json`: Detailed validation results per epoch and instance
- `res_values_train.json`: Training statistics (mean, std, quantiles)
- `res_times_train.json`: Training time statistics
- `res_values_val.json`: Validation statistics
- `res_times_val.json`: Validation time statistics
- `dataset.json`: Dataset split definition
"""
function ep_train_and_val(
	folder,
	directory,
	dataset,
	gold,
	idxs_train,
	idxs_val,
	opt;
	maxEP = 10,
	maxIT = 50,
	lt = BundleNetworks.RnnTModelfactory(),
	t_strat = BundleNetworks.constant_t_strategy(),
	t = 0.000001,
	nn = create_NN(lt),
	test_mode = false,
	retrain_mode = false,
	location = "./results/",
	cr_init::Bool = false,
	exactGrad = true,
	telescopic = false,
	γ = 0.1,
	use_gold = true,
	instance_features = false,
	seed = 1,
	single_prediction::Bool = false
)

	# Set computation device to CPU (GPU code commented out)
	BundleNetworks.device = cpu
	device = cpu

	# Detect file format from first file extension
	format = split(directory[1], ".")[end]

	# Save dataset split definition for reproducibility
	f = open(location * "dataset.json", "w")
	JSON.print(f, Dict(
		"train" => directory[idxs_train], 
		"validation" => directory[idxs_val], 
		"test" => [directory[i] for i in eachindex(directory) if !(i in idxs_val) && !(i in idxs_train)]
	))
	close(f)

	# Initialize result tracking structures
	res_values_train, res_times_train = [], []  # Training aggregated metrics
	res_values_val, res_times_val = [], []      # Validation aggregated metrics
	res = Dict()      # Detailed training results per epoch and instance
	res_val = Dict()  # Detailed validation results per epoch and instance
	
	# Pre-allocate result dictionaries for all epochs and instances
	for epoch in 1:maxEP
		res[epoch] = Dict()
		for ins_path in directory[idxs_train]
			res[epoch][ins_path] = Dict()
		end
	end

	for epoch in 1:maxEP
		res_val[epoch] = Dict()
		for ins_path in directory[idxs_val]
			f = split(split(ins_path, "/")[end], ".")[1]
			res_val[epoch][ins_path] = Dict()
		end
	end
	
	# Adjust epochs for test/retrain modes
	maxEP = test_mode ? 1 : maxEP
	test_mode = retrain_mode ? true : test_mode
	
	# Initialize model copies
	nn_copy = nn
	nn_best = deepcopy(nn)  # Best model tracker
	obj_best = 0.0          # Best validation objective

	# Initialize cutting-plane relaxation (CR) parameters for warm-starting
	crs = Dict()       # CR scaling factors
	cr_duals = Dict()  # CR dual variables
	for (idx, ϕ) in dataset
		inst = ϕ.inst
		if !(cr_init)
			# Simple initialization: zero duals and unit scaling
			cr_duals[idx] = zeros((sizeK(inst), sizeV(inst)))
			crs[idx] = 1.0
		else
			# Warm-start from cutting-plane relaxation solution
			crs[idx], cr_duals[idx] = CR(inst)[1:2]
		end
		GC.gc()
	end
	
	# Initialize random number generator with seed
	rng = Random.MersenneTwister(seed)

	# Setup optimizer state for the neural network
	state = Flux.setup(opt, nn)

	# Initialize TensorBoard logger for visualization
	lg = TBLogger(location, min_level = Logging.Info)
	
	# Main training loop
	with_logger(lg) do
		for epoch in 1:maxEP
			# Per-epoch metric accumulators
			values, times = Float64[], Float64[]
			
			# Shuffle training and validation sets each epoch
			shuffle!(rng, idxs_train)
			shuffle!(rng, idxs_val)
			
			# ========== TRAINING PHASE ==========
			for idx_t in idxs_train
				# Get instance and objective function
				ins_path, ϕ = dataset[idx_t]
				
				# Initialize bundle method solver with neural network
				B = initializeBundle(
					tLearningBundleFactory(), 
					ϕ, 
					t,                      # Proximity parameter
					cr_duals[ins_path],     # Initial dual variables
					lt,                     # Learning model factory
					nn, 
					maxIT; 
					exactGrad,              # Use exact gradient formula
					instance_features       # Include instance features
				)
				B.params.maxIt = maxIT

				# Training: compute gradients and update model
				t0 = time()
				co = BundleNetworks.train!(
					B, ϕ, state; 
					γ,                      # Telescopic weight decay
					samples = 1,            # Number of samples
					normalization_factor = 1.0,
					telescopic,             # Use telescopic loss
					single_prediction       # Constant t parameter
				)

				append!(times, time() - t0)
				# Evaluate objective at final solution
				append!(values, ϕ(reshape(BundleNetworks.zS(B), (sizeK(ϕ.inst), sizeV(ϕ.inst)))) * ϕ.rescaling_factor)

				# Update neural network from bundle (may have been modified during training)
				nn = B.nn

				# Store detailed results for this instance
				res[epoch][ins_path]["time"] = time() - t0
				res[epoch][ins_path]["obj"] = values[end]
				res[epoch][ins_path]["optimality"] = co
				
				GC.gc()
			end
			
			# Store aggregated training statistics for this epoch
			# Format: [mean, std, quantile_0%, quantile_25%, quantile_50%, quantile_75%, quantile_100%]
			push!(res_values_train, [mean(values), std(values), quantile(values)...])
			push!(res_times_train, [mean(times), std(times), quantile(times)...])

			# ========== VALIDATION PHASE ==========
			values, times = Float64[], Float64[]
			for idx_i in idxs_val
				println("Validation Instance")
				ins_path, ϕ = dataset[idx_i]
				
				# Initialize bundle with more iterations for validation (5*maxIT)
				B = initializeBundle(
					tLearningBundleFactory(), 
					ϕ, 
					t, 
					cr_duals[ins_path], 
					lt, 
					nn, 
					5 * maxIT + 1; 
					exactGrad, 
					instance_features
				)

				B.params.maxIt = 5 * maxIT

				# Validation: no gradient computation, just evaluate
				t0 = time()
				co = BundleNetworks.Bundle_value_gradient!(B, ϕ, false, single_prediction)
				println()
				
				append!(times, time() - t0)
				# Track best objective in first maxIT iterations
				append!(values, maximum(B.all_objs[1:maxIT+1]))
				
				# Reset to original model (validation doesn't update the model)
				nn = nn_copy

				# Store validation results at different iteration counts
				f = ins_path
				res_val[epoch][f]["time"] = time() - t0
				res_val[epoch][f]["obj maxIT"] = maximum(B.all_objs[1:maxIT+1]) * ϕ.rescaling_factor
				res_val[epoch][f]["obj 2*maxIT"] = maximum(B.all_objs[1:2*maxIT+1]) * ϕ.rescaling_factor
				res_val[epoch][f]["obj 5*maxIT"] = maximum(B.all_objs[1:end]) * ϕ.rescaling_factor
				res_val[epoch][f]["optimality"] = co
				
				GC.gc()
			end
			
			# Store aggregated validation statistics
			push!(res_values_val, [mean(values), std(values), quantile(values)...])
			push!(res_times_val, [mean(times), std(times), quantile(times)...])

			# Compute mean validation objectives at different iteration counts
			objV = mean([res_val[epoch][f]["obj maxIT"] for f in keys(res_val[epoch])])
			objVx2 = mean([res_val[epoch][f]["obj 2*maxIT"] for f in keys(res_val[epoch])])
			objVx5 = mean([res_val[epoch][f]["obj 5*maxIT"] for f in keys(res_val[epoch])])

			# If gold solutions are available, compute and log optimality gaps
			if use_gold
				# Percentage gap: |solution - optimal| / optimal * 100
				GAP_t = mean([abs(res[epoch][f]["obj"] - gold[f]) / gold[f] for f in keys(res[epoch])]) * 100
				GAP_v = mean([abs(res_val[epoch][f]["obj maxIT"] - gold[f]) / gold[f] for f in keys(res_val[epoch])]) * 100
				GAP_vx5 = mean([abs(res_val[epoch][f]["obj 5*maxIT"] - gold[f]) / gold[f] for f in keys(res_val[epoch])]) * 100
				GAP_vx2 = mean([abs(res_val[epoch][f]["obj 2*maxIT"] - gold[f]) / gold[f] for f in keys(res_val[epoch])]) * 100

				# Log gaps to TensorBoard
				@info " Train " GAP_percentage = GAP_t log_step_increment = 0
				@info " Validation " GAP_percentage = GAP_v log_step_increment = 0
				@info " Validation x5" GAP_percentage = GAP_vx5 log_step_increment = 0
				@info " Validation x2" GAP_percentage = GAP_vx2 log_step_increment = 0
			end
			
			# Track best model based on validation objective at maxIT iterations
			if obj_best < objV
				obj_best = objV
				nn_best = deepcopy(nn)
			end

			# Log objective values to TensorBoard
			@info " Train " LSP = res_values_train[end][1] log_step_increment = 0
			@info " Validation x5 " LSP = objVx5 log_step_increment = 0
			@info " Validation " LSP = objV log_step_increment = 1

		end
	end
	
	# Save final and best models
	@save (location * "nn.bson") nn
	@save (location * "nn_bestLV.bson") nn_best

	# Save all training results
	saveJSON(location * "res_train.json", res)

	f = open(location * "res_values_train.json", "w")
	JSON.print(f, res_values_train)
	close(f)

	f = open(location * "res_times_train.json", "w")
	JSON.print(f, res_times_train)
	close(f)

	# Save all validation results
	saveJSON(location * "res_val.json", res_val)

	f = open(location * "res_values_val.json", "w")
	JSON.print(f, res_values_val)
	close(f)

	f = open(location * "res_times_val.json", "w")
	JSON.print(f, res_times_val)
	close(f)
	
	return  # Results saved to disk
end

"""
    saveJSON(name, res)

Helper function to save a dictionary to a JSON file.

# Arguments
- `name::String`: Output file path
- `res::Dict`: Dictionary to save
"""
function saveJSON(name, res)
	f = open(name, "w")
	JSON.print(f, res)
	close(f)
end

"""
    main(args)

Main entry point for the episodic training script.

# Arguments
- `args::Vector{String}`: Command-line arguments (typically ARGS)

# Description
This script implements episodic training for neural network-guided bundle methods,
where the model is trained one instance at a time (episodic) rather than in batches.
This approach is useful for:
- Online learning scenarios
- When instances are very heterogeneous
- When memory constraints prevent batching

# Key Features
- Episodic (instance-by-instance) training
- Optional cutting-plane relaxation initialization
- Telescopic loss (considering all visited points)
- Validation at multiple iteration counts (maxIT, 2*maxIT, 5*maxIT)
- TensorBoard logging of gaps and objectives

# Command-line Arguments (Required)
- `--lr`: Learning rate
- `--mti`: Maximum number of training instances
- `--mvi`: Maximum number of validation instances
- `--seed`: Random seed
- `--maxIT`: Maximum bundle iterations per training instance
- `--maxEP`: Maximum training epochs

# Command-line Arguments (Optional)
- `--data`: Instance folder path (default: "ADAM")
- `--cn`: Gradient clipping norm (default: -1, no clipping)
- `--cr_init`: Initialize from CR duals (default: false)
- `--exactGrad`: Use exact gradient formula (default: true)
- `--telescopic`: Use telescopic loss (default: false)
- `--gamma`: Telescopic weight decay (default: 0.1)
- `--use_gold`: Compute gaps with gold solutions (default: true)
- `--gold_location`: Path to gold solutions (default: "../golds/MCNDforTest/gold.json")
- `--instance_features`: Include instance features (default: true)
- `--single_prediction`: Use constant t parameter (default: false)
- `--dataset_location`: Path to dataset split file (default: "-1", auto-generate)
- `--sample_inside`: Sample inside model vs output space (default: true)

# Output
Creates a results folder `resLogs/res_goldLossWeights_...` containing:
- Trained models (nn.bson, nn_bestLV.bson)
- Detailed results per epoch and instance
- Aggregated statistics (mean, std, quantiles)
- TensorBoard logs
"""
function main(args)
	# Configure argument parser
	s = ArgParseSettings(
		"Training an unrolling model" *
		"version info, default values, " *
		"options with types, variable " *
		"number of arguments.",
		version = "Version 1.0",  # Version information
		add_version = true        # Auto-add --version option
	)

	# Define all command-line arguments
	@add_arg_table! s begin
		"--data"
			arg_type = String
			default = "ADAM"
			help = "path to instance folder (or optimizer name in old version)"
		"--lr"
			required = true
			arg_type = Float32
			help = "learning rate for Adam optimizer"
		"--cn"
			default = -1
			arg_type = Int64
			help = "Gradient clipping norm. If -1, no clipping is applied"
		"--mti"
			required = true
			arg_type = Int64
			help = "Maximum number of training instances to use"
		"--mvi"
			required = true
			arg_type = Int64
			help = "Maximum number of validation instances to use"
		"--seed"
			required = true
			arg_type = Int64
			help = "Random seed for reproducibility (shuffling, initialization)"
		"--maxIT"
			required = true
			arg_type = Int64
			help = "Maximum number of unrolled bundle method iterations during training"
		"--maxEP"
			required = true
			arg_type = Int64
			help = "Maximum number of training epochs"
		"--cr_init"
			arg_type = Bool
			default = false
			help = "If true, initialize dual variables from cutting-plane relaxation; otherwise from zero"
		"--exactGrad"
			arg_type = Bool
			default = true
			help = "If true, use exact gradient formula for ∂MP/∂t; otherwise approximate as zero"
		"--telescopic"
			arg_type = Bool
			default = false
			help = "If true, loss includes all visited points weighted by γ^iteration; if false, only final point"
		"--gamma"
			arg_type = Float32
			default = 0.1
			help = "Weight decay factor for telescopic sum: loss = Σ γ^i * ϕ(x_i)"
		"--use_gold"
			arg_type = Bool
			default = true
			help = "If true, compute and log optimality gaps using gold (optimal) solutions"
		"--gold_location"
			arg_type = String
			default = "../golds/MCNDforTest/gold.json"
			help = "Path to JSON file containing gold (optimal) solution values"
		"--instance_features"
			arg_type = Bool
			default = true
			help = "If true, include static instance features (from linear relaxation) as NN input"
		"--single_prediction"
			arg_type = Bool
			default = false
			help = "If true, use constant proximity parameter t; if false, predict t at each iteration"
		"--dataset_location"
			arg_type = String
			default = "-1"
			help = "Path to pre-defined dataset split JSON. If '-1', auto-generate split from folder"
		"--sample_inside"
			arg_type = Bool
			default = true
			help = "If true, sample inside the model (latent space); if false, sample in output space"
	end

	# Parse command-line arguments
	parsed_args = parse_args(args, s)

	# Extract all parameters
	folder = parsed_args["data"]
	lr = parsed_args["lr"]
	cn = parsed_args["cn"]
	mti = parsed_args["mti"]
	mvi = parsed_args["mvi"]
	seed = parsed_args["seed"]
	maxIT = parsed_args["maxIT"]
	maxEP = parsed_args["maxEP"]
	cr_init = parsed_args["cr_init"]
	exactGrad = parsed_args["exactGrad"]
	telescopic = parsed_args["telescopic"]
	use_gold = parsed_args["use_gold"]
	gold_location = parsed_args["gold_location"]
	γ = parsed_args["gamma"]
	instance_features = parsed_args["instance_features"]
	single_prediction = parsed_args["single_prediction"]
	dataset_location = parsed_args["dataset_location"]
	sample_inside = parsed_args["sample_inside"]
	
	# Auto-disable telescopic if gamma is zero
	telescopic = γ == 0.0 ? false : true

	# Read all instance files from folder
	directory = readdir(folder)

	# Setup optimizer with optional gradient clipping
	if cn > 0
		opt = Flux.OptimiserChain(Flux.Optimise.Adam(lr), ClipNorm(cn))
	else
		opt = Flux.OptimiserChain(Flux.Optimise.Adam(lr))
	end
	
	# Select learning model factory based on sampling strategy
	lt = sample_inside ? BundleNetworks.RnnTModelSampleInsidefactory() : RnnTModelfactory()

	# Initialize random number generator and shuffle instances
	rng = Random.MersenneTwister(seed)
	shuffle!(rng, directory)

	# Define train/validation split
	idxs_train = collect(1:mti)
	idxs_val = collect((mti+1):(mti+mvi))
	
	# Load or generate dataset split
	datasets = Dict()
	if !(dataset_location == "-1")
		# Use pre-defined dataset split
		f = JSON.open(dataset_location, "r")
		datasets = JSON.parse(f)
		close(f)
	else
		# Auto-generate split from shuffled directory
		datasets["training"] = directory[1:(mti)]
		datasets["validation"] = directory[(mti+1):(mti+mvi)]
	end
	
	# Initialize data structures
	dataset = []  # Will store (filename, objective_function) pairs
	gold = Dict()  # Will store optimal solutions
	format = split(directory[1], ".")[end]

	# Load dataset based on file format
	if format == "dat"
		# Handle .dat format: instances and gold solutions separate
		tmp_idx = 0
		for set in [datasets["training"], datasets["validation"]]
			for f in set
				# Read instance
				ins = my_read_dat(folder * f)
				
				# Construct and rescale objective function
				ϕ = BundleNetworks.constructFunction(ins, 1.0)
				_, g = BundleNetworks.value_gradient(ϕ, zeros(sizeK(ins), sizeV(ins)))
				ϕ = BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))
				
				push!(dataset, (f, ϕ))
				tmp_idx += 1
				
				# Periodic garbage collection
				if tmp_idx % 100 == 0
					GC.gc()
				end
			end
		end
		
		# Load gold solutions from separate file
		gold_location = "./golds/" * split(folder, "/")[end-1] * "/gold.json"
		f = JSON.open(gold_location, "r")
		gold = JSON.parse(f)
		close(f)

	else
		# Handle JSON format: instances with embedded gold solutions
		tmp_idx = 0
		for set in [datasets["training"], datasets["validation"]]
			for f in set
				# Read instance and gold solution together
				ins, Ld = my_read_dat_json(folder * f)
				
				# Construct and rescale objective function
				ϕ = BundleNetworks.constructFunction(ins, 1.0)
				_, g = BundleNetworks.value_gradient(ϕ, zeros(sizeK(ins), sizeV(ins)))
				ϕ = BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))
				
				push!(dataset, (f, ϕ))
				gold[f] = Ld
				tmp_idx += 1
				
				if tmp_idx % 100 == 0
					GC.gc()
					# CUDA.reclaim()  # Uncomment for GPU usage
					# CUDA.free_memory()
				end
			end
		end
	end
	
	# Update counts based on actual dataset size (in case of pre-defined split)
	mti, mvi = length(datasets["training"]), length(datasets["validation"])
	idxs_train, idxs_val = collect(1:mti), collect((mti+1):(mti+mvi))
	
	# Create results folder with descriptive name containing all hyperparameters
	res_folder =
		"res_goldLossWeights_" * 
		(instance_features ? "with" : "without") * "InstFeat_init" * 
		(cr_init ? "CR" : "Zero") * "_lr" * string(lr) * "_cn" * string(cn) * 
		"_maxIT" * string(maxIT) * "_maxEP" * string(maxEP) * "_data" *
		string(split(folder, "/")[end-1]) * "_exactGrad" * string(exactGrad) * 
		"_gamma" * string(γ) * "_seed" * string(seed) * 
		"_single_prediction" * string(single_prediction) * 
		"_sampleInside" * string(sample_inside)
	
	# Ensure unique folder name by appending counter
	sN = sum([1 for j in readdir("resLogs") if contains(j, res_folder)]; init = 0.0)
	location = "resLogs/" * res_folder * "_" * string(sN) * "_" * "/"
	mkdir(location)
	
	# Run episodic training and validation
	a = ep_train_and_val(
		folder, directory, dataset, gold, idxs_train, idxs_val, opt; 
		maxIT, maxEP, location, cr_init, exactGrad, telescopic, γ, 
		use_gold, instance_features, seed, single_prediction, lt
	)
end

# Execute main function with command-line arguments
main(ARGS)