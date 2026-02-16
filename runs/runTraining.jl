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

"""
    gap(a, b)

Calculate the relative percentage gap between two values.

# Arguments
- `a::Float64`: First value
- `b::Float64`: Second value (typically the optimal/target value)

# Returns
- `Float64`: Percentage gap calculated as |a - b| / max(a, b) * 100
"""
gap(a, b) = abs(a - b) / max(a, b) * 100

using CUDA
using BSON: @save
using ArgParse
using Logging
using TensorBoardLogger
using MLUtils
using ParameterSchedulers
using ChainRules, ChainRulesCore

"""
    main(args)

Main training function for the neural network-guided bundle method.

# Arguments
- `args::Vector{String}`: Command-line arguments (typically ARGS)

# Description
This function implements a complete training pipeline for learning to optimize using
neural network-guided bundle methods. The process includes:

1. **Configuration**: Parse command-line arguments for all hyperparameters
2. **Data Loading**: Load and preprocess optimization problem instances
3. **Model Initialization**: Create neural network architecture
4. **Training Loop**: Iteratively train the model using unrolled optimization
5. **Validation**: Evaluate on validation set and track best model
6. **Logging**: Save results, metrics, and trained models

# Training Strategy
- Uses unrolled differentiation through bundle method iterations
- Supports incremental curriculum learning (gradually increasing iterations)
- Implements batch processing with configurable batch sizes
- Features learning rate scheduling with exponential decay
- Tracks multiple metrics: loss, objective values, optimality gaps

# Command-line Arguments (Required)
- `--lr`: Learning rate for optimizer
- `--mti`: Maximum number of training instances
- `--mvi`: Maximum number of validation instances
- `--seed`: Random seed for reproducibility
- `--maxItBack`: Maximum unrolled iterations for backward pass
- `--maxEP`: Maximum number of training epochs

# Command-line Arguments (Optional, with defaults)
- `--data`: Path to instance folder (default: "./data/MCNDforTest/")
- `--decay`: Learning rate decay factor (default: 0.9)
- `--lambda`: Weight for final trial point objective (default: 0.0)
- `--gamma`: Weight for telescopic sum of objectives (default: 0.0)
- `--delta`: Additional regularization parameter (default: 0.0)
- `--cn`: Gradient clipping norm (default: 5)
- `--maxIt`: Maximum bundle iterations during training (default: -1, uses maxItBack)
- `--maxItVal`: Maximum iterations for validation (default: 100)
- `--soft_updates`: Use soft updates for stabilization (default: true)
- `--h_representation`: Hidden layer size (default: 64)
- `--use_softmax`: Use softmax vs sparsemax (default: true)
- `--use_graph`: Use bipartite graph features (default: true)
- `--batch_size`: Training batch size (default: 1)
- `--incremental`: Enable curriculum learning (default: false)
- `--always_batch`: Force batch mode even for batch_size=1 (default: false)
- `--h_act`: Activation function: softplus/tanh/gelu/relu (default: "softplus")
- `--sampling_gamma`: Sample in latent space for attention (default: false)
- `--sampling_t`: Sample proximity parameter (default: true)
- `--reduced_components`: Skip duplicate subgradients (default: false)
- `--scheduling_ss`: Learning rate schedule step size (default: 100)
- `--h3`: Use single hidden representation (default: false)
- `--use_tanh`: Apply tanh activation (default: false)

# Output
Creates a results folder containing:
- Trained neural network models (nn.bson, nn_best.bson)
- Training metrics (loss.json, obj.json, gaps.json)
- Validation metrics (obj_val.json, gaps_val.json, val_times.json)
- Dataset split definition (dataset.json)
- TensorBoard logs for visualization
"""
function main(args)
	# Configure argument parser with comprehensive help text
	s = ArgParseSettings(
		"Training an unrolling model" *
		"version info, default values, " *
		"options with types, variable " *
		"number of arguments.",
		version = "Version 1.0",  # Version information
		add_version = true        # Auto-add --version option
	)

	# Define all command-line arguments with types, defaults, and help text
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
			help = "learning rate decay factor applied at scheduled intervals"
		"--lambda"
			default = 0.0
			arg_type = Float32
			help = "the final loss will be lambda*(phi(last_trial_point))+(1-lambda)*(phi(last_stabilization_trial_point)) + ..."
		"--gamma"
			default = 0.0
			arg_type = Float32
			help = "parameter that controls the weights in the telescopic sum of all objective values of trial points found during execution"
		"--delta"
			default = 0.0
			arg_type = Float32
			help = "additional loss regularization parameter"
		"--cn"
			default = 5
			arg_type = Int64
			help = "Clip Norm for gradient clipping"
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
			help = "Random seed for reproducibility"
		"--maxItBack"
			required = true
			arg_type = Int64
			help = "Maximum number of unrolled iterations for backward pass (gradient computation)"
		"--maxIt"
			default = -1
			arg_type = Int64
			help = "Maximum number of unrolled iterations during forward pass. If -1, uses maxItBack value"
		"--maxItVal"
			default = 100
			arg_type = Int64
			help = "Maximum number of unrolled iterations for validation evaluation"
		"--maxEP"
			required = true
			arg_type = Int64
			help = "Maximum number of training epochs"
		"--soft_updates"
			arg_type = Bool
			default = true
			help = "If true, use soft updates (with softmax) for selecting the stabilization point"
		"--h_representation"
			arg_type = Int64
			default = 64
			help = "Size of the hidden representation layer for the NN model"
		"--use_softmax"
			arg_type = Bool
			default = true
			help = "If true, use softmax to compute distribution for convex combination of gradients; otherwise use sparsemax"
		"--use_graph"
			arg_type = Bool
			default = true
			help = "If true, use bipartite graph representation to compute additional instance features"
		"--batch_size"
			arg_type = Int64
			default = 1
			help = "Batch size for training. Default is 1 (single instance)"
		"--incremental"
			arg_type = Bool
			default = false
			help = "If true, enable curriculum learning: start with fewer iterations and gradually increase to maxIt over first maxEP/2 epochs"
		"--always_batch"
			arg_type = Bool
			default = false
			help = "If true, use batch processing implementation even when batch_size=1"
		"--h_act"
			arg_type = String
			default = "softplus"
			help = "Activation function to use in the model: 'softplus', 'tanh', 'gelu', or 'relu'"
		"--sampling_gamma"
			arg_type = Bool
			default = false
			help = "If true, sample in the latent space to predict keys and queries for attention mechanism"
		"--sampling_t"
			arg_type = Bool
			default = true
			help = "If true, sample the proximity parameter t from a distribution"
		"--reduced_components"
			arg_type = Bool
			default = false
			help = "If true, only add bundle component if subgradient is new; otherwise update existing component values"
		"--scheduling_ss"
			arg_type = Int64
			default = 100
			help = "Step size (in epochs) for learning rate scheduling"
		"--h3"
			arg_type = Bool
			default = false
			help = "If true, use only one hidden representation instead of 3 separate representations"
		"--use_tanh"
			arg_type = Bool
			default = false
			help = "If true, apply tanh activation in specific model components"
	end

	# Parse command-line arguments into a dictionary
	parsed_args = parse_args(args, s)
	
	# Extract all hyperparameters from parsed arguments
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
	maxBaIt = maxBaIt < 0 ? maxIt : maxBaIt  # Use maxIt if maxBaIt not specified
	bgr = parsed_args["use_graph"]
	batch_size = parsed_args["batch_size"]
	incremental = parsed_args["incremental"]
	a_b = parsed_args["always_batch"]
	scheduling_ss = parsed_args["scheduling_ss"]
	
	# Map activation function string to actual function
	h_act = (parsed_args["h_act"] == "softplus") ? softplus : 
	        (parsed_args["h_act"] == "tanh" ? tanh : 
	        (parsed_args["h_act"] == "gelu" ? gelu : relu))
	
	sampling_θ = parsed_args["sampling_gamma"]
	sampling_t = parsed_args["sampling_t"]
	
	# Special activation for GA (Graph Assignment) problems
	act = contains(folder, "GA") ? relu : identity
	h3 = parsed_args["h3"]
	use_tanh = parsed_args["use_tanh"]

	reduced_components = parsed_args["reduced_components"]
	
	# Select distribution function for gradient combination
	distribution_function = use_softmax ? softmax : (BundleNetworks.sparsemax)
	
	# Initialize random number generator with specified seed
	rng = Random.MersenneTwister(seed)
	directory = shuffle(rng, readdir(folder))
	
	# Detect file format from first file extension
	format = split(directory[1], ".")[end]

	# Create neural network model with specified architecture
	factory = bgr ? BundleNetworks.AttentionModelFactory() : BundleNetworks.AttentionModelFactory()
	global nn = BundleNetworks.create_NN(
		factory; 
		h_representation, 
		h_act, 
		sampling_θ, 
		sampling_t, 
		h3_representations = h3, 
		use_tanh
	)
	nn.h_representation = BundleNetworks.h_representation(nn)
	BundleNetworks.reset!(nn, batch_size)
	
	# Set compute device (GPU if available and desired)
	use_gpu = true
	device = CUDA.functional() && use_gpu ? gpu : cpu
	
	# Initialize data structures
	dataset = []  # Will store (filename, objective_function) pairs
	gold = Dict()  # Will store optimal solutions for each instance
	
	println(format)
	
	# Load dataset based on file format
	if format == "dat"
		# Handle .dat format: read instances and load separate gold solutions
		tmp_idx = 0
		for f in directory[1:(mti+mvi)]
			# Read instance from .dat file
			ins = my_read_dat(folder * f)
			
			# Construct objective function with initial scaling
			ϕ = BundleNetworks.constructFunction(ins, 1.0)
			
			# Compute gradient at zero to determine rescaling factor
			_, g = BundleNetworks.value_gradient(ϕ, zeros(sizeLM(ins)))
			
			# Reconstruct with gradient-based rescaling for numerical stability
			ϕ = BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))
			
			push!(dataset, (f, ϕ))
			tmp_idx += 1
			
			# Periodic memory management
			if tmp_idx % 100 == 0
				GC.gc()
				CUDA.reclaim()
				CUDA.free_memory()
			end
		end

		# Load gold solutions from separate JSON file
		f = JSON.open("./golds/" * split(folder, "/")[end-1] * "/gold.json", "r")
		gold = JSON.parse(f)
		close(f)

	else
		# Handle JSON format instances
		if contains(directory[1], "uc")
			# Special handling for Unit Commitment (UC) problems
			tmp_idx = 0
			f = JSON.open("./golds/" * split(folder, "/")[end-1] * "/gold.json", "r")
			gold = JSON.parse(f)
			close(f)
			
			for f in directory[1:(mti+mvi)]
				ins = Instances.read_nUC(folder * f)
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
			# Standard JSON format handling
			tmp_idx = 0
			for f in directory[1:(mti+mvi)]
				# Read instance and gold solution together
				# Use different reader for GA (Graph Assignment) problems
				ins, Ld = contains(folder, "GA") ? my_read_ga_json(folder * f) : my_read_dat_json(folder * f)
				
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

	# Setup optimizer: Adam with gradient clipping
	opt = Flux.OptimiserChain(Flux.Optimise.Adam(lr), ClipNorm(cn))
	opt_st = Flux.setup(opt, nn)
	
	# Setup learning rate scheduler with exponential decay
	scheduler = ParameterSchedulers.Stateful(Exp(start = lr, decay = decay))

	# Split dataset into training, validation, and test sets
	idxs = collect(1:mti)  # Training indices
	idxs_v = collect((mti+1):(mti+mvi))  # Validation indices
	idxs_test = collect((mti+mvi+1):length(directory))  # Test indices
	
	# Initialize best model tracker
	global nn_best = deepcopy(nn)
	
	# Create validation model (deterministic, no sampling)
	global nn_val = deepcopy(nn)
	nn_val.sample_t = false
	nn_val.sample_γ = false
	
	# Initialize metric tracking arrays
	results = Dict("training" => Dict(), "validation" => Dict(), "test" => Dict())
	v, losses, gaps = [], [], []  # Training metrics
	v_v, losses_v, gaps_v = [], [], []  # Validation metrics
	v_v_ti, losses_v_ti, gaps_v_ti = [], [], []  # Validation metrics at maxIt iterations
	times_val = []  # Validation computation times
	cum_grads = []  # Cumulative gradients (for analysis)
	
	# Early stopping variables
	ciwi = 0  # Consecutive iterations without improvement
	last_loss = Inf
	last_train_loss = Inf
	threshold = 1.0e-7  # Threshold for loss improvement

	# Create results folder with comprehensive naming scheme
	# Includes all hyperparameters for experiment tracking
	res_folder =
		"BatchVersion_bs_" * string(batch_size) * "_seed" * string(seed) * "_" * 
		string(a_b) * "_" * string(split(folder, "/")[end-1]) * "_" * 
		string(lr) * "_" * string(decay) * "_" * string(cn) * "_" * 
		string(mti) * "_" * string(mvi) * "_" * string(seed) * "_" * 
		string(maxIt) * "_" * string(maxEp) * "_" * string(soft_updates) * "_" *
		string(h_representation) * "_" * string(sampling_θ) * string(sampling_t) * "_" * 
		string(h_act) * "_" * string(use_softmax) * "_" * string(gamma) * "_" * 
		string(lambda) * "_" * string(delta) * "_" * string(distribution_function) * "_" * 
		string(bgr) * "_" * string(incremental) * "_rc" * string(reduced_components) * 
		"_ss" * string(scheduling_ss) * "_h3" * string(h3) * "_usetanh" * string(use_tanh)
	
	# Ensure unique folder name by appending counter
	sN = sum([1 for j in readdir("resLogs") if contains(j, res_folder)]; init = 0.0)
	res_folder = "resLogs/" * res_folder * "_" * string(sN + 1)
	mkdir(res_folder)
	
	# Save dataset split for reproducibility
	f = JSON.open(res_folder * "/dataset.json", "w")
	JSON.print(f, Dict(
		"training" => directory[idxs], 
		"validation" => directory[idxs_v], 
		"test" => directory[idxs_test]
	))
	close(f)

	# Prepare batched datasets
	batched_trainset = dataset[idxs]
	batched_valset = dataset[idxs_v]
	
	# Additional early stopping tracking
	iterations_without_improvement = 0
	last_loss = Inf
	
	# Initialize TensorBoard logger
	lg = TBLogger(res_folder, min_level = Logging.Info)
	
	# Main training loop
	with_logger(lg) do
		for it in 1:maxEp
			# Initialize per-epoch metric accumulators
			ls, ls_v, ls_v_tI = [], [], []  # Losses
			gs, gs_v, gs_v_tI = [], [], []  # Gaps
			vs, vs_v, vs_v_tI = [], [], []  # Objective values
			time_val = 0.0
			
			# Shuffle training data each epoch
			shuffle!(rng, batched_trainset)
			
			# Training phase: iterate through mini-batches
			first = 1
			last = batch_size
			for it_idx in 1:ceil(mti / batch_size)
				# Extract current batch
				sample = batched_trainset[first:last]
				
				# Handle batch vs single instance mode
				idx = batch_size > 1 || a_b ? [s[1] for s in sample] : sample[1][1]
				ϕ = batch_size > 1 || a_b ? [s[2] for s in sample] : sample[1][2]
				
				# Initialize starting point (zero vector)
				z = batch_size == 1 && !a_b ? 
					zeros(prod(sizeInputSpace(ϕ))) : 
					[zeros(prod(sizeInputSpace(ϕi))) for ϕi in ϕ]
				
				# Select appropriate bundle factory
				bt = batch_size == 1 && !a_b ? 
					SoftBundleFactory() : 
					BatchedSoftBundleFactory()
				
				# Initialize bundle method solver
				B = BundleNetworks.initializeBundle(
					bt, ϕ, z, factory, maxIt + 1, reduced_components
				)
				
				# Set number of iterations (with curriculum learning support)
				B.maxIt = incremental ? min(2 * it * maxIt / maxEp, maxIt) : maxIt

				# Reset neural network state for new batch
				BundleNetworks.reset!(nn, max(1, last - first + 1))
				BundleNetworks.reinitialize_Bundle!(B)
				
				mv = 0.0
				B.maxIt = incremental ? min(2 * it * maxIt / maxEp, maxIt) : maxIt
				
				# Compute normalization factor based on number of subproblems
				r_f = (batch_size > 1 || a_b ? 
					sum(BundleNetworks.numberSP(f) for f in ϕ) : 
					BundleNetworks.numberSP(ϕ)) * maxIt / (last - first + 1)
				
				# Forward and backward pass: compute loss and gradients
				# Loss is negated because we maximize objective but minimize loss
				vv, grads = Flux.withgradient((m) -> 
					.-BundleNetworks.bundle_execution(
						B, ϕ, m; 
						soft_updates = soft_updates, 
						λ = lambda, 
						γ = gamma, 
						δ = delta, 
						distribution_function, 
						verbose = 0, 
						inference = false, 
						act
					) / r_f, nn)
				vv = -vv  # Convert back to maximization objective

				# Update neural network parameters using optimizer
				_, nn = Flux.Optimisers.update!(opt_st, nn, grads[1])

				mv += vv / round(maxIt / maxBaIt)
				
				# Record metrics for this batch
				if batch_size == 1 && !a_b
					append!(ls, mv)
					append!(vs, maximum(B.obj) * ϕ.rescaling_factor)
					append!(gs, BundleNetworks.sign(ϕ) * gap(maximum(B.obj) * ϕ.rescaling_factor, gold[idx]))
				else
					append!(ls, mv)
					append!(vs, mean(maximum(B.obj[j, :]) * ϕ[j].rescaling_factor for j in eachindex(idx)))
					append!(gs, mean([BundleNetworks.sign(ϕ[j]) * gap(maximum(B.obj[j, :]) * ϕ[j].rescaling_factor, gold[idx[j]]) for j in eachindex(idx)]))
				end
				
				# Move to next batch
				first += batch_size
				last += batch_size
				last = min(last, mti)
			end

			# Memory cleanup after training epoch
			GC.gc()
			CUDA.reclaim()
			CUDA.free_memory()

			# Log training metrics to TensorBoard
			@info "Train" GAP_percentage = mean(gs) log_step_increment = 0
			@info "Train" LSP_value = mean(vs) log_step_increment = 0
			@info "Train" Loss_value = mean(ls) log_step_increment = 0

			# Store epoch-level training metrics
			append!(v, mean(vs))
			append!(losses, mean(ls))
			append!(gaps, mean(gs))

			# Learning rate scheduling: decay every scheduling_ss epochs
			if it % scheduling_ss == 0
				Flux.adjust!(opt_st, ParameterSchedulers.next!(scheduler))
			end
			
			# Update validation model with current training weights
			nn_val.encoder = nn.encoder
			nn_val.decoder_t = nn.decoder_t
			nn_val.decoder_γk = nn.decoder_γk
			nn_val.decoder_γq = nn.decoder_γq

			# Validation phase: evaluate on validation set
			first = 1
			last = batch_size
			for it_idx in 1:ceil(mvi / batch_size)
				sample = batched_valset[first:last]
				idx = batch_size > 1 || a_b ? [s[1] for s in sample] : sample[1][1]
				ϕ = batch_size > 1 || a_b ? [s[2] for s in sample] : sample[1][2]
				z = batch_size == 1 && !a_b ? 
					zeros(prod(sizeInputSpace(ϕ))) : 
					[zeros(prod(sizeInputSpace(ϕi))) for ϕi in ϕ]
				bt = batch_size == 1 && !a_b ? 
					SoftBundleFactory() : 
					BatchedSoftBundleFactory()
				
				# Initialize bundle with potentially more iterations for validation
				B = BundleNetworks.initializeBundle(
					bt, ϕ, z, factory, max(maxIt, maxItVal) + 1, reduced_components
				)
				B.maxIt = max(maxIt, maxItVal)
				BundleNetworks.reset!(nn_val, max(1, last - first + 1))
				BundleNetworks.reinitialize_Bundle!(B)
				
				# Time the validation run
				t0 = time()
				r_f = (batch_size > 1 || a_b ? 
					sum(BundleNetworks.numberSP(f) for f in ϕ) : 
					BundleNetworks.numberSP(ϕ)) * maxIt / (last - first + 1)
				
				# Run validation (no gradient computation)
				val = BundleNetworks.bundle_execution(
					B, ϕ, nn_val; 
					soft_updates = soft_updates, 
					λ = lambda, 
					γ = gamma, 
					δ = delta, 
					distribution_function, 
					verbose = 0, 
					inference = false, 
					act
				) / r_f
				
				time_val = time() - t0
				append!(ls_v, val)

				# Record validation metrics
				if batch_size == 1 && !a_b
					append!(vs_v, maximum(B.obj) * ϕ.rescaling_factor)
					append!(gs_v, BundleNetworks.sign(ϕ) * gap(maximum(B.obj) * ϕ.rescaling_factor, gold[idx]))
					# Also track metrics at maxIt iterations specifically
					append!(vs_v_tI, maximum(B.obj[1:min(maxIt, maxItVal)]))
					append!(gs_v_tI, sign(ϕ) * gap(maximum(B.obj[1:min(maxIt, maxItVal)]) * ϕ.rescaling_factor, gold[idx]))
				else
					append!(vs_v, mean(maximum(B.obj[j, :]) * ϕ[j].rescaling_factor for j in eachindex(idx)))
					append!(gs_v, mean([BundleNetworks.sign(ϕ[j]) * gap(maximum(B.obj[j, :]) * ϕ[j].rescaling_factor, gold[idx[j]]) for j in eachindex(idx)]))
					append!(vs_v_tI, mean(maximum(B.obj[j, 1:min(maxIt, maxItVal)]) * ϕ[j].rescaling_factor for j in eachindex(idx)))
					append!(gs_v_tI, mean([BundleNetworks.sign(ϕ[j]) * gap(maximum(B.obj[j, 1:min(maxIt, maxItVal)]) * ϕ[j].rescaling_factor, gold[idx[j]]) for j in eachindex(idx)]))
					first += batch_size
					last += batch_size
					last = min(last, mvi)
				end
			end
			
			# Track best model based on validation gap at maxIt iterations
			if mean(gs_v_tI) > minimum(gaps_v; init = Inf)
				nn_best = deepcopy(nn)
			end
			
			# Store validation metrics
			append!(v_v, mean(vs_v))
			append!(losses_v, mean(ls_v))
			append!(gaps_v, mean(gs_v))
			append!(v_v, mean(vs_v_tI))
			append!(gaps_v, mean(gs_v_tI))
			append!(times_val, time_val)

			# Early stopping check: count epochs without improvement
			if abs(last_loss - losses_v[end]) < threshold
				ciwi += 1
			else
				ciwi = 0
			end

			last_loss = losses_v[end]
			if ciwi > 3
				# Uncomment to enable early stopping after 3 epochs without improvement
				#break
			end

			# Log validation metrics to TensorBoard
			@info "Validation" GAP_percentage = mean(gs_v) log_step_increment = 0
			@info "Validation" GAP_percentage_li = mean(gs_v_tI) log_step_increment = 0
			@info "Validation" LSP_value = mean(vs_v) log_step_increment = 0
			@info "Validation" Loss_value = mean(ls_v) log_step_increment = 1

			# Print epoch summary to console
			println(it, " Training - lsp: ", v[end], "  gap: ", gaps[end])
			println(it, " Validation - lsp: ", v_v[end], "  gap: ", gaps_v[end])

			# Memory cleanup after validation
			GC.gc()
			CUDA.reclaim()
			CUDA.free_memory()
		end
	end
	
	# Move models to CPU for saving
	nn = cpu(nn)
	nn_best = cpu(nn_best)

	# Save trained models
	@save res_folder * "/nn.bson" nn
	@save res_folder * "/nn_best.bson" nn_best

	# Save all training metrics to JSON files
	f = open(res_folder * "/loss.json", "w")
	JSON.print(f, losses)
	close(f)

	f = open(res_folder * "/obj.json", "w")
	JSON.print(f, v)
	close(f)

	f = open(res_folder * "/gaps.json", "w")
	JSON.print(f, gaps)
	close(f)

	# Save validation metrics (both maxIt and maxItVal)
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

# Execute main function with command-line arguments
main(ARGS)