using BundleNetworks, Instances, Statistics, Flux, LinearAlgebra, JuMP, JSON
using ArgParse
using BSON: @load
include("./readingFunctions.jl")

"""
    test_model(folder, directory, nn, test_idxs, maxIT=100, t=0.000001, lt=BundleNetworks.RnnTModelfactory())

Test a trained neural network model on a set of optimization problem instances using the bundle method.

# Arguments
- `folder::String`: Path to the folder containing test instances
- `directory::Vector{String}`: List of files in the directory (used to detect file format)
- `nn`: Trained neural network model to be tested
- `test_idxs::Vector{String}`: List of test instance filenames to evaluate
- `maxIT::Int=100`: Maximum number of bundle method iterations per instance
- `t::Float64=0.000001`: Initial proximity parameter for the bundle method
- `lt`: Learning model factory (default: RnnTModelfactory)

# Returns
- Results are saved to a JSON file named `res_test2_<dataset_name>.json`

# Description
This function evaluates a trained neural network on optimization test instances by:
1. Loading and preprocessing test instances
2. Running the bundle method with the neural network guidance
3. Computing performance metrics (objective values, times, optimality gaps)
4. Saving results to a JSON file

The function supports two input formats:
- `.dat` files: Instance data with separate gold solutions file
- `.json` files: Instance data with embedded gold solutions
"""
function test_model(
    folder,
    directory,
    nn,
    test_idxs,
    maxIT = 100,
    t = 0.000001,
    lt = BundleNetworks.RnnTModelfactory(),
)
    # Set computation device to CPU
    BundleNetworks.device = cpu
    device = cpu
    
    # Detect file format from the first file in directory
    format = split(directory[1], ".")[end]
    
    # Flag for single vs. batch prediction mode
    single_prediction = false
    
    # Initialize results dictionary to store metrics for each instance
    res = Dict()
    for ins_path in test_idxs
        res[ins_path] = Dict()
    end
    
    # Dictionaries to store dataset instances and gold (optimal) solutions
    dataset = Dict()
    gold = Dict()
    
    println(format)
    
    # Load instances based on file format
    if format == "dat"
        # Handle .dat format: instances and gold solutions are separate
        tmp_idx = 0
        for f in test_idxs
            # Read instance from .dat file
            ins = my_read_dat(folder * f)
            
            # Construct objective function with initial scaling factor of 1.0
            ϕ = BundleNetworks.constructFunction(ins, 1.0)
            
            # Compute gradient at zero to determine rescaling factor
            _, g = BundleNetworks.value_gradient(ϕ, zeros(sizeK(ins), sizeV(ins)))
            
            # Reconstruct function with gradient-based rescaling for numerical stability
            ϕ = BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))
            
            # Store processed instance in dataset
            dataset[f] = (f, ϕ)
            
            tmp_idx += 1
            # Periodic garbage collection to manage memory
            if tmp_idx % 100 == 0
                GC.gc()
                #CUDA.reclaim()  # Uncomment for GPU usage
                #CUDA.free_memory()
            end
        end
        
        # Load gold (optimal) solutions from separate JSON file
        gold_location = "./golds/" * split(folder, "/")[end-1] * "/gold.json"
        f = JSON.open(gold_location, "r")
        gold = JSON.parse(f)
        close(f)
    else
        # Handle .json format: instances include gold solutions
        tmp_idx = 0
        for f in test_idxs
            # Read instance and gold solution together
            ins, Ld = my_read_dat_json(folder * f)
            
            # Construct and rescale objective function (same as above)
            ϕ = BundleNetworks.constructFunction(ins, 1.0)
            _, g = BundleNetworks.value_gradient(ϕ, zeros(sizeK(ins), sizeV(ins)))
            ϕ = BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))
            
            # Store instance and gold solution
            dataset[f] = (f, ϕ)
            gold[f] = Ld
            
            tmp_idx += 1
            # Periodic garbage collection
            if tmp_idx % 100 == 0
                GC.gc()
                # CUDA.reclaim()  # Uncomment for GPU usage
                # CUDA.free_memory()
            end
        end
    end
    
    # Initialize cutting-plane relaxation (CR) parameters
    cr_init = false  # Flag to use CR initialization or simple defaults
    crs = Dict()      # CR scaling factors
    cr_duals = Dict() # CR dual solutions
    
    # Compute or initialize CR parameters for each instance
    for i in keys(dataset)
        idx, ϕ = dataset[i]
        inst = ϕ.inst
        
        if !(cr_init)
            # Simple initialization: zero duals and unit scaling
            cr_duals[idx] = zeros((sizeK(inst), sizeV(inst)))
            crs[idx] = 1.0
        else
            # Use cutting-plane relaxation for warm-start
            crs[idx], cr_duals[idx] = CR(inst)[1:2]
        end
        GC.gc()
    end
    
    # Main testing loop: solve each instance with the bundle method
    for idx_i in test_idxs
        ins_path, ϕ = dataset[idx_i]
        
        # Initialize bundle method solver with:
        # - The objective function ϕ
        # - Proximity parameter t
        # - Initial dual solution from CR
        # - The neural network model (deep copied)
        # - Maximum iterations
        B = BundleNetworks.initializeBundle(
            BundleNetworks.tLearningBundleFactory(), 
            ϕ, 
            t, 
            cr_duals[ins_path], 
            lt, 
            deepcopy(nn), 
            maxIT + 1; 
            exactGrad = true,  # Use exact gradients
            instance_features = true  # Include instance-specific features
        )
        
        # Set maximum iterations
        B.params.maxIt = maxIT
        
        # Record start time
        t0 = time()
        
        # Solve the optimization problem using bundle method with NN guidance
        # t_strat: neural network-based proximity parameter strategy
        # unstable=false: use stable solving mode
        co, timesD = solve!(B, ϕ; t_strat = BundleNetworks.nn_t_strategy(), unstable = false)
        
        println()
        
        # Store results for this instance
        f = ins_path
        res[f]["time"] = time() - t0  # Total solving time
        res[f]["objs"] = B.all_objs[1:end] * ϕ.rescaling_factor  # Objective values (rescaled)
        res[f]["times"] = timesD  # Iteration timestamps
        res[f]["gaps"] = [gap(i, gold[idx_i]) for i in res[f]["objs"]]  # Optimality gaps
        
        # Garbage collection after each instance
        GC.gc()
    end
    
    # Save all results to JSON file
    f = open("res_test2_$(split(folder,"/")[end-1]).json", "w")
    JSON.print(f, res)
    close(f)
end

"""
    main(args)

Main entry point for the testing script. Parses command-line arguments and executes the testing pipeline.

# Arguments
- `args::Vector{String}`: Command-line arguments (typically ARGS)

# Command-line Arguments
- `--data`: Path to the folder containing test instances (required)
- `--model`: Path to the trained model directory (required)
- `--dataset`: Path to the dataset split definition file (required)

# Description
This function:
1. Parses command-line arguments
2. Loads the trained neural network model
3. Loads the test set definition
4. Calls test_model to evaluate the model on test instances
"""
function main(args)
    # Configure argument parser with description
    s = ArgParseSettings(
        "Training an unrolling model" *
        "version info, default values, " *
        "options with types, variable " *
        "number of arguments.",
        version = "Version 1.0",  # Version information
        add_version = true        # Auto-add --version option
    )
    
    # Define command-line arguments
    @add_arg_table! s begin
        "--data"
            arg_type = String
            required = true
            help = "path to the instance folder"
        "--model"
            arg_type = String
            required = true
            help = "path to the model"
        "--dataset"
            arg_type = String
            required = true
            help = "path to the dataset file fixing train, validation and test sets"
    end
    
    # Parse arguments
    parsed_args = parse_args(args, s)
    folder = parsed_args["data"]
    model_path = parsed_args["model"]
    dataset_path = parsed_args["dataset"]
    
    # Get list of files in the instance folder
    directory = readdir(folder)
    
    # Load the best trained neural network model from BSON file
    @load "$(model_path)/nn_bestLV.bson" nn_best
    global nn = (nn_best)
    
    # Disable sampling mode (use deterministic predictions)
    nn.sample = false
    
    # Load test set definition from JSON file
    f = open(dataset_path * "dataset.json", "r")
    dataset = JSON.parse(f)["test"]
    close(f)
    
    # Run testing pipeline
    a = test_model(folder, directory, nn, dataset)
end

# Execute main function with command-line arguments
main(ARGS)