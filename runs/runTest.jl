using BundleNetworks
using Instances
using Flux, Zygote
using Plots
using Random
using JSON
using LinearAlgebra
using Statistics
using TensorBoardLogger, Logging
using BSON: @load    # For loading serialized neural network weights (.bson files)
using ArgParse

# Load instance reading utilities (my_read_dat, my_read_dat_json, my_read_ga_json)
include("../runs/readingFunctions.jl")


"""
    main(args)

Entry point for evaluating a pre-trained `BatchedSoftBundle` neural network model
on a set of test instances.

This script loads a previously trained attention-based model from disk, then runs
the `BatchedSoftBundle` optimization loop in inference mode on each test instance
from a pre-defined dataset split. For each instance it records the objective value
history and per-phase timing information, then writes all results to a JSON file.

# Workflow
1. Parse command-line arguments (model path, data path, iteration count, etc.).
2. Load the pre-trained neural network from a `.bson` checkpoint file.
3. Move all network components to GPU and disable stochastic sampling (inference mode).
4. Load the test/train split from a dataset JSON file associated with the model.
5. For each test instance:
   a. Read the problem instance from disk.
   b. Construct and rescale the Lagrangian objective function.
   c. Initialize and reinitialize a `BatchedSoftBundle`.
   d. Run `bundle_execution` in inference mode.
   e. Store objective values and timing data.
6. Write all results to a JSON output file.

# Command-line arguments
- `--folder`: Path to the directory containing problem instance files
  (default: `"/data1/demelas/MCNDsmallCom40/"`).
- `--model_folder`: Subdirectory under `res/` where the trained model checkpoint
  (`nn_best.bson`) and dataset split (`dataset.json`) are stored.
- `--name`: Short identifier appended to the output filename to distinguish runs.
- `--dataset_folder`: Subdirectory under `res/` containing the `dataset.json` split file.
  Defaults to `model_folder` if set to `"-1"`.
- `--iterations`: Number of bundle iterations to run per instance (default: `250`).

# Output
A JSON file named `Results_<name>_<dataset_name>.json` in the current directory,
containing:
- `"times"`: Per-instance dictionary of phase timing vectors from `bundle_execution`.
- `"objs"`: Per-instance vector of rescaled objective values at each bundle iteration.
"""
function main(args)

    # --- Argument parsing ---
    s = ArgParseSettings(
        "Training an unrolling model" *
        "version info, default values, " *
        "options with types, variable " *
        "number of arguments.",
        version     = "Version 1.0",   # Version string printed by --version
        add_version = true             # Automatically add --version flag
    )

    @add_arg_table! s begin
        "--folder"
        arg_type = String
        default  = "/data1/demelas/MCNDsmallCom40/"
        help     = "Path to the directory containing problem instance files."

        "--model_folder"
        arg_type = String
        default  = "BatchVersion_bs_1_true_MCNDsmallCom40_1.0e-6_0.9_5_200_200_1_10_49_true_128_false_softplus_false_0.999_0.0_0.0_sparsemax_false_false_3.0"
        help     = "Subdirectory under res/ containing the trained model checkpoint (nn_best.bson)."

        "--name"
        arg_type = String
        default  = "1"
        help     = "Short identifier appended to the output filename (e.g., run index or experiment tag)."

        "--dataset_folder"
        arg_type = String
        default  = "-1"
        help     = "Subdirectory under res/ containing dataset.json. Defaults to model_folder if '-1'."

        "--iterations"
        arg_type = Int64
        default  = 250
        help     = "Number of bundle iterations to run per instance."
    end

    # Parse command-line arguments into a dictionary
    parsed_args = parse_args(args, s)

    # Unpack parsed arguments into local variables
    folder       = parsed_args["folder"]
    model_folder = parsed_args["model_folder"]
    name         = parsed_args["name"]

    # Use model_folder as dataset_folder if no explicit dataset folder is provided
    dataset_folder = parsed_args["dataset_folder"] == "-1" ?
        parsed_args["model_folder"] :
        parsed_args["dataset_folder"]

    maxIt = parsed_args["iterations"]   # Number of bundle iterations per instance

    # Initialize the results container with two sub-dictionaries:
    # "times": per-instance phase timing data
    # "objs": per-instance objective value history
    res = Dict("times" => Dict(), "objs" => Dict())

    # =========================================================================
    # Model loading and GPU setup
    # =========================================================================

    # Load the best neural network checkpoint saved during training
    @load "res/$(model_folder)/nn_best.bson" nn_best
    global nn = (nn_best)

    # Replace the first encoder layer with an identity pass-through.
    # This is used when the first layer was a problem-specific feature extractor
    # that should be bypassed at inference time (e.g., when features are pre-computed).
    nn.encoder = Chain(x -> identity(x), nn.encoder[2:end]...)

    # Move all network components to GPU for inference
    nn.encoder    = gpu(nn.encoder)
    nn.decoder_t  = gpu(nn.decoder_t)
    nn.decoder_γk = gpu(nn.decoder_γk)
    nn.decoder_γq = gpu(nn.decoder_γq)
    nn = gpu(nn)

    # Disable stochastic sampling in the network (use deterministic inference mode)
    nn.sample_γ = false   # No noise in the γ (DMP weight) decoder
    nn.sample_t = false   # No noise in the t (step size) decoder

    # =========================================================================
    # Dataset split loading
    # =========================================================================

    # Load the dataset JSON which contains the train/test instance filename split
    dataset_path = "res/$(dataset_folder)/dataset.json"
    f       = JSON.open(dataset_path, "r")
    dataset = JSON.parse(f)
    close(f)

    # =========================================================================
    # Instance and bundle setup
    # =========================================================================

    # Factory for the attention-based model (used to create features for the network)
    factory = BundleNetworks.AttentionModelFactory()

    # Discover all files in the instance folder and detect the file format
    directory = readdir(folder)
    format    = split(directory[1], ".")[end]   # "json" or "dat"

    # Use soft stabilization point updates during inference
    # (blends the new trial point and stabilization point via softmax weights)
    soft_updates = true

    # =========================================================================
    # Main inference loop: evaluate on all test instances
    # =========================================================================
    for idx in dataset["test"]
        # --- Load the problem instance ---
        ins = []
        if format == "json"
            # JSON format: returns (instance, gold_bound); discard gold bound at inference
            ins, _ = my_read_dat_json(folder * idx)
        else
            # .dat format: no gold bound in the file
            ins = my_read_dat(folder * idx)
        end

        # --- Construct and rescale the Lagrangian objective function ---
        # Step 1: construct with unit rescaling to compute the gradient at zero
        ϕ    = BundleNetworks.constructFunction(ins, 1.0)
        _, g = BundleNetworks.value_gradient(ϕ, zeros(sizeInputSpace(ϕ)))

        # Step 2: reconstruct with gradient-norm rescaling so ‖g(0)‖ ≈ 1
        # Wrapped in a single-element vector for the BatchedSoftBundle API
        ϕ = [BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))]

        # --- Initialize the BatchedSoftBundle ---
        # maxIt + 1 is used at initialization to pre-allocate one extra column
        # (column 1 = initialization point; columns 2..maxIt+1 = iteration points)
        B = BundleNetworks.initializeBundle(
            BundleNetworks.BatchedSoftBundleFactory(),
            ϕ,
            [zeros(sizeK(ins) * sizeV(ins))],   # Starting point: zero multipliers (K×V dimensional)
            factory,
            maxIt + 1
        )

        # Set the bundle's internal maxIt to match the allocated size
        B.maxIt = maxIt + 1

        # Reset the neural network's hidden state for this instance
        # (required for recurrent/stateful architectures)
        BundleNetworks.reset!(nn, 1, maxIt + 1)

        # Initialize the best objective value tracker
        mv = 0.0

        # Reinitialize the bundle to its starting state, preserving the initialization point
        BundleNetworks.reinitialize_Bundle!(B)

        # Set the actual iteration count for the execution loop
        B.maxIt = maxIt

        # --- Run the bundle execution in inference mode ---
        v, times = BundleNetworks.bundle_execution(
            B, ϕ, nn;
            soft_updates         = soft_updates,
            λ                    = 0.0,          # No weight on the final trial point in the loss
            γ                    = 0.0,          # No telescopic discount term
            δ                    = 0.0,          # No null-step regularization
            distribution_function = BundleNetworks.sparsemax,  # Sparse attention weights
            verbose              = 0,            # Silent output
            inference            = true          # Return (objective, times) instead of training loss
        )

        # --- Store results for this instance ---
        res["times"][idx] = times

        # Collect the rescaled objective history up to the last inserted bundle component
        # (B.obj[1:B.li] covers all visited points; rescaling_factor recovers the original scale)
        res["objs"][idx] = reshape(B.obj[1:B.li], :) .* ϕ[1].rescaling_factor

        # Print the inference objective and the best objective seen across all iterations
        println(v, " ", maximum(reshape(B.obj[1:B.li], :)))
    end

    # =========================================================================
    # Write results to disk
    # =========================================================================
    # Output filename encodes the run name and dataset name for traceability
    output_name = "Results_$(name)_$(split(folder, "/")[end-1]).json"
    f = open(output_name, "w")
    JSON.print(f, cpu(res))   # Move any GPU arrays to CPU before JSON serialization
    close(f)
end


# Entry point: pass command-line arguments to main
main(ARGS)