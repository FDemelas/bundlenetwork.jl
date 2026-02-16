using BundleNetworks, Instances, Statistics, Flux, Zygote, LinearAlgebra, JSON
import BundleNetworks: sizeLM, gap
using ArgParse

# Load instance reading utilities (my_read_dat, my_read_dat_json, my_read_ga_json)
include("../runs/readingFunctions.jl")


"""
    main(args)

Entry point for the benchmark script comparing multiple optimization methods
(gradient descent variants and bundle method variants) on a set of problem instances.

For each combination of instance, initial step size `t`, and method, the script:
1. Reads the problem instance from disk (JSON or `.dat` format).
2. Constructs the associated Lagrangian objective function.
3. Runs the selected optimization method for the configured number of iterations.
4. Records the objective value history and timing information.
5. Writes all results to a JSON output file.

# Supported methods
- **Descent**: Vanilla gradient descent with step size `t` (Flux optimizer).
- **Adam**: Adam optimizer with learning rate `t` (Flux optimizer).
- **Bundle constant**: Bundle method with a constant t-strategy (no regularization update).
- **Bundle soft**: Bundle method with a soft long-term t-strategy.
- **Bundle hard**: Bundle method with a hard long-term t-strategy.
- **Bundle balancing**: Bundle method with a balancing long-term t-strategy.

# Command-line arguments
- `--folder`: Path to the directory containing the problem instance files
  (default: `"./data/MCNDforTest/"`).
- `--maxIterBundle`: Maximum number of iterations for bundle method variants
  (default: `100`).
- `--maxIterDescentType`: Maximum number of iterations for gradient descent variants
  (default: `1000`).
- `--TS`: Space-separated list of step-size / regularization-parameter values to test.

# Output
A JSON file named `Results_<dataset_name>.json` in the current directory, containing
a nested dictionary: `method → t → instance_filename → {obj, time, times}`.
"""
function main(args)

    # --- Argument parsing ---
    s = ArgParseSettings(
        "Training an unrolling model" *
        "version info, default values, " *
        "options with types, variable " *
        "number of arguments.",
        version    = "Version 1.0",  # Version string printed by --version
        add_version = true           # Automatically add --version flag
    )

    @add_arg_table! s begin
        "--folder"
        arg_type = String
        default  = "./data/MCNDforTest/"
        help     = "Path to the directory containing problem instance files."

        "--maxIterBundle"
        arg_type = Int64
        default  = 100
        help     = "Maximum number of iterations for bundle method variants."

        "--maxIterDescentType"
        arg_type = Int64
        default  = 1000
        help     = "Maximum number of iterations for gradient descent variants (Adam, Descent)."

        "--TS"
        arg_type = Float64
        nargs    = '*'
        help     = "Space-separated list of step-size / regularization-parameter values to benchmark."
    end

    parsed_args = parse_args(args, s)

    # Unpack parsed arguments
    TS     = parsed_args["TS"]       # e.g. [10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1]
    folder = parsed_args["folder"]   # Path to instance directory
    mI     = parsed_args["maxIterBundle"]      # Max iterations for bundle methods
    mI_DT  = parsed_args["maxIterDescentType"] # Max iterations for descent methods

    # --- Instance discovery ---
    directory = readdir(folder)   # List all files in the instance folder

    # Detect the file format from the first file's extension (e.g., "json" or "dat")
    format = split(directory[1], ".")[end]

    # For Generalized Assignment (GA) instances, the dual variables must be non-negative
    # (relu projection); for MCND instances, no projection is needed (identity)
    act = contains(folder, "GA") ? relu : identity

    # Batch index list (currently unused for batching, kept for compatibility)
    batches = collect(1:length(directory))

    # --- Method definitions ---
    # Each entry is (display_name, optimizer_type_or_t_strategy_instance)
    # Descent and Adam entries use Flux optimizer types (constructed later with step size t)
    # Bundle entries use pre-constructed t-strategy instances
    opts = [
        ("Descent",           Descent),
        ("Adam",              Adam),
        ("Bundle constant",   BundleNetworks.constant_t_strategy()),
        ("Bundle soft",       BundleNetworks.soft_long_term_t_strategy(BundleNetworks.heuristic_t_strategy_1())),
        ("Bundle hard",       BundleNetworks.hard_long_term_t_strategy(BundleNetworks.heuristic_t_strategy_1())),
        ("Bundle balancing",  BundleNetworks.balancing_long_term_t_strategy(BundleNetworks.heuristic_t_strategy_1())),
    ]

    # --- Results container ---
    # Nested dictionary: method_name → t_value → instance_filename → results_dict
    res = Dict(
        m[1] => Dict(
            t => Dict(
                i => Dict()
                for i in directory
            )
            for t in TS
        )
        for m in opts
    )

    # --- Load gold standard (reference) objective values ---
    # Used to compute the optimality gap for .dat format instances
    f     = JSON.open("./golds/MCNDforTest/gold.json", "r")
    golds = JSON.parse(f)
    close(f)

    # --- Force CPU execution for this benchmark ---
    # GPU is disabled here to ensure reproducible timing comparisons
    BundleNetworks.device = cpu
    device = cpu

    # =========================================================================
    # Main benchmark loop: iterate over all instances, step sizes, and methods
    # =========================================================================
    for i in 1:length(directory)
        for t in TS
            for method in opts

                v_b = []   # Objective value history for this (instance, t, method) triple
                t0  = time()

                # --- Load the problem instance ---
                ins, gold = [], []
                if format == "json"
                    # JSON format: load instance and gold bound together
                    ins, gold = contains(folder, "GA") ?
                        my_read_ga_json(folder * directory[i]) :      # GA instance
                        my_read_dat_json(folder * directory[i])       # MCND instance
                else
                    # .dat format: load instance from file, gold bound from the gold JSON
                    ins  = my_read_dat(folder * directory[i])
                    gold = golds[directory[i]]
                end

                tInit = time()   # Start timing after instance loading

                if !(contains(method[1], "Bundle"))
                    # =============================================================
                    # Gradient Descent / Adam branch
                    # =============================================================

                    # Construct the Lagrangian with unit rescaling to get the initial gradient
                    opt = method[2](t)
                    ϕ   = BundleNetworks.constructFunction(ins, 1.0)

                    # Evaluate the gradient at zero to determine the rescaling factor
                    # (normalizes the problem so that ‖g(0)‖ ≈ 1)
                    v, g = BundleNetworks.value_gradient(ϕ, zeros(sizeInputSpace(ϕ)))

                    # Reconstruct the function with the gradient-norm rescaling factor
                    ϕ = BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))

                    # Initialize the dual variable vector at zero
                    z    = zeros(BundleNetworks.numberSP(ϕ))
                    bz   = copy(z)      # Best point found so far
                    best_ϕ  = -Inf      # Best objective value found so far
                    l_ep    = 0         # Iteration index of the best point
                    ss      = t         # Current step size (may be reduced on stagnation)

                    timesD = Dict("lsp" => [], "update" => [], "times" => [])
                    state  = Flux.setup(opt, z)   # Initialize Flux optimizer state

                    for ep in 1:mI_DT
                        t0 = time()

                        # Oracle call: evaluate objective and subgradient at current z
                        vi, ∂L = value_gradient(ϕ, z)
                        append!(timesD["lsp"], time() - t0)

                        t1 = time()

                        # Gradient ascent step: move in the direction of the subgradient
                        # (Flux.update! expects a descent direction, so negate ∂L)
                        Flux.update!(state, z, -∂L)

                        # Apply the activation projection (relu for GA, identity for MCND)
                        z = act(z)

                        # Record the rescaled objective value for this iteration
                        append!(v_b, vi .* ϕ.rescaling_factor)

                        # Track the best point found so far (for gap reporting)
                        if vi > best_ϕ
                            bz      = z
                            best_ϕ  = vi
                            l_ep    = ep
                        end

                        # Step size decay: halve the step size if no improvement
                        # for more than 2 consecutive iterations
                        ss = l_ep - ep > 2 ? ss / 2 : ss
                        opt = method[2](ss)   # Rebuild optimizer with updated step size

                        append!(timesD["update"], time() - t1)
                        append!(timesD["times"],  time() - t0)
                    end

                    res[method[1]][t][directory[i]]["times"] = timesD

                else
                    # =============================================================
                    # Bundle method branch (Vanilla Bundle with heuristic t-strategy)
                    # =============================================================

                    # Construct the Lagrangian with unit rescaling to get the initial gradient
                    ϕ   = BundleNetworks.constructFunction(ins, 1.0)

                    # Evaluate the gradient at zero to determine the rescaling factor
                    v, g = BundleNetworks.value_gradient(ϕ, zeros(sizeInputSpace(ϕ)))

                    # Reconstruct the function with the gradient-norm rescaling factor
                    ϕ = BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))

                    # (Unused in this branch but kept for API consistency)
                    factory = BundleNetworks.RnnTModelfactory()

                    # Initialize the VanillaBundle at zero with the given regularization parameter t
                    B = BundleNetworks.initializeBundle(
                        VanillaBundleFactory(), ϕ, t,
                        zeros(BundleNetworks.numberSP(ϕ))
                    )

                    # Override the maximum iteration count from the command-line argument
                    B.params.maxIt = mI

                    # Run the bundle method with the selected t-strategy (method[2])
                    oc, tD = BundleNetworks.solve!(B, ϕ; t_strat = method[2])

                    # Collect the rescaled objective history (skip entry 0 = initialization)
                    v_b = B.all_objs[2:end] .* ϕ.rescaling_factor

                    res[method[1]][t][directory[i]]["times"] = tD
                end

                # --- Store per-run summary results ---
                res[method[1]][t][directory[i]]["time"] = time() - tInit   # Total wall-clock time
                res[method[1]][t][directory[i]]["obj"]  = v_b              # Objective value history
            end
        end
    end

    # --- Write results to disk ---
    # Output filename encodes the dataset name extracted from the folder path
    output_name = "Results_" * split(folder, "/")[end-1] * ".json"
    f = open(output_name, "w")
    JSON.print(f, res)
    close(f)
end


# Entry point: pass command-line arguments to main
main(ARGS)