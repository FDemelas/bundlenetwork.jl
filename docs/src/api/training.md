# Training API Reference

## Batch Training
```@docs
main(args)
```

### Function: `main`

Main entry point for batch training.

**Source**: `runs/train_batch.jl`

**Arguments**:
- `args::Vector{String}`: Command-line arguments

**Required Arguments**:
- `--lr::Float64`: Learning rate
- `--mti::Int64`: Maximum training instances
- `--mvi::Int64`: Maximum validation instances  
- `--seed::Int64`: Random seed
- `--maxItBack::Int64`: Max iterations for backward pass
- `--maxEP::Int64`: Maximum epochs

**Optional Arguments**:
- `--data::String`: Instance folder path (default: "./data/MCNDforTest/")
- `--decay::Float64`: Learning rate decay (default: 0.9)
- `--lambda::Float32`: Final point weight (default: 0.0)
- `--gamma::Float32`: Telescopic weight (default: 0.0)
- `--cn::Int64`: Gradient clipping norm (default: 5)
- `--batch_size::Int64`: Batch size (default: 1)
- `--h_representation::Int64`: Hidden size (default: 64)

**Output Files**:
- `nn.bson`: Final model
- `nn_best.bson`: Best validation model
- `loss.json`, `obj.json`, `gaps.json`: Training metrics
- `obj_val.json`, `gaps_val.json`: Validation metrics

**Example**:
```julia
args = [
    "--data", "./data/MCNDforTest/",
    "--lr", "0.001",
    "--mti", "100",
    "--mvi", "20",
    "--seed", "42",
    "--maxItBack", "50",
    "--maxEP", "100"
]
main(args)
```

## Episodic Training

### Function: `ep_train_and_val`

Execute episodic training and validation.

**Source**: `runs/train_episodic.jl`

**Signature**:
```julia
function ep_train_and_val(
    folder, directory, dataset, gold, 
    idxs_train, idxs_val, opt;
    maxEP=10, maxIT=50, kwargs...
)
```

**Arguments**:
- `folder::String`: Instance folder path
- `directory::Vector{String}`: File list
- `dataset::Vector{Tuple}`: (filename, objective) pairs
- `gold::Dict`: Optimal solutions
- `idxs_train::Vector{Int}`: Training indices
- `idxs_val::Vector{Int}`: Validation indices
- `opt`: Flux optimizer

**Keyword Arguments**:
- `maxEP::Int=10`: Maximum epochs
- `maxIT::Int=50`: Bundle iterations
- `cr_init::Bool=false`: Use CR initialization
- `exactGrad::Bool=true`: Exact gradient formula
- `telescopic::Bool=false`: Telescopic loss
- `γ::Float64=0.1`: Telescopic weight decay
- `instance_features::Bool=false`: Include instance features

**Returns**: Nothing (saves to disk)

**Example**:
```julia
opt = Flux.OptimiserChain(Flux.Optimise.Adam(0.001), ClipNorm(5))
ep_train_and_val(
    "./data/", directory, dataset, gold, 
    1:100, 101:120, opt;
    maxEP=50, maxIT=50, telescopic=true
)
```

### Function: `saveJSON`

Helper to save dictionary to JSON.

**Signature**:
```julia
function saveJSON(name::String, res::Dict)
```

**Arguments**:
- `name::String`: Output file path
- `res::Dict`: Dictionary to save

**Example**:
```julia
results = Dict("loss" => [1.0, 0.8, 0.6])
saveJSON("results.json", results)
```

## Common Helper Functions

### `gap`

Calculate relative percentage gap.

**Signature**:
```julia
gap(a, b) = abs(a - b) / max(a, b) * 100
```

**Arguments**:
- `a::Float64`: First value (solution)
- `b::Float64`: Second value (optimal)

**Returns**: `Float64` - Percentage gap

**Example**:
```julia
solution = 95.0
optimal = 100.0
g = gap(solution, optimal)  # Returns 5.0
```
