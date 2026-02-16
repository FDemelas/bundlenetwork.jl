# Installation

## Prerequisites

- Julia 1.9 or higher
- (Optional) CUDA-compatible GPU for acceleration

## Installing Julia

Download Julia from [julialang.org](https://julialang.org/downloads/)

### Linux/macOS
```bash
wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz
tar zxvf julia-1.9.4-linux-x86_64.tar.gz
export PATH="$PATH:$PWD/julia-1.9.4/bin"
```

### Windows

Download and run the installer from the Julia website.

## Installing BundleNetworks

### Option 1: Clone from GitHub
```bash
git clone https://github.com/yourusername/BundleNetworks.jl.git
cd BundleNetworks.jl
```

### Option 2: Julia Package Manager (if registered)
```julia
using Pkg
Pkg.add("BundleNetworks")
```

## Dependencies

Install required packages:
```julia
using Pkg

# Core dependencies
Pkg.add([
    "BundleNetworks",
    "Instances", 
    "Flux",
    "Zygote",
    "LinearAlgebra",
    "Statistics",
])

# Optional: GPU support
Pkg.add("CUDA")

# Utilities
Pkg.add([
    "JSON",
    "BSON",
    "ArgParse",
    "Random",
])

# Logging and visualization
Pkg.add([
    "TensorBoardLogger",
    "Logging",
])

# Training utilities
Pkg.add([
    "MLUtils",
    "ParameterSchedulers",
    "ChainRules",
    "ChainRulesCore",
])
```

## Verifying Installation
```julia
using BundleNetworks
using Flux
using CUDA

# Check CUDA availability
CUDA.functional()  # Should return true if GPU is available

# Check BundleNetworks
println("BundleNetworks loaded successfully!")
```

## Setting Up Data

1. Create data directories:
```bash
mkdir -p data
mkdir -p golds
mkdir -p resLogs
```

2. Download or generate problem instances

3. Create gold solutions file (if using .dat format):
```bash
mkdir -p golds/MCNDforTest
# Add your gold.json file
```

## Next Steps

- See [Quick Start](@ref) for your first training run
- Explore [Tutorials](@ref) for detailed examples
