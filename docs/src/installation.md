
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

### Step 1: Clone from GitHub
```bash
git clone git@github.com:FDemelas/bundlenetwork.jl.git
cd bundlenetwork.jl
```

### Step 2: Julia Package Manager (if registered)
```julia
# Use the package manager
using Pkg

# Activate the project directory
Pkg.activate(".")

# Install dependencies
Pkg.instantiate()
```

## Verifying Installation
```julia
using BundleNetworks
using Flux
using CUDA

# Check CUDA availability
CUDA.functional()  # Should return true if GPU is available

# Confirm BundleNetworks loads successfully
println("BundleNetworks loaded successfully!")
```

## Setting Up Data

1. Create data directories:
```bash
mkdir -p data
mkdir -p golds
mkdir -p resLogs
```

2. Download or generate problem instances and place them in `./data/<your_folder>`

3. Create gold solutions file in `./golds/<your_folder>`. If the directory does not exist, create it:
```bash
mkdir -p golds/<your_folder>
# Add your gold.json file here
```

## Next Steps

- See [Quick Start](#) for your first training run
- Explore [Tutorials](#) for detailed examples
