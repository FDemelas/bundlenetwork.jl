"""
    BundleNetworks

A Julia module implementing bundle methods for nonsmooth optimization, with support
for both classical heuristic-based approaches and machine learning-based variants
that use neural networks to replace or augment parts of the classical algorithm.

# Overview

Bundle methods are proximal cutting-plane algorithms for maximizing nonsmooth
concave functions. At each iteration they maintain a *bundle* of subgradient
information from previously visited points and solve a *Dual Master Problem* (DMP)
to compute the next search direction.

This module provides four bundle variants, selectable via factory types:

| Factory | Bundle Type | t-Strategy | Direction |
|---|---|---|---|
| `VanillaBundleFactory` | `Bundle` | Heuristic (hand-crafted) | DMP solution |
| `tLearningBundleFactory` | `DeepBundle` | Neural network | DMP solution |
| `SoftBundleFactory` | `SoftBundle` | Neural network | Neural network |
| `BatchedSoftBundleFactory` | `BatchedSoftBundle` | Neural network (batched) | Neural network (batched) |

# Module Structure

## Bundle types and factories
- `AbstractBundle.jl`: Abstract type hierarchy for all bundle variants.
- `VanillaBundle.jl`: Classical proximal bundle method.
- `DualBundle.jl`: Shared logic for all DMP-based bundle variants.
- `tLearningBundle.jl`: Bundle with a neural network t-strategy.
- `SoftBundle.jl`: Fully neural-network-driven bundle (single instance).
- `BatchedSoftBundle.jl`: Fully neural-network-driven bundle (batched).

## t-Strategies
- `tStrategy.jl`: Heuristic strategies for updating the regularization parameter `t`
  (constant, heuristic, soft/hard/balancing long-term strategies).

## Hyperparameters
- `BundleParameters.jl`: `BundleParameters` struct with all tunable hyperparameters.

## Objective functions
- `AbstractConcave.jl`: Abstract type and interface for concave objective functions.
- `InnerLoss.jl`: Loss functions used during neural network training.
- `LagrangianMCND.jl`: Lagrangian relaxation for the Multi-Commodity Network Design problem.
- `LagrangianGA.jl`: Lagrangian relaxation for the Generalized Assignment problem.
- `LagrangianTUC.jl`: Lagrangian relaxation for the Time-Uncapacitated lot-sizing problem.

## Neural network models
- `AbstractModel.jl`: Abstract type hierarchy for all neural network models.
- `AttentionModel.jl`: Attention-based model predicting both `t` and the DMP weights.
- `tRecurrentMlp.jl`: Recurrent MLP model for predicting `t`.
- `tRecurrentMlp2.jl`: Alternative recurrent MLP variant for predicting `t`.
- `Deviations.jl`: Deviation/noise layers used in stochastic network variants.

## Auxiliary utilities
- `instanceFeatures.jl`: Feature extraction from MCND instances for GNN-based models.
- `sparsemax.jl`: Sparsemax activation function with custom Zygote-compatible backward pass.

# Device Management

The module automatically selects GPU execution if CUDA is available and `use_gpu = true`.
All computations are dispatched through the `device` function:
- GPU: `device = gpu` (via `Flux.gpu`)
- CPU: `device = cpu` (fallback)

# Exported Symbols

- `create_NN`: Construct a neural network model from a factory.
- `initializeBundle`: Construct and initialize a bundle from a factory, function, and starting point.
- `tLearningBundleFactory`, `SoftBundleFactory`, `BatchedSoftBundleFactory`, `VanillaBundleFactory`: Bundle factory types.
- `solve!`: Run the classical bundle method to convergence.
- `test_local_retraining`, `test`, `training_loop`, `training_epoch`: Training and evaluation utilities.
- `sizeLM`, `constructFunction`, `value_gradient`, `my_read_dat`: Instance and function utilities.
- `gap`: Compute the optimality gap as `((b - a) / b) * 100`.
- `sizeInputSpace`: Return the input space dimension of a concave function.
"""
module BundleNetworks

# --- Standard library ---
using LinearAlgebra       # Matrix operations (dot products, norms, rank, etc.)
using SparseArrays        # Sparse matrix support (used in constraint representations)
using Random              # Random number generation (for stochastic neural network variants)
using Statistics          # mean, std, etc. (used in loss computation and feature normalization)

# --- Optimization modeling ---
using JuMP                # Algebraic modeling language for mathematical programming
using HiGHS               # Open-source LP/MIP solver (fallback for non-Gurobi environments)
using Gurobi              # Commercial QP/MIP solver used for the Dual Master Problem

# --- Machine learning and automatic differentiation ---
using Flux                # Neural network library (layers, optimizers, training utilities)
using ChainRules          # Ecosystem for custom AD rules
using ChainRulesCore      # Core primitives for defining custom `rrule` backward passes
using Zygote              # Primary AD backend (reverse-mode, used for training loss gradients)

# --- GPU support ---
using CUDA                # NVIDIA GPU computing support
import CUDA: CuArray      # GPU array type (used for explicit type dispatch on GPU tensors)
import Flux: gpu, cpu     # Device transfer functions (move arrays between CPU and GPU)

# --- Problem instance utilities ---
using Instances           # Package providing MCND and other problem instance types
import Instances: LR, cpuInstanceMCND, sizeLM  # Lagrangian relaxation and instance helpers

# --- GPU runtime configuration ---
# Pin the CUDA runtime to version 12.1 to ensure compatibility with the installed drivers
CUDA.set_runtime_version!(v"12.1")

# --- Device selection ---
# Use GPU if CUDA is functional and use_gpu is enabled; otherwise fall back to CPU.
# All array transfers throughout the module are performed via `device(...)`.
use_gpu = true            # Set to `false` to force CPU execution regardless of GPU availability
device  = CUDA.functional() && use_gpu ? gpu : cpu

# --- Utility functions ---
"""
    gap(a, b) -> Float64

Compute the relative optimality gap (in percent) between a primal bound `a`
and a reference value `b`:
    gap(a, b) = ((b - a) / b) * 100

Commonly used to measure the distance between a heuristic solution value `a`
and an upper bound or optimal value `b`.
"""
gap(a, b) = ((b - a) / b) * 100

# =========================================================================
# Source file includes (order matters: abstract types before concrete types,
# shared utilities before specialized implementations)
# =========================================================================


# --- Neural network model interface ---
include("Models/AbstractModel.jl")           # AbstractModel, AbstractModelFactory, AbstractTModelFactory


# --- Hyperparameters ---
include("HyperParameters/BundleParameters.jl") # BundleParameters struct


# --- Abstract type hierarchy ---
include("Bundle/AbstractBundle.jl")          # AbstractBundle, DualBundle, AbstractSoftBundle, factories

# --- t-Strategy implementations ---
include("tStrategies/tStrategy.jl")          # constant, heuristic, soft/hard/balancing long-term strategies

# --- Objective function interface and implementations ---
include("ObjectiveFunctions/AbstractConcave.jl")  # AbstractConcaveFunction interface
include("ObjectiveFunctions/InnerLoss.jl")        # Training loss functions
include("ObjectiveFunctions/LagrangianMCND.jl")   # Lagrangian relaxation: Multi-Commodity Network Design
include("ObjectiveFunctions/LagrangianGA.jl")     # Lagrangian relaxation: Generalized Assignment
include("ObjectiveFunctions/LagrangianTUC.jl")    # Lagrangian relaxation: Time-Uncapacitated lot-sizing

# --- Auxiliary utilities ---
include("Auxiliary/instanceFeatures.jl")     # GNN feature extraction from MCND instances
include("Auxiliary/sparsemax.jl")            # Sparsemax activation + custom Zygote rrule

# --- Neural network model implementations ---
include("Models/Deviations.jl")              # Stochastic deviation/noise layers for exploration
include("Models/tAndDirectionModels/AttentionModel.jl")  # Attention model: predicts t and DMP weights jointly
include("Models/tModels/tRecurrentMlp.jl")   # Recurrent MLP: predicts t only (variant 1)
include("Models/tModels/tRecurrentMlp2.jl")  # Recurrent MLP: predicts t only (variant 2)

# --- Bundle implementations (concrete types, depend on all of the above) ---
include("Bundle/DualBundle.jl")              # Shared DMP logic: create/solve/update DQP, update bundle
include("Bundle/SoftBundle.jl")              # SoftBundle: single-instance, fully neural-network-driven
include("Bundle/BatchedSoftBundle.jl")       # BatchedSoftBundle: batched, fully neural-network-driven
include("Bundle/VanillaBundle.jl")           # Bundle (Vanilla): classical heuristic t-strategy
include("Bundle/tLearningBundle.jl")         # DeepBundle: DMP direction + neural network t-strategy

# =========================================================================
# Public API exports
# =========================================================================

# Neural network construction
export create_NN                    # Construct a neural network model from a factory type

# Bundle construction
export initializeBundle             # Initialize a bundle from a factory, function, and starting point

# Bundle factory types (used for dispatch in initializeBundle and bundle_execution)
export tLearningBundleFactory       # Factory for DeepBundle (neural network t-strategy)
export SoftBundleFactory            # Factory for SoftBundle (fully neural-network-driven, single instance)
export BatchedSoftBundleFactory     # Factory for BatchedSoftBundle (batched neural-network-driven)
export VanillaBundleFactory         # Factory for Bundle (classical heuristic t-strategy)

# Bundle execution
export solve!                       # Run the classical bundle method to convergence (VanillaBundle / DeepBundle)

# Training and evaluation utilities
export test_local_retraining        # Local retraining loop for adapting a pre-trained model to a new instance
export test                         # Evaluation function: run bundle and report objective / gap
export training_loop                # Outer training loop over multiple epochs and instances
export training_epoch               # Single training epoch: forward + backward + parameter update

# Instance and function utilities
export sizeLM                       # Return the size of the Lagrange multiplier space
export constructFunction            # Construct a Lagrangian function from a problem instance
export value_gradient               # Evaluate a concave function and its subgradient at a point
export my_read_dat                  # Read a problem instance from a .dat file

# Miscellaneous utilities
export gap                          # Optimality gap: ((b - a) / b) * 100
export sizeInputSpace               # Return the input space dimension of a concave function

end # module BundleNetworks