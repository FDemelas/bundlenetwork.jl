# BundleNetworks.jl Documentation

*Neural Network-Guided Bundle Methods for Non-Smooth Optimization*

## Overview

BundleNetworks.jl implements:
- A machine learning approach to accelerate bundle methods by learning optimal hyperparameters from training data.
- A machine learning-based unrolling model that predicts the coefficients of the convex combination of gradients (considered as step size), as well as the step size itself.

## Features

- **Neural Network-Guided Optimization**: Learn bundle method parameters using attention mechanisms.
- **Flexible Training**: Supports batch and episodic training modes.
- **Curriculum Learning**: Gradually increases problem difficulty.
- **Comprehensive Evaluation**: Tracks training, validation, and testing metrics with TensorBoard integration.
- **GPU Support**: Optional CUDA acceleration.

## Quick Example
```julia
# Train a model
julia runTraining.jl \
  --data ./data/<your_folder>/ \
  --lr 0.001 \
  --mti 100 \
  --mvi 20 \
  --seed 42 \
  --maxItBack 50 \
  --maxEP 100

# Test the model
julia runTest.jl \
  --data ./data/<your_folder>/ \
  --model ./resLogs/model_folder/ \
  --dataset ./resLogs/model_folder/
```

## Package Contents
```@contents
Pages = [
    "installation.md",
    "quickstart.md",
]
Depth = 2
```

## Manual Outline
```@contents
Pages = [
    "manual/architecture.md",
    "manual/bundle_methods.md",
    "manual/data_formats.md",
]
Depth = 1
```

## Index
```@index
```