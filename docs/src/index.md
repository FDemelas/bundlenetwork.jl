# BundleNetworks.jl Documentation

*Neural Network-Guided Bundle Methods for Non-Smooth Optimization*

## Overview

BundleNetworks.jl implements a machine learning approach to accelerate bundle methods 
by learning optimal hyperparameters from training data. The neural network predicts 
proximity parameters and gradient aggregation weights at each iteration, improving 
convergence speed compared to traditional bundle methods.

## Features

- **Neural Network-Guided Optimization**: Learn bundle method parameters using attention mechanisms
- **Flexible Training**: Batch and episodic training modes
- **Curriculum Learning**: Gradually increase problem difficulty
- **Comprehensive Evaluation**: Training, validation, and test tracking with TensorBoard integration
- **GPU Support**: Optional CUDA acceleration

## Quick Example
```julia
# Train a model
julia train_batch.jl \
  --data ./data/MCNDforTest/ \
  --lr 0.001 \
  --mti 100 \
  --mvi 20 \
  --seed 42 \
  --maxItBack 50 \
  --maxEP 100

# Test the model
julia test.jl \
  --data ./data/MCNDforTest/ \
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
