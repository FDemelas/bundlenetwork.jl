"""
    InnerLoss <: AbstractConcaveFunction

Inner loss function for training neural networks on the MNIST image classification dataset.

# ⚠️ Important Note on Concavity
This function is **not actually concave** in the mathematical sense. It is included
to test the package's capabilities for meta-learning tasks where the "concavity"
assumption may be relaxed or approximately satisfied in practice.

# Purpose
Represents the loss landscape of a neural network trained on MNIST digit classification.
Used in meta-learning scenarios where we want to optimize over the parameters of
an inner optimization problem (the neural network training).

# Problem Setting
- **Input**: 28×28 grayscale images (MNIST digits)
- **Output**: One of 10 classes (digits 0-9)
- **Encoding**: Images flattened to vectors, labels as one-hot vectors
- **Batch processing**: Handles multiple samples simultaneously

# Fields
- `x::Any`: Input batch - Images flattened to vectors (784 × batch_size)
- `y::Any`: Output batch - One-hot encoded labels (10 × batch_size)
- `loss::Any`: Loss function (typically negative logit cross-entropy for maximization)
- `layers::Any`: Vector specifying network architecture [input_size, hidden_sizes..., output_size]
- `rescaling_factor::Any`: Divisor to scale loss values (for numerical stability)

# Network Architecture
The neural network has `length(layers) - 1` hidden layers where:
- `layers[1]`: Input dimension (typically 784 for 28×28 images)
- `layers[i]` for i ∈ [2, ..., n-1]: Hidden layer sizes
- `layers[n]`: Output dimension (10 for digit classification)

Each layer uses sigmoid activation except the output layer.

# Mathematical Formulation
For a network with parameters θ = [W₁, b₁, W₂, b₂, ..., Wₙ, bₙ]:
```
h₁ = σ(W₁'x + b₁)
h₂ = σ(W₂'h₁ + b₂)
...
ŷ = σ(Wₙ'hₙ₋₁ + bₙ)

L(θ) = loss(ŷ, y) / rescaling_factor
```

where σ is the sigmoid activation function.

# Why Not Concave?
Neural network loss landscapes are typically:
- Non-convex (and thus their negatives are non-concave)
- Highly non-linear with many local minima
- Complex with saddle points and plateaus

However, in meta-learning contexts, we may still apply optimization methods
designed for concave functions as heuristics.

# Meta-Learning Context
In meta-learning, we optimize over:
1. **Inner loop**: Train the network on a task (here: MNIST classification)
2. **Outer loop**: Learn good initialization or hyperparameters

This struct represents the inner loss landscape that the outer loop optimizes over.

# See Also
- `constructFunction`: Constructor for creating InnerLoss instances
- `prediction`: Forward pass without loss computation (inference)
- `sizeInputSpace`: Compute parameter space dimension
"""
struct InnerLoss <: AbstractConcaveFunction
    x::Any
    y::Any
    loss::Any
    layers::Any
    rescaling_factor::Any
end

"""
    (l::InnerLoss)(params)

Forward pass computing the loss for given network parameters.

# Arguments
- `params`: Flattened vector containing all network parameters [W₁, b₁, W₂, b₂, ..., Wₙ, bₙ]

# Returns
- Loss value scaled by `rescaling_factor`: `loss(ŷ, y) / rescaling_factor`

# Process
1. **Unpack parameters**: Extract weights (W) and biases (b) for each layer from flat vector
2. **Forward propagation**: Apply each layer transformation sequentially
3. **Loss computation**: Compute loss between predictions and true labels
4. **Scaling**: Divide by rescaling factor for numerical stability

# Parameter Layout
Parameters are stored as a flattened vector in the following order:
```
[W₁ (flattened), b₁, W₂ (flattened), b₂, ..., Wₙ (flattened), bₙ]
```

For a layer with input dimension `d_in` and output dimension `d_out`:
- Weight matrix W: `d_in × d_out` elements (stored column-major)
- Bias vector b: `d_out` elements

Total parameters: Σᵢ (dᵢ × dᵢ₊₁ + dᵢ₊₁)


# Activation Function
Uses **sigmoid** activation for all layers:
```
σ(x) = 1 / (1 + exp(-x))
```

Properties:
- Output range: (0, 1)
- Smooth and differentiable
- Suitable for intermediate representations

# Batch Processing
The forward pass handles batches automatically:
- Input `x`: (input_dim, batch_size)
- Each layer: `h = σ(W' × h + b)` broadcasts over batch dimension
- Output `ŷ`: (output_dim, batch_size)
- Loss: Computed over entire batch

# Numerical Considerations
- Uses `copy()` to avoid aliasing issues with gradients
- Rescaling factor prevents numerical overflow/underflow
- All operations preserve Float32 precision for GPU compatibility
"""
function (l::InnerLoss)(params)
    # Initialize with input data
    h, y = l.x, l.y
    
    # Track position in parameter vector
    first = 1
    
    # Propagate through each layer
    for i in 1:(length(l.layers) - 1)
        # Extract weight matrix for this layer
        # Size: layers[i] × layers[i+1]
        W = copy(reshape(
            params[first:(first + prod(l.layers[i:i+1]) - 1)], 
            l.layers[i:i+1]...
        ))
        first += prod(l.layers[i:i+1]) - 1
        
        # Extract bias vector for this layer
        # Size: layers[i+1]
        b = copy(params[first:first + l.layers[i+1] - 1])
        
        # Apply layer transformation: h = σ(W'h + b)
        h = copy(sigmoid.(W' * h .+ b))
        
        # Move to next layer's parameters
        first += l.layers[i+1]
    end
    
    # Final activation is the prediction
    ŷ = h
    
    # Compute and scale loss
    return l.loss(ŷ, y) / l.rescaling_factor
end

# Declare as Flux layer for proper integration
Flux.@layer InnerLoss

"""
    prediction(l::InnerLoss, params)

Performs inference (forward pass) without computing loss.

# Arguments
- `l::InnerLoss`: The inner loss function structure
- `params`: Network parameters (same format as callable)

# Returns
- Probability distribution over classes (10,) or (10, batch_size)
- Output is softmax-normalized to sum to 1

# Difference from Callable
- **Callable `l(params)`**: Computes loss (for training)
- **`prediction(l, params)`**: Returns class probabilities (for inference)

# Use Cases
- Model evaluation on test set
- Visualization of predictions
- Deployment/inference phase
- Computing accuracy metrics

# Process
1. Forward propagation through all layers (same as callable)
2. Apply **softmax** to final layer output instead of computing loss
3. Return normalized probability distribution

# Softmax Function
Converts logits to probabilities:
```
softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
```

Properties:
- Outputs sum to 1
- Each output ∈ (0, 1)
- Preserves relative ordering
- Differentiable


# Batch Processing
- Input batch: (28×28, batch_size)
- Output: (10, batch_size) probability distributions
- Each column sums to 1.0

# Implementation Note
This function should **only be used during inference**, not during training,
as it doesn't compute the loss required for gradient-based optimization.
"""
function prediction(l, params)   
    # Initialize with input data
    h, y = l.x, l.y
    
    # Track position in parameter vector
    first = 1
    
    # Forward propagation through all layers (same as callable)
    for i in 1:(length(l.layers) - 1)
        # Extract weight matrix
        W = copy(reshape(
            params[first:(first + prod(l.layers[i:i+1]) - 1)], 
            l.layers[i:i+1]...
        ))
        first += prod(l.layers[i:i+1]) - 1
        
        # Extract bias vector
        b = copy(params[first:first + l.layers[i+1] - 1])
        
        # Apply layer transformation
        h = copy(sigmoid.(W' * h .+ b))
        
        # Move to next layer
        first += l.layers[i+1]
    end
    
    # Apply softmax to get probability distribution
    # (instead of computing loss as in callable)
    return softmax(h)
end

"""
    constructFunction(data, rescaling_factor::Real, layers = [28 * 28, 20, 10])

Constructs an `InnerLoss` function for MNIST classification.

# Arguments
- `data`: Tuple `(x, y)` containing input images and labels
  - `x`: Images as 3D array (28, 28, batch_size) or 2D (784, batch_size)
  - `y`: Integer labels (batch_size,) with values 0-9
- `rescaling_factor::Real`: Divisor for loss scaling (default: no specific default shown)
- `layers`: Network architecture specification (default: [784, 20, 10])

# Returns
- `InnerLoss` instance ready for training or inference

# Default Architecture
```
[28 × 28, 20, 10] = [784, 20, 10]
- Input layer: 784 neurons (28×28 pixels)
- Hidden layer: 20 neurons
- Output layer: 10 neurons (one per digit class)
```

# Data Preprocessing
Performs several transformations on input data:

## Label Encoding
Converts integer labels to one-hot vectors:
```
# Input: y = [3, 7, 2, ...]  (class indices 0-9)
# Output: y = [[0,0,0,1,0,0,0,0,0,0],
#              [0,0,0,0,0,0,0,1,0,0],
#              [0,0,1,0,0,0,0,0,0,0], ...]
#         (10 × batch_size matrix)
```

## Image Reshaping
Flattens 2D images to vectors:
```
# Input: x shape (28, 28, batch_size)
# Output: x shape (784, batch_size)
```

# Loss Function
Uses **negative logit cross-entropy** for maximization:
```
f(ŷ, y) = -Flux.logitcrossentropy(ŷ, y)
```

The negative sign converts minimization to maximization, which is the
convention for `AbstractConcaveFunction`.

# Cross-Entropy Loss
Logit cross-entropy directly on logits (before softmax):
```
CE(ŷ, y) = -Σᵢ yᵢ log(softmax(ŷ)ᵢ)
```

More numerically stable than applying softmax then computing entropy.

# Device Handling
Automatically moves data to appropriate device (CPU/GPU):
- `device(x)`: Moves images to current device
- `device(y)`: Moves labels to current device
- `cpu(layers)`: Keeps architecture spec on CPU (metadata)

# Rescaling Factor Guidelines
The rescaling factor affects optimization dynamics:
- **Too small** (e.g., 1.0): Large gradient magnitudes, potential instability
- **Too large** (e.g., 10000.0): Small gradients, slow learning
- **Recommended**: 10-1000 depending on batch size and architecture

Typical values:
- Small networks: 10-50
- Medium networks: 50-200
- Large networks: 100-1000

# See Also
- `sizeInputSpace`: Compute required parameter vector size
- `prediction`: Inference without loss computation
- `value_gradient`: Compute loss and gradients
"""
function constructFunction(data, rescaling_factor::Real, layers = [28 * 28, 20, 10])
    # Unpack data
    x, y = data
    
    # Convert integer labels to one-hot encoding
    # For each label yᵢ, create vector with 1 at position yᵢ and 0 elsewhere
    y = hcat([[j == y[i] for j in 0:9] for i in eachindex(y)]...)
    
    # Reshape images from 3D (28, 28, batch) to 2D (784, batch)
    # prod(size(x)[1:2]) computes 28 × 28 = 784
    x = reshape(x, prod(size(x)[1:2]), size(x)[3])
    
    # Define loss function (negative for maximization convention)
    f(i, j) = -Flux.logitcrossentropy(i, j)
    
    # Create and return InnerLoss instance
    return InnerLoss(
        device(x),              # Move images to GPU/CPU
        device(y),              # Move labels to GPU/CPU
        f,                      # Loss function
        cpu(layers),            # Keep architecture on CPU (metadata)
        rescaling_factor        # Scaling factor
    )
end

"""
    sizeInputSpace(l::InnerLoss)

Computes the dimension of the parameter space for the neural network.

# Arguments
- `l::InnerLoss`: The inner loss function structure

# Returns
- `Int`: Total number of parameters in the network

# Formula
For a network with architecture [d₁, d₂, ..., dₙ]:
```
Total parameters = Σᵢ₌₁ⁿ⁻¹ (dᵢ × dᵢ₊₁ + dᵢ₊₁)
                 = Σᵢ₌₁ⁿ⁻¹ dᵢ₊₁(dᵢ + 1)
```

Where:
- `dᵢ × dᵢ₊₁`: Weight matrix parameters for layer i
- `dᵢ₊₁`: Bias vector parameters for layer i

# Breakdown by Layer
For each layer connection i → i+1:
- **Weights**: `layers[i] × layers[i+1]` parameters
- **Biases**: `layers[i+1]` parameters
- **Total**: `layers[i] × layers[i+1] + layers[i+1]`

# Relationship to Network Capacity
The parameter count determines:
- **Model capacity**: More parameters → more expressiveness
- **Memory requirements**: Linear in parameter count
- **Training time**: More parameters → slower training
- **Overfitting risk**: More parameters → higher risk with small datasets

# See Also
- `constructFunction`: Creates InnerLoss with specified architecture
- Parameter initialization strategies
- Network architecture design
"""
function sizeInputSpace(l::InnerLoss)
    return sum([prod(l.layers[i:i+1]) + l.layers[i+1] for i in 1:length(l.layers) - 1])
end

"""
    numberSP(l::InnerLoss)

Returns the number of subproblems (always 1 for InnerLoss).

# Returns
- `1`: InnerLoss represents a single optimization problem

# Purpose
This function is part of a generic interface for problems that may
decompose into multiple subproblems. For MNIST classification with
InnerLoss, there is no such decomposition, so it always returns 1.

# Context
In other parts of the package (e.g., Lagrangian decomposition), a problem
might split into multiple independent subproblems that can be solved
in parallel. This function reports how many such subproblems exist.

For InnerLoss:
- Single network optimization problem
- No decomposition structure
- Always returns 1
"""
function numberSP(l::InnerLoss)
    return 1
end

"""
    value_gradient(ϕ::InnerLoss, z::AbstractArray)

Computes loss value and gradient for the inner loss function.

# Arguments
- `ϕ::InnerLoss`: The inner loss function
- `z::AbstractArray`: Network parameters (flattened vector)

# Returns
- `value`: Loss value (scalar)
- `gradient`: Gradient w.r.t. parameters (same shape as z)

# Specialization
This is a **specialized implementation** of `value_gradient` for `InnerLoss`.
Unlike the default implementation in `AbstractConcaveFunction`, this version:
- Keeps data on GPU/CPU (no unnecessary transfers)
- Uses `device(z)` to ensure parameters are on correct device

# Why Override?
The default `value_gradient` for `AbstractConcaveFunction`:
1. Moves input to CPU: `z = cpu(z)`
2. Computes on CPU
3. Moves results back: `device(obj), device(grad[1])`

For neural networks, this is inefficient because:
- Networks benefit from GPU acceleration
- CPU ↔ GPU transfers are expensive
- Gradients are large (thousands of parameters)


# Performance Considerations
- **GPU acceleration**: Keeps computation on GPU throughout
- **Memory efficiency**: No unnecessary copies
- **Batch processing**: Efficiently handles batched data

# See Also
- `value_gradient(::AbstractConcaveFunction, ...)`: Default implementation
- `Flux.withgradient`: Automatic differentiation
"""
function value_gradient(ϕ::InnerLoss, z::AbstractArray)
    # Ensure parameters are on correct device, compute value and gradient
    obj, grad = Flux.withgradient((x) -> ϕ(x), device(z))
    
    # Return both on correct device
    return device(obj), device(grad[1])
end