"""
    sparsemax(γi; dims=1) -> AbstractArray

Compute the Sparsemax activation function along dimension `dims`.

Sparsemax is a sparse alternative to Softmax that projects its input onto the
probability simplex, but produces **exactly zero** for components that are
sufficiently below the threshold. This yields sparse probability distributions,
which is useful for attention mechanisms and multi-label classification.

The algorithm follows the projection-onto-simplex procedure from:
> *From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification*
> André F. T. Martins, Ramón Fernandez Astudillo (ICML 2016)

The closed-form solution is:
    sparsemax(z)ᵢ = max(zᵢ - τ(z), 0)

where the threshold τ(z) is computed from the sorted input as:
    τ(z) = (∑ᵢ₌₁ᵏ z₍ᵢ₎ - 1) / k,    k = max{j : j·z₍ⱼ₎ - ∑ᵢ₌₁ʲ z₍ᵢ₎ + 1 > 0}

# Arguments
- `γi`: Input array (modified in-place for numerical stability via max-translation).
- `dims`: Dimension along which the Sparsemax is computed (default: `1`).

# Returns
A non-negative array with the same shape as `γi`, summing to 1 along `dims`.
Entries below the threshold τ are exactly zero (sparse output).
"""
function sparsemax(γi; dims=1)
    # Translate input by its maximum for numerical stability (analogous to the
    # log-sum-exp trick for Softmax); does not change the Sparsemax output
    γi .-= maximum(γi; dims)

    # Sort the (shifted) input in decreasing order along `dims`
    z = sort(γi; dims, rev = true)

    # Compute the cumulative sum of the sorted values: cs[k] = ∑ᵢ₌₁ᵏ z₍ᵢ₎
    cs = cumsum(z; dims)

    # Build an index matrix that assigns each element its rank along `dims`.
    # Shape matches `z`: rng[i,j] = j if dims==2, rng[i,j] = i if dims==1.
    # This allows vectorized comparison across the entire array.
    rng = device(
        dims == 2 ?
            repeat(collect(1:size(z, 2))', size(z, 1)) :   # Row of column indices (dims=2)
            repeat(collect(1:size(z, 1))', size(z, 2))'    # Column of row indices (dims=1)
    )

    # Boolean mask: entry (i,j) is true iff rank k satisfies the support condition
    # k·z₍ₖ₎ + 1 > ∑ᵢ₌₁ᵏ z₍ᵢ₎, i.e., the k-th sorted element is still in the support
    is_gt = (rng .* z .+ 1 .> cs)

    # Find the largest rank k satisfying the support condition (the support size)
    k = maximum(rng .* is_gt; dims)

    # Compute the threshold τ = (∑ᵢ∈support z₍ᵢ₎ - 1) / k
    # This is the value subtracted from each input component before applying relu
    τ = (sum(z .* is_gt; dims) .- 1) ./ k

    # Apply the projection: sparsemax(γi)ᵢ = max(γi - τ, 0)
    # Components below τ are mapped to exactly zero (sparse output)
    return relu(γi .- τ)
end


"""
    ChainRulesCore.rrule(::typeof(sparsemax), γi; dims=1)

Custom reverse-mode differentiation rule for the Sparsemax function.

The gradient of a loss `L` with respect to the Sparsemax input `γi` is:

    ∂L/∂γi = dl - (∑ⱼ dlⱼ · 𝟙[sparsemax(γi)ⱼ ≠ 0]) / |support| · 𝟙[sparsemax(γi)ᵢ ≠ 0]

where `dl` is the upstream gradient and `|support|` is the number of non-zero
output components. In words: the gradient is the upstream signal minus its mean
over the support, masked to zero outside the support.

This is the analytical gradient of the simplex projection, derived from the
KKT conditions of the underlying quadratic program.

# Arguments
- `γi`: Input array (modified in-place during the forward pass for stability).
- `dims`: Dimension along which Sparsemax is applied (default: `1`).

# Returns
- `val`: The Sparsemax output (on device).
- `loss_pullback`: A function mapping the upstream cotangent `dl` to the
  input cotangent `∂L/∂γi`. Returns `(NoTangent(), gradient, NoTangent())`
  matching the `(function, γi, dims)` argument signature.
"""
function ChainRulesCore.rrule(::typeof(sparsemax), γi; dims=1)
    # --- Forward pass (identical to sparsemax, recomputed to cache intermediate values) ---

    # Shift by maximum for numerical stability
    γi .-= maximum(γi; dims)

    # Sort in decreasing order to find the support
    z = sort(γi; dims, rev = true)

    # Cumulative sum of sorted values: cs[k] = ∑ᵢ₌₁ᵏ z₍ᵢ₎
    cs = cumsum(z; dims)

    # Index matrix: rng[i,j] encodes the rank of element (i,j) along `dims`
    rng = device(
        dims == 2 ?
            repeat(collect(1:size(z, 2))', size(z, 1)) :   # Column rank indices (dims=2)
            repeat(collect(1:size(z, 1))', size(z, 2))'    # Row rank indices (dims=1)
    )

    # Boolean mask for the support: true at rank k iff k·z₍ₖ₎ + 1 > ∑ᵢ₌₁ᵏ z₍ᵢ₎
    is_gt = (rng .* z .+ 1 .> cs)

    # Size of the support (largest rank k satisfying the condition)
    k = maximum(rng .* is_gt; dims)

    # Threshold τ = (∑ᵢ∈support z₍ᵢ₎ - 1) / k
    τ = (sum(z .* is_gt; dims) .- 1) ./ k

    # Forward output: sparsemax(γi)ᵢ = max(γi - τ, 0)
    val = relu(γi .- τ)

    # --- Backward pass ---
    function loss_pullback(dl)
        # Identify the support: components where the Sparsemax output is non-zero
        non_zeros = device(val .!= 0)

        # Analytical gradient of the simplex projection:
        #   ∂L/∂γᵢ = dlᵢ - (∑ⱼ dlⱼ · 𝟙[support]) / |support|   if i ∈ support
        #   ∂L/∂γᵢ = 0                                             if i ∉ support
        #
        # In matrix form: dl - mean(dl over support) · 𝟙[support]
        # Implemented as: dl - (dl · non_zeros)ᵀ summed over dims, divided by support size
        gradient = dl .- sum(dl * non_zeros'; dims) ./ sum(non_zeros; dims)

        # Return cotangents in the order (function, γi, dims):
        # - NoTangent() for the function itself (not differentiable w.r.t. the function)
        # - gradient for γi (the meaningful input)
        # - NoTangent() for dims (not differentiable w.r.t. an integer keyword)
        return (NoTangent(), gradient, NoTangent())
    end

    return device(val), device(loss_pullback)
end