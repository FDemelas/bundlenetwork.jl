"""
Abstract type for modeling the 't-strategies', i.e., the heuristics used to handle 
increments and decrements of the parameter `t`. This parameter acts as:
  - a regularization term for the quadratic component of the loss in the Dual Master Problem, and
  - a step size controller.
All concrete t-strategy types must subtype this.
"""
abstract type abstract_t_strategy end

"""
Abstract type modeling the long-term component of a t-strategy.
Long-term strategies wrap a middle-term strategy and add higher-level logic
(e.g., gating conditions) before delegating increment/decrement decisions.
Subtypes of this must also subtype `abstract_t_strategy`.
"""
abstract type abstract_long_term_t_strategy <: abstract_t_strategy end

"""
Abstract type modeling the middle-term component of a t-strategy.
Middle-term strategies implement the core increment/decrement mechanics
(e.g., multiplicative updates with bounds). They are typically composed
inside a long-term strategy.
"""
abstract type abstract_middle_term_t_strategy end

"""
Constant t-strategy: `t` is never modified.
Use this when you want to disable any adaptive behavior of `t` and keep it fixed
throughout the algorithm's execution.
"""
struct constant_t_strategy <: abstract_t_strategy end

"""
Neural network-based t-strategy.
In this strategy, a neural network takes the current bundle state as input
and directly outputs the new value of `t`. This replaces the entire 
short/middle/long-term hierarchy with a single learned policy.
"""
struct nn_t_strategy <: abstract_t_strategy end

"""
Middle-term t-strategy using fixed multiplicative factors.
Increments and decrements are performed by multiplying `t` by fixed parameters
(`t_incr > 1` for increments, `0 < t_decrement < 1` for decrements),
clamped within the interval `[t_min, t_max]`.
See `increment_t` and `decrement_t` for implementation details.
"""
struct heuristic_t_strategy_1 <: abstract_middle_term_t_strategy end

"""
Soft long-term t-strategy.
Wraps a middle-term strategy and adds a guard on decrements:
a decrement is only allowed when `B.vStar > tSPar2 * B.ϵ`,
i.e., when the current subgradient gap is large enough relative to the tolerance.
Increments are always delegated directly to the middle-term strategy.

# Fields
- `middle_term_strategy::abstract_middle_term_t_strategy`: The underlying middle-term strategy used for actual updates.
"""
struct soft_long_term_t_strategy <: abstract_long_term_t_strategy 
    middle_term_strategy::abstract_middle_term_t_strategy
end

"""
Hard long-term t-strategy.
Similar to `soft_long_term_t_strategy`, but with an inverted response when 
a decrement is requested but the guard condition is NOT met:
instead of doing nothing, it *increments* `t` to push harder toward convergence.

# Fields
- `middle_term_strategy::abstract_middle_term_t_strategy`: The underlying middle-term strategy used for actual updates.
"""
struct hard_long_term_t_strategy <: abstract_long_term_t_strategy 
    middle_term_strategy::abstract_middle_term_t_strategy
end

"""
Balancing long-term t-strategy.
Tries to keep the quadratic term and the linear term of the dual objective at 
comparable magnitudes. Increments and decrements are only triggered when a 
magnitude imbalance is detected (controlled by `tSPar2`).

# Fields
- `middle_term_strategy::abstract_middle_term_t_strategy`: The underlying middle-term strategy used for actual updates.
"""
struct balancing_long_term_t_strategy <: abstract_long_term_t_strategy 
    middle_term_strategy::abstract_middle_term_t_strategy
end

# --- Constant strategy: no-ops ---

"""
    increment_t(B::AbstractBundle, ts::constant_t_strategy)

No-op. The constant t-strategy keeps `t` unchanged, so increment requests are ignored.
"""
function increment_t(B::AbstractBundle, ts::constant_t_strategy)
end

"""
    decrement_t(B::AbstractBundle, ts::constant_t_strategy)

No-op. The constant t-strategy keeps `t` unchanged, so decrement requests are ignored.
"""
function decrement_t(B::AbstractBundle, ts::constant_t_strategy)
end    

# --- Heuristic middle-term strategy ---

"""
    increment_t(B::AbstractBundle, ts::heuristic_t_strategy_1)

Increments `t` multiplicatively by `t_incr`, clamped to the upper bound `t_max`.
The update rule is: `t ← max(t, min(t * t_incr, t_max))`

The `max(t, ...)` ensures `t` never decreases from an increment call,
and the `min(..., t_max)` ensures it never exceeds the allowed ceiling.

# Parameters used from `B.params`
- `t_incr`: Multiplicative factor > 1 for incrementing `t`.
- `t_max`: Upper bound on `t`.
"""
function increment_t(B::AbstractBundle, ts::heuristic_t_strategy_1)
    B.params.t = max(B.params.t, min(B.params.t * B.params.t_incr, B.params.t_max))
end

"""
    decrement_t(B::AbstractBundle, ts::heuristic_t_strategy_1)

Decrements `t` multiplicatively by `t_decrement`, clamped to the lower bound `t_min`.
The update rule is: `t ← min(t, max(t * t_decrement, t_min))`

The `min(t, ...)` ensures `t` never increases from a decrement call,
and the `max(..., t_min)` ensures it never goes below the allowed floor.

# Parameters used from `B.params`
- `t_decrement`: Multiplicative factor in (0, 1) for decrementing `t`.
- `t_min`: Lower bound on `t`.
"""
function decrement_t(B::AbstractBundle, ts::heuristic_t_strategy_1)
    B.params.t = min(B.params.t, max(B.params.t * B.params.t_decrement, B.params.t_min))
end    

# --- Soft long-term strategy ---

"""
    increment_t(B::AbstractBundle, ts::soft_long_term_t_strategy)

Delegates the increment unconditionally to the wrapped middle-term strategy.
"""
function increment_t(B::AbstractBundle, ts::soft_long_term_t_strategy)
    increment_t(B, ts.middle_term_strategy)
end

"""
    decrement_t(B::AbstractBundle, ts::soft_long_term_t_strategy)

Allows a decrement only if the current gap `B.vStar` is sufficiently large 
relative to the tolerance: `B.vStar > tSPar2 * B.ϵ`.

This guard prevents reducing `t` when the algorithm is already near convergence,
which could destabilize the method. If the condition is met, delegates to the
wrapped middle-term strategy.
"""
function decrement_t(B::AbstractBundle, ts::soft_long_term_t_strategy)
    if B.vStar > B.params.tSPar2 * B.ϵ
        decrement_t(B, ts.middle_term_strategy)
    end
end    

# --- Hard long-term strategy ---

"""
    increment_t(B::AbstractBundle, ts::hard_long_term_t_strategy)

Delegates the increment unconditionally to the wrapped middle-term strategy.
"""
function increment_t(B::AbstractBundle, ts::hard_long_term_t_strategy)
    increment_t(B, ts.middle_term_strategy)
end

"""
    decrement_t(B::AbstractBundle, ts::hard_long_term_t_strategy)

Uses an inverted logic compared to `soft_long_term_t_strategy`:
- If `B.vStar > tSPar2 * B.ϵ` (gap is large): a decrement *would* be appropriate,
  but this branch is currently commented out (no action is taken).
- Otherwise (gap is small, near convergence): instead of doing nothing,
  it *increments* `t` to actively push the algorithm forward.

Note: The decrement branch is commented out; currently only the increment
on the else branch is active.
"""
function decrement_t(B::AbstractBundle, ts::hard_long_term_t_strategy)
    if B.vStar > B.params.tSPar2 * B.ϵ
        # Decrement is currently disabled (under investigation/tuning)
        # decrement_t(B, ts.middle_term_strategy)
    else
        # When near convergence, aggressively increase t to accelerate progress
        increment_t(B, ts.middle_term_strategy)
    end
end    

# --- Balancing long-term strategy ---

"""
    increment_t(B::AbstractBundle, ts::balancing_long_term_t_strategy)

Increments `t` only if the quadratic term dominates the linear term by a sufficient margin.
Specifically, triggers increment when:
    `t_star * (w'w / 2) > tSPar2 * (α[1:len(θ)]' * θ)`

This prevents over-regularization: `t` is only increased when the quadratic 
contribution to the dual objective is already notably larger than the linear part.
"""
function increment_t(B::AbstractBundle, ts::balancing_long_term_t_strategy)
    # Increment only if the quadratic term (scaled by t_star) outweighs the linear term
    if B.params.t_star * B.w' * B.w / 2 > B.params.tSPar2 * B.α[1:length(B.θ)]' * B.θ
        increment_t(B, ts.middle_term_strategy)
    end
end

"""
    decrement_t(B::AbstractBundle, ts::balancing_long_term_t_strategy)

Decrements `t` only if the linear term dominates the quadratic term by a sufficient margin.
Specifically, triggers decrement when:
    `tSPar2 * t_star * (w'w / 2) < α[1:len(θ)]' * θ`

This avoids under-regularization: `t` is only decreased when the linear contribution
to the dual objective clearly outweighs the (scaled) quadratic part.
"""
function decrement_t(B::AbstractBundle, ts::balancing_long_term_t_strategy)
    # Decrement only if the linear term outweighs the scaled quadratic term
    if B.params.tSPar2 * B.params.t_star * B.w' * B.w / 2 < B.α[1:length(B.θ)]' * B.θ
        decrement_t(B, ts.middle_term_strategy)
    end
end    

# --- Neural network strategy ---

"""
    increment_t(B::AbstractBundle, ts::nn_t_strategy)

Uses the neural network stored in `B.nn` to compute the new value of `t`.
Features are extracted from the current bundle state, moved to the appropriate 
device (GPU/CPU), and then passed through the network. The output is moved back 
to CPU and assigned to `B.params.t`.

Note: In this strategy, both `increment_t` and `decrement_t` perform the 
same operation — the network decides the new `t` regardless of the direction signal.
"""
function increment_t(B::AbstractBundle, ts::nn_t_strategy)
    ϕ = create_features(B, B.nn)   # Build input feature vector from bundle state
    ϕ = device(ϕ)                  # Move features to the appropriate device (e.g., GPU)
    B.params.t = cpu(B.nn(ϕ, B))  # Forward pass through the network; move result back to CPU
end

"""
    decrement_t(B::AbstractBundle, ts::nn_t_strategy)

Uses the neural network stored in `B.nn` to compute the new value of `t`.
Identical to `increment_t` for `nn_t_strategy`: the network output directly 
sets `t`, ignoring the increment/decrement distinction.

See also: `increment_t(B, ts::nn_t_strategy)`
"""
function decrement_t(B::AbstractBundle, ts::nn_t_strategy)
    ϕ = create_features(B, B.nn)   # Build input feature vector from bundle state
    ϕ = device(ϕ)                  # Move features to the appropriate device (e.g., GPU)
    B.params.t = cpu(B.nn(ϕ, B))  # Forward pass through the network; move result back to CPU
end