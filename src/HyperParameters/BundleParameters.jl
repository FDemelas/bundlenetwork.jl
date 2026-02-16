"""
Hyperparameters for the `tLearningBundle` and `VanillaBundle` variants.

This struct centralizes all tunable parameters controlling the behavior of the
bundle method: iteration limits, bundle size, step-size management, Serious Step /
Null Step decision thresholds, and stopping criteria.

Most parameters are relevant only for the `VanillaBundle` (which uses heuristic
t-strategies); the `tLearningBundle` uses a neural network to predict `t` directly
and ignores most t-strategy parameters.

All fields have default values and can be set via keyword arguments.

# Fields

## Iteration control
- `maxIt::Int64`: Maximum number of bundle iterations (default: `1000`).
- `remotionStep::Int64`: Number of consecutive iterations a component must be unused
  before it is eligible for removal from the bundle (default: `50`).

## Regularization parameter `t`
- `t::Float64`: Current value of the regularization/step-size parameter, used as the
  weight of the quadratic proximal term in the Dual Master Problem objective (default: `100.0`).
- `t_max::Int64`: Upper bound on `t`; increments are clamped to this value (default: `10000`).
- `t_min::Float64`: Lower bound on `t`; decrements are clamped to this value (default: `1e-5`).
- `t_incr::Float64`: Multiplicative factor for incrementing `t`: `t ← t * t_incr`.
  Must be greater than 1 (default: `1.1`).
- `t_decrement::Float64`: Multiplicative factor for decrementing `t`: `t ← t * t_decrement`.
  Must satisfy `0 < t_decrement < 1` (default: `0.9`).
- `t_star::Float64`: Reference step-size used in certain t-strategies and in the stopping
  criterion, in place of the current `t`. Kept fixed throughout the run (default: `100000.0`).

## Serious Step / Null Step decision
- `m1::Float64`: Threshold for the Serious Step acceptance condition:
  a trial point is accepted if `f(z_new) - f(zS) ≥ m1 * vStar`.
  Must satisfy `m1 ∈ [0, 1)` (default: `0.01`).
- `minSS::Int64`: Minimum number of consecutive Serious Steps required before an
  increment of `t` is considered by the t-strategy (default: `1`).

## Stopping criterion
- `ϵ::Float64`: Tolerance for the stopping criterion:
  the algorithm stops when `t_star * ‖w‖² + αᵀθ ≤ ϵ * (max(0, f(zS)) + 1)` (default: `1e-6`).

## Bundle size
- `max_β_size::Int`: Maximum number of bundle components (cutting planes) to retain.
  When exceeded, the least useful components are removed (default: `200`).

## Long-term t-strategy parameters
- `tSPar2::Float32`: Scaling threshold used in the long-term t-strategies
  (`soft_long_term_t_strategy`, `hard_long_term_t_strategy`, `balancing_long_term_t_strategy`)
  to gate increment and decrement decisions (default: `0.01`).

## Diagnostics
- `log::Bool`: If `true`, diagnostic information is recorded during execution (default: `false`).
"""
Base.@kwdef mutable struct BundleParameters
    # --- Iteration control ---
    maxIt::Int64 = 1000          # Maximum number of bundle iterations
    remotionStep::Int64 = 50     # Consecutive unused iterations before a component is removed

    # --- Regularization parameter t ---
    t::Float64 = 100.0           # Current regularization / step-size parameter
    t_max::Int64 = 10000         # Upper bound on t (increments are clamped here)
    t_min::Float64 = 1e-5        # Lower bound on t (decrements are clamped here)
    t_incr::Float64 = 1.1        # Multiplicative increment factor (must be > 1)
    t_decrement::Float64 = 0.9   # Multiplicative decrement factor (must be in (0, 1))
    t_star::Float64 = 100000.0   # Fixed reference step-size for stopping criterion and some t-strategies

    # --- Serious Step / Null Step decision ---
    m1::Float64 = 0.01           # SS acceptance threshold ∈ [0, 1): f(z_new) - f(zS) ≥ m1 * vStar
    minSS::Int64 = 1             # Minimum consecutive SS before an increment of t is triggered

    # --- Stopping criterion ---
    ϵ::Float64 = 1e-6            # Optimality tolerance: stop when the DMP gap is below ϵ * (|f(zS)| + 1)

    # --- Bundle size ---
    max_β_size::Int = 200        # Maximum number of active bundle components

    # --- Long-term t-strategy parameters ---
    tSPar2::Float32 = 0.01       # Scaling threshold for long-term t-strategy gating conditions

    # --- Diagnostics ---
    log::Bool = false            # If true, record diagnostic logs during execution
end