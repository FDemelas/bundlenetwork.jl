"""
Abstract type to model all the bundles of this package.
All concrete bundle types must subtype this. A bundle, in the context of 
bundle methods for nonsmooth optimization, stores the cutting-plane model 
(subgradient information) used to build the next iterate.
"""
abstract type AbstractBundle end

"""
Abstract type for bundles that solve a Dual Master Problem to compute 
the new search direction.
In bundle methods, the Dual Master Problem determines the next trial point 
by combining subgradient cuts. All dual-based bundle variants should subtype this.
"""
abstract type DualBundle <: AbstractBundle end

"""
Abstract type for Bundle Factories.
Factories are responsible solely for the construction and initialization 
of bundle objects. This separation of concerns keeps bundle logic independent 
from instantiation details, following the Factory design pattern.
"""
abstract type AbstractBundleFactory end

"""
Abstract type for Soft Bundles, where both the t-strategy and the 
Dual Master Problem solver are replaced by a neural network.
In a soft bundle, the neural network acts as an end-to-end policy:
it receives the current bundle state and directly outputs the next 
iterate, bypassing the classical optimization-based inner loop entirely.
"""
abstract type AbstractSoftBundle <: AbstractBundle end

"""
Factory for creating `SoftBundle` instances.
A `SoftBundle` replaces the classical Dual Master Problem and t-strategy 
with a neural network that jointly handles both roles.
"""
struct SoftBundleFactory <: AbstractBundleFactory end

"""
Factory for creating `tLearningBundle` instances.
A `tLearningBundle` keeps the classical Dual Master Problem structure 
but replaces only the t-strategy with a neural network (`nn_t_strategy`),
learning to adaptively tune the regularization/step-size parameter `t`.
"""
struct tLearningBundleFactory <: AbstractBundleFactory end

"""
Factory for creating `BatchedSoftBundle` instances.
A `BatchedSoftBundle` is a batched variant of the `SoftBundle`,
designed to handle multiple bundle instances in parallel (e.g., for 
training the neural network policy across several problem instances simultaneously).
"""
struct BatchedSoftBundleFactory <: AbstractBundleFactory end

"""
Factory for creating `VanillaBundle` instances.
A `VanillaBundle` is the classical bundle method implementation,
using a standard Dual Master Problem and a conventional (non-learned) t-strategy.
Use this as the baseline against which learned variants are compared.
"""
struct VanillaBundleFactory <: AbstractBundleFactory end