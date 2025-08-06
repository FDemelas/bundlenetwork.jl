"""
Abstract type to model all the bundles of this package.
"""
abstract type AbstractBundle end

"""
Abstract type to encode all the bundles that solve the Dual Master Problem to obtain the new searching direction.
"""
abstract type DualBundle <: AbstractBundle end


"""
Abstract Bundle Factory to modelize all the Bundle Factories.
Factories are used only for the construction and initialization of the Bundles.
"""
abstract type AbstractBundleFactory end


"""
Abstract Bundle to modelize all the Soft Bundle in which both t-strategies and Dual Master Problem are substituted by a neural network.
"""
abstract type AbstractSoftBundle <: AbstractBundle end

struct SoftBundleFactory <: AbstractBundleFactory end
struct tLearningBundleFactory <: AbstractBundleFactory end
struct VanillaBundleFactory <: AbstractBundleFactory end
