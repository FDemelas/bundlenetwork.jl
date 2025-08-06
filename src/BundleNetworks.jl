module BundleNetworks

using LinearAlgebra
using JuMP, HiGHS,Gurobi
using SparseArrays
using Flux, ChainRules, ChainRulesCore
using Instances
import Instances: LR, cpuInstanceMCND,sizeLM
using Random
using Statistics
using Zygote
using CUDA
import CUDA: CuArray, @ćuda
import Flux: gpu, cpu

CUDA.set_runtime_version!(v"12.1")
use_gpu = true #false # true
device = CUDA.functional() && use_gpu ? gpu : cpu

gap(a,b) = ( (b-a)/b)*100

include("Bundle/AbstractBundle.jl")

include("tStrategies/tStrategy.jl")

include("HyperParameters/BundleParameters.jl")
include("ObjectiveFunctions/AbstractConcave.jl")

include("Models/AbstractModel.jl")

include("ObjectiveFunctions/InnerLoss.jl")
include("ObjectiveFunctions/LagrangianMCND.jl")
include("ObjectiveFunctions/LagrangianGA.jl")
include("ObjectiveFunctions/LagrangianTUC.jl")

include("Auxiliary/instanceFeatures.jl")
include("Auxiliary/sparsemax.jl")

include("Models/Deviations.jl")

include("Bundle/SoftBundle.jl")

include("Models/tAndDirectionModels/AttentionModel.jl")

include("Models/tModels/tRecurrentMlp.jl")
include("Models/tModels/tRecurrentMlp2.jl")

include("Bundle/DualBundle.jl")
include("Bundle/VanillaBundle.jl")
include("Bundle/tLearningBundle.jl")
export create_NN

export initializeBundle
export tLearningBundleFactory,SoftBundleFactory,VanillaBundleFactory
export solve!

export test_local_retraining,test,training_loop,training_epoch
export sizeLM,  constructFunction,value_gradient,my_read_dat
export gap
export sizeInputSpace
end
