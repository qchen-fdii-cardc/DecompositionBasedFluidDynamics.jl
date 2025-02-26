module DecompositionBasedFluidDynamics

using LinearAlgebra

# Write your package code here.
include("pod.jl")
include("dmd.jl")

export pod, reconstruct
export dmd, reconstruct_dmd
end
