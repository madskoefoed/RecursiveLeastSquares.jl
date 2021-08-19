module RecursiveLeastSquares

using LinearAlgebra
using Distributions

const Fl = AbstractFloat
const FIVector{T<:FI} = Vector{T}
const FIMatrix{T<:FI} = Matrix{T}

include("types.jl")
include("kernels.jl")
include("RLS.jl")
include("KRLS.jl")
include("GP.jl")

# export types
export Kernel

# Export functions
export RLS, KRLS, GP

end
