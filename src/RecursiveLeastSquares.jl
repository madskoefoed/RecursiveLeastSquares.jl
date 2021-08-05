#module RecursiveLeastSquares

using LinearAlgebra
using Distributions

const FI = Union{Integer, AbstractFloat}
const FIVector{T<:FI} = Vector{T}
const FIMatrix{T<:FI} = Matrix{T}

include("src/types.jl")
include("src/kernels.jl")
include("src/RLS.jl")
include("src/KRLS.jl")
include("src/GP.jl")

include("example/example.jl")

#end
