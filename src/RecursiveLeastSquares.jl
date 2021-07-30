#module RecursiveLeastSquares

using Base: AbstractFloat
using LinearAlgebra
using Distributions

include("src/types.jl")
include("src/kernels.jl")
include("src/RLS.jl")
include("src/KRLS.jl")
include("src/GP.jl")

include("example/example.jl")

#end
