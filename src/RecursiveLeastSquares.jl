module RecursiveLeastSquares

using LinearAlgebra
using Distributions

const REALVEC{T<:Real} = Vector{T}
const REALMAT{T<:Real} = Matrix{T}

include("types.jl")
include("kernels.jl")
include("RLS.jl")
include("KRLS.jl")
include("GP.jl")

# export types
export Linear, RBF, RationalQuadratic, Periodic, LocallyPeriodic

# Export functions
export RLS, KRLS, GP

end
