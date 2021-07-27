using Base: Real
abstract type Kernel end

mutable struct Linear{T} <: Kernel where T<:Real
    σ::T
    function Linear(σ::T) where T<:Real
        @assert σ > 0 "The output variance, σ, must be positive."
        new(σ)
    end
end

mutable struct RBF{T} <: Kernel where T<:Real
    l::T
    σ::T
    function RBF(l::T, σ::T) where T<:Real
        @assert σ > 0 "The output variance, σ, must be positive."
        @assert l > 0 "The lengthscale, l, must be positive."
        new(l, σ)
    end
end

mutable struct RationalQuadratic{T} <: Kernel where T<:Real
    a::T
    l::T
    σ::T
    function RationalQuadratic(a::T, l::T, σ::T) where T<:Real
        @assert a > 0 "The relative weighting of different lengthscales, a, must be positive."
        @assert σ > 0 "The output variance, σ, must be positive."
        @assert l > 0 "The lengthscale, l, must be positive."
        new(a, l, σ)
    end
end

mutable struct Periodic{T} <: Kernel where T<:Real
    p::T
    l::T
    σ::T
    function Periodic(p::T, l::T, σ::T) where T<:Real
        @assert p > 0 "The period, p, must be positive."
        @assert σ > 0 "The output variance, σ, must be positive."
        @assert l > 0 "The lengthscale, l, must be positive."
        new(p, l, σ)
    end
end