abstract type Kernel end

mutable struct Linear <: Kernel
    σ::Real
    function Linear(σ::Real)
        @assert σ > 0 "The output variance, σ, must be positive."
        new(σ)
    end
end

mutable struct RBF <: Kernel
    l::Real
    σ::Real
    function RBF(l::Real, σ::Real)
        @assert σ > 0 "The output variance, σ, must be positive."
        @assert l > 0 "The lengthscale, l, must be positive."
        new(l, σ)
    end
end

mutable struct RationalQuadratic <: Kernel
    a::Real
    l::Real
    σ::Real
    function RationalQuadratic(a::Real, l::Real, σ::Real)
        @assert a > 0 "The relative weighting of different lengthscales, a, must be positive."
        @assert σ > 0 "The output variance, σ, must be positive."
        @assert l > 0 "The lengthscale, l, must be positive."
        new(a, l, σ)
    end
end

mutable struct Periodic <: Kernel
    p::Real
    l::Real
    σ::Real
    function Periodic(p::Real, l::Real, σ::Real)
        @assert p > 0 "The period, p, must be positive."
        @assert σ > 0 "The output variance, σ, must be positive."
        @assert l > 0 "The lengthscale, l, must be positive."
        new(p, l, σ)
    end
end

mutable struct LocallyPeriodic <: Kernel
    p::Real
    l::Real
    σ::Real
    function LocallyPeriodic(p::Real, l::Real, σ::Real)
        @assert p > 0 "The period, p, must be positive."
        @assert σ > 0 "The output variance, σ, must be positive."
        @assert l > 0 "The lengthscale, l, must be positive."
        new(p, l, σ)
    end
end

abstract type Model end

mutable struct RLS{T <: AbstractFloat} <: Model
    y::Vector{T}
    w::Vector{T}
    λ::Real
    ŷ::Vector{T}
    σ::Vector{T}
end

mutable struct KRLS{T <: AbstractFloat} <: Model
    y::REALVEC
    x::REALMAT
    kernel::Kernel
    λ::Real
    budget::Integer
    mode::String
    μ::Vector{T}
    Σ::Matrix{T}
    Q::Matrix{T}
    ŷ::Vector{T}
    σ::Vector{T}
    #function KRLS(λ::Real, budget::Integer, forgetting::String)
    #    @assert λ <= 1 && λ > 0 "Forgetting factor, λ, must be ]0;1]."
    #    @assert budget > 0 "Budget size must be a positive integer."
    #    @assert (forgetting == "B2P" || forgetting == "UI") "Forgetting mode must be either B2P (back 2 prior) or UI (uncertainty injection)."
    #    new(λ, budget, forgetting)
    #end
end