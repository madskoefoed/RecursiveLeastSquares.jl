abstract type Kernel end

mutable struct Linear <: Kernel
    σ::FI
    function Linear(σ::FI)
        @assert σ > 0 "The output variance, σ, must be positive."
        new(σ)
    end
end

mutable struct RBF <: Kernel
    l::FI
    σ::FI
    function RBF(l::FI, σ::FI)
        @assert σ > 0 "The output variance, σ, must be positive."
        @assert l > 0 "The lengthscale, l, must be positive."
        new(l, σ)
    end
end

mutable struct RationalQuadratic <: Kernel
    a::FI
    l::FI
    σ::FI
    function RationalQuadratic(a::FI, l::FI, σ::FI)
        @assert a > 0 "The relative weighting of different lengthscales, a, must be positive."
        @assert σ > 0 "The output variance, σ, must be positive."
        @assert l > 0 "The lengthscale, l, must be positive."
        new(a, l, σ)
    end
end

mutable struct Periodic <: Kernel
    p::FI
    l::FI
    σ::FI
    function Periodic(p::FI, l::FI, σ::FI)
        @assert p > 0 "The period, p, must be positive."
        @assert σ > 0 "The output variance, σ, must be positive."
        @assert l > 0 "The lengthscale, l, must be positive."
        new(p, l, σ)
    end
end

mutable struct LocallyPeriodic <: Kernel
    p::FI
    l::FI
    σ::FI
    function LocallyPeriodic(p::FI, l::FI, σ::FI)
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
    λ::FI
    ŷ::Vector{T}
    σ::Vector{T}
end

mutable struct KRLS{T <: AbstractFloat} <: Model
    y::FIVector
    x::FIMatrix
    kernel::Kernel
    λ::FI
    budget::Integer
    mode::String
    μ::Vector{T}
    Σ::Matrix{T}
    Q::Matrix{T}
    ŷ::Vector{T}
    σ::Vector{T}
    #function KRLS(λ::FI, budget::Integer, forgetting::String)
    #    @assert λ <= 1 && λ > 0 "Forgetting factor, λ, must be ]0;1]."
    #    @assert budget > 0 "Budget size must be a positive integer."
    #    @assert (forgetting == "B2P" || forgetting == "UI") "Forgetting mode must be either B2P (back 2 prior) or UI (uncertainty injection)."
    #    new(λ, budget, forgetting)
    #end
end

KRLS() = 
