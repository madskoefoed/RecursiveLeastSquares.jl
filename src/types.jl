abstract type Kernel end

mutable struct Linear{FI} <: Kernel
    σ::FI
    function Linear(σ::FI)
        @assert σ > 0 "The output variance, σ, must be positive."
        new{FI}(σ)
    end
end

mutable struct RBF{FI} <: Kernel
    l::FI
    σ::FI
    function RBF(l::FI, σ::FI)
        @assert σ > 0 "The output variance, σ, must be positive."
        @assert l > 0 "The lengthscale, l, must be positive."
        new{FI}(l, σ)
    end
end

mutable struct RationalQuadratic{FI} <: Kernel
    a::FI
    l::FI
    σ::FI
    function RationalQuadratic(a::FI, l::FI, σ::FI)
        @assert a > 0 "The relative weighting of different lengthscales, a, must be positive."
        @assert σ > 0 "The output variance, σ, must be positive."
        @assert l > 0 "The lengthscale, l, must be positive."
        new{FI}(a, l, σ)
    end
end

mutable struct Periodic{FI} <: Kernel
    p::FI
    l::FI
    σ::FI
    function Periodic(p::FI, l::FI, σ::FI)
        @assert p > 0 "The period, p, must be positive."
        @assert σ > 0 "The output variance, σ, must be positive."
        @assert l > 0 "The lengthscale, l, must be positive."
        new{FI}(p, l, σ)
    end
end

mutable struct LocallyPeriodic{FI} <: Kernel
    p::FI
    l::FI
    σ::FI
    function LocallyPeriodic(p::FI, l::FI, σ::FI)
        @assert p > 0 "The period, p, must be positive."
        @assert σ > 0 "The output variance, σ, must be positive."
        @assert l > 0 "The lengthscale, l, must be positive."
        new{FI}(p, l, σ)
    end
end