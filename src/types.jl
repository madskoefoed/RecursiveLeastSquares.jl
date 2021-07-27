abstract type Kernel end

mutable struct RBF{T} <: Kernel where T<:Real
    l::T
    σᶠ::T
end

mutable struct Linear{T} <: Kernel where T<:Real
    σᶠ::T
end