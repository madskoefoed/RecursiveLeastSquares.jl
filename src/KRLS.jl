using Base: AbstractFloat
using LinearAlgebra

function kernel(x::Matrix{TYPE}, x₀::Matrix{TYPE}, lengthscale = 1.0) where TYPE <: AbstractFloat
    N1, D1 = size(x)
    N2, D2 = size(x₀)
    @assert D1 == D2 "The number of columns of x and x₀ do not match."
    k = zeros(N1, N2)
    for i1 in 1:N1
        for i2 in 1:N2
            k[i1, i2] = exp(-sum((x[i1, :] .- x₀[i2, :]).^2)/(2*lengthscale^2))
        end
    end
    return k
end
kernel(x::Matrix{TYPE}, σ = 1.0) where TYPE <: AbstractFloat = kernel(x, x, σ)
#kernel(x::Vector{TYPE}, σ = 1.0) where TYPE <: AbstractFloat = kernel(reshape(x, 1, length(x)), reshape(x, 1, length(x)), σ)

function KRLS(y::Vector{TYPE}, x::Matrix{TYPE}; λ = 0.99, s2n = 1.0, lengthscale = 1.0) where TYPE <: AbstractFloat

    @assert λ <= 1 && λ > 0 "Forgetting factor, λ, must be ]0;1]."

    T, D = size(x)
    pred = zeros(T)
    err  = zeros(T)

    # Initialization
    k̄ = kernel(x[1, :], lengthscale)[1] # scalar
    μ = [y[1] * k̄]                      # vector
    Σ = k̄ - k̄^2 / (s2n + k̄)             # scalar
    Q = 1/k̄                             # scalar

    pred = zeros(T)
    err  = zeros(T)

    err[1] = y[1]

    for t in 2:T
        #println("t = ", t)
        # Kernel
        k̄ = kernel(x[t, :], lengthscale)[1]
        K = kernel(x[1:t-1, :], lengthscale)
        k = kernel(x[1:t-1, :], reshape(x[t, :], 1, D), lengthscale)

        # Predict
        q = vec(Q * k)
        h = vec(Σ * q)
        ȳ = dot(q, μ)
        γ² = k̄ - dot(k, q)
        s2f = γ² + dot(q, h)
        s2y = s2n + s2f

        pred[t] = ȳ

        # Get posteriors of N(f|μ, Σ)
        push!(μ, ȳ)
        μ = μ + (y[t] - ȳ)/s2y * vcat(h, s2f)
        s = repeat(vcat(h, s2f), 1, 1)
        Σ = hcat(vcat(Σ, h'), s)
        Σ = Σ - (s * s') / s2y
        s = vcat(q, [-1.0])
        Q = hcat(vcat(Q, zeros(1, t-1)), zeros(t, 1)) + s * s' / γ²
        
    end
    return (μ = μ, Σ = Σ, errors = err, predictions = pred)
end

T = 1000;
x = hcat(ones(T), 1:T);
b = hcat(repeat([1, 1, 1, 1], inner = 250), repeat([-1], 1000));
y = vec(sum(sin.((x) * 0.01) .* b; dims = 2)) .+ randn(T) * 0.25;

model = KRLS(y, x; λ = 0.98, s2n = 0.1, lengthscale = 10)

using Plots
plot(y)
plot!(model.predictions)