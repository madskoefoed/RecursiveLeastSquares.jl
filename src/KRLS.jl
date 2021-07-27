function KRLS(y::Vector{TYPE}, x::Matrix{TYPE}, kerneltype::Kernel; λ = 0.99, s2n = 1.0) where TYPE <: AbstractFloat

    @assert λ <= 1 && λ > 0 "Forgetting factor, λ, must be ]0;1]."

    T, D = size(x)
    pred = zeros(T)
    err  = zeros(T)

    # Initialization
    k̄ = kernel(x[1, :], kerneltype)[1] # scalar
    μ = [y[1] * k̄]                      # vector
    Σ = k̄ - k̄^2 / (s2n + k̄)             # scalar
    Q = 1/k̄                             # scalar

    pred = zeros(T)
    err  = zeros(T)

    err[1] = y[1]

    for t in 2:T
        #println("t = ", t)
        # Kernel
        k̄ = kernel(x[t, :], kerneltype)[1]
        K = kernel(x[1:t-1, :], kerneltype)
        k = kernel(x[1:t-1, :], reshape(x[t, :], 1, D), kerneltype)

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
k = RBF(1.0, 5.0);

model = KRLS(y, x, k; λ = 0.98, s2n = 0.1)

using Plots
plot(y)
plot!(model.predictions)