function KRLS(y::Vector{TYPE}, x::Matrix{TYPE}, kernel_struct::Kernel; λ = 0.99, s2n = 1.0) where TYPE <: AbstractFloat

    @assert λ <= 1 && λ > 0 "Forgetting factor, λ, must be ]0;1]."

    T, D = size(x)
    pred = zeros(T)
    err  = zeros(T)

    # Initialization
    k̄ = kernel(x[1, :], kernel_struct)[1] # scalar
    μ = [y[1] * k̄]                      # vector
    Σ = k̄ - k̄^2 / (s2n + k̄)             # scalar
    Q = 1/k̄                             # scalar

    pred = zeros(T)
    err  = zeros(T)

    err[1] = y[1]

    for t in 2:T

        # Kernel
        k̄ = kernel(x[t, :], kernel_struct)[1]
        k = kernel(x[1:t-1, :], reshape(x[t, :], 1, D), kernel_struct)

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