function KRLS(y::FIVector,
              x::FIMatrix,
              kernel_struct::Kernel,
              λ::FI = 0.99,
              s2n::FI = 1.0)

    @assert λ <= 1 && λ > 0 "Forgetting factor, λ, must be ]0;1]."

    T, D = size(x)
    pred = zeros(T)
    σ²   = zeros(T)
    LL   = zeros(T)

    # Initialization
    xt = reshape(x[1, :], 1, D)
    k̄ = kernel(xt, kernel_struct)[1] # scalar
    μ = [y[1] * k̄]                   # vector
    Σ = k̄ - k̄^2 / (s2n + k̄)          # scalar
    Q = 1/k̄                          # scalar

    for t in 2:T

        xt = reshape(x[t, :], 1, D)
        # Kernel
        k̄ = kernel(xt, kernel_struct)[1]
        k = kernel(x[1:t-1, :], xt, kernel_struct)

        # Predict
        q = vec(Q * k)
        h = vec(Σ * q)
        ȳ = dot(q, μ)
        γ² = k̄ - dot(k, q)
        s2f = γ² + dot(q, h)
        s2y = s2n + s2f
        pred[t] = ȳ
        σ²[t]  = s2y
        #LL[t] = logpdf(Normal(pred[t], s2y), y[t])

        # Get posteriors of N(f|μ, Σ)
        push!(μ, ȳ)
        μ = μ + (y[t] - ȳ)/s2y * vcat(h, s2f)
        s = repeat(vcat(h, s2f), 1, 1)
        Σ = hcat(vcat(Σ, h'), s)
        Σ = Σ - (s * s') / s2y
        s = vcat(q, [-1.0])
        Q = hcat(vcat(Q, zeros(1, t-1)), zeros(t, 1)) + s * s' / γ²
        
    end
    LL = cumsum(LL) ./ (1:T)
    return (predictions = pred, variances = σ², loglikelihood = LL)
end

function KRLS(y::FIVector,
              x::FIVector,
              kernel_struct::Kernel,
              λ::FI = 0.99,
              s2n::FI = 1.0)
    KRLS(y, reshape(x, :, 1), kernel_struct, λ, s2n)
end