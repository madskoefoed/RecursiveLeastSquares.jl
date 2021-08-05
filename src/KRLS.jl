function KRLS(y::FIVector,
              x::FIMatrix,
              kernel_struct::Kernel,
              M::Integer = size(y),
              λ::FI = 1,
              s2n::FI = 1,
              jitter::AbstractFloat = 1e-8)

    @assert λ <= 1 && λ > 0 "Forgetting factor, λ, must be ]0;1]."
    @assert M <= length(y) && M > 0 "Budget size, M, must be [1;T]."

    T  = length(y)
    ŷ  = zeros(T)
    σ² = zeros(T)
    basis = Int[]
    push!(basis, 1)
    xb = x[basis, :]

    # Initialization
    ktt = kernel(x[[1], :], kernel_struct)[1] # scalar
    ktt = ktt + jitter
    μ = [y[1] * ktt]  / (s2n + ktt)           # predictive mean at time 1
    Σ = ktt - ktt^2 / (s2n + ktt)             # scalar
    Q = 1/ktt                                 # scalar
    σ²[1] = y[1]^2 / (s2n + ktt) + s2n        # predictive variance at time 1

    for t in 2:T
        # Kernel
        ktt = kernel(x[[t], :], kernel_struct)[1] + jitter
        kbt = kernel(xb, x[[t], :], kernel_struct)

        # Forget
        # B2P - Back 2 Prior
        #Σ = Σ * λ .+ kernel(xb, kernel_struct) * (1 - λ)
        #μ = sqrt(λ) * μ
        # Uncertainty injection
        #Σ = Σ / λ

        # Predictive mean
        q = vec(Q * kbt)
        ȳ = dot(q, μ)

        # Predictive variance
        γ² = ktt - dot(kbt, q)
        #γ² < 0.0 && (γ² = jitter) # Ensure that gamma squared is not negative
        h = vec(Σ * q)
        s2f = γ² + dot(q, h)
        #s2f < 0.0 && (s2f = jitter) # Ensure that s2f is not negative
        s2y = s2n + s2f

        ŷ[t]  = ȳ
        σ²[t] = s2y

        # Get posteriors of N(f|μ, Σ)
        push!(μ, ȳ)
        μ = μ + (y[t] - ȳ)/s2y * vcat(h, s2f)
        s = repeat(vcat(h, s2f), 1, 1)
        Σ = hcat(vcat(Σ, h'), s)
        Σ = Σ - (s * s') / s2y
        s = vcat(q, [-1.0])
        Q = hcat(vcat(Q, zeros(1, t-1)), zeros(t, 1)) + s * s' / γ²
        
        # Dictionary update
        push!(basis, t)
        xb = x[basis, :]
    end
    return (predictions = ŷ, variances = σ², basis = basis)
end

function KRLS(y::FIVector,
              x::FIVector,
              kernel_struct::Kernel,
              M::Integer = size(y),
              λ::FI = 1,
              s2n::FI = 1,
              jitter::AbstractFloat = 1e-8)
    KRLS(y, reshape(x, :, 1), kernel_struct, M, λ, s2n, jitter)
end