"""
    KRLS(y, x, K, M, λ, s2n, forgetting)

Kernel Recursive Least Squares (Tracker) algorithm with a fixed budget size via growing and pruning.

# Arguments
- `y::FIVector`
- `x::FIMatrix`
- `K::Kernel`
- `M::Integer`
- `λ::FI = 1`
- `s2n::FI = 1`
- `forgetting::String = "B2P"`

# Examples
```julia
2 + 3
```
"""
function KRLS(y::FIVector,
              x::FIMatrix,
              K::Kernel,
              M::Integer = size(y),
              λ::FI = 1,
              s2n::FI = 1,
              forgetting::String = "B2P")

    @assert s2n > 0 "Observation noise (regularization), s2n, must be strictly positive."
    @assert λ <= 1 && λ > 0 "Forgetting factor, λ, must be ]0;1]."
    @assert M <= length(y) && M > 0 "Budget size, M, must be [1;T]."
    @assert (forgetting == "B2P" || forgetting == "UI") "Input, forget, must be B2P or UI."

    T  = length(y)
    ŷ  = zeros(T)
    σ² = zeros(T)
    basis = [1]
    xb = x[basis, :]

    # Initialization
    ktt = kernel(x[1:1, :], K)[1] + eps()
    μ = [y[1] * ktt]  / (s2n + ktt) # predictive mean at time 1
    Σ = ktt - ktt^2 / (s2n + ktt)
    Q = reshape([1/ktt], 1, 1)
    s0n = y[1]^2 / (s2n + ktt)
    s0d = 1.0
    s02 = s0n / s0d
    σ²[1] = s02 * s2n              # predictive variance at time 1
    m = 1

    for t in 2:T
        # Forget
        forget!(μ, Σ, λ, kernel(xb, K), forgetting)

        ktt = kernel(x[t:t, :], K)[1] + eps()
        kbt = kernel(xb, x[t:t, :], K)

        q = vec(Q * kbt)
        h = vec(Σ * q)

        # Projection uncertainty
        γ² = ktt - dot(kbt, q)
        #γ² < 0.0 && (γ² = eps()) # Ensure that gamma squared is not negative
        
        # Noiseless prediction variance
        s2f = γ² + dot(q, h)
        #s2f < 0.0 && (s2f = eps()) # Ensure that s2f is not negative

        # Predictive mean
        ȳ = dot(q, μ)

        # Predictive variance
        s2y = s2n + s2f

        ŷ[t]  = ȳ
        σ²[t] = s2y * s02

        # Observe yₜ

        # Estimate of s02 via Maximum Likelihood
        s0n = s0n + λ * (y[t] - ȳ) / s2y
        s0d = s0d + λ
        s02 = s0n / s0d

        # Get posteriors of N(f|μ, Σ)
        push!(μ, ȳ)
        p = vcat(h, s2f)
        μ = μ + (y[t] - ȳ)/s2y * p
        s = reshape(p, :, 1)
        Σ = hcat(vcat(Σ, h'), s)
        Σ = Σ - (p * p') / s2y
        
        # Ensure that gamma is not too small
        if γ² < (10*eps())
            μ  = μ[1:end-1]
            Σ  = Σ[1:end-1, 1:end-1]
        else
            # Dictionary update
            push!(basis, t)
            xb = x[basis, :]
            m = m + 1

            p = vcat(q, [-1.0])
            Q = hcat(vcat(Q, zeros(1, m - 1)), zeros(m, 1)) + p * p' / γ²
            
            if m > M
                # MSE pruning criterion
                errors = ((Q * μ) ./ diag(Q)).^2
                dd, r = findmin(errors)

                # Remove element r
                d  = setdiff(1:m, r)
                Qs = Q[d, :]
                qs = Q[r, r]
                Q  = Q[d, d]
                Q  = Q - (Qs * Qs')/qs
                μ  = μ[d]
                Σ  = Σ[d, d]
                m  = m - 1
                xb = xb[d, :]
                basis = basis[d]
            end
        end
    end
    return (predictions = ŷ, variances = σ², basis = basis, xb = xb, μ = μ, Σ = Σ, Q = Q)
end

function KRLS(y::FIVector,
              x::FIVector,
              K::Kernel,
              M::Integer = size(y),
              λ::FI = 1,
              s2n::FI = 1,
              forgetting::String = "B2P")
    KRLS(y, reshape(x, :, 1), K, M, λ, s2n, forgetting)
end

function forget!(μ, Σ, λ, kb, forgetting)
    if forgetting == "B2P" 
        Σ = Σ * λ .+ kb * (1 - λ)
        μ = sqrt(λ) * μ
    elseif forgetting == "UI"
        Σ = Σ / λ
    end
end

predict(μ::Vector, Q::Matrix, x::FIMatrix, xb::FIMatrix, K::Kernel) = kernel(xb, x, K)' * Q * μ