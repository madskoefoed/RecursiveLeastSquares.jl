"""
    RLS(y, x, λ)

Recursive Least Squares algorithm.

# Arguments
- `y::FIVector`
- `x::FIMatrix`
- `λ::FI = 1`

# Examples
```julia
2 + 3
```
"""
function RLS(y::FIVector, 
             x::FIMatrix,
             λ::FI = 0.99)

    @assert λ <= 1 && λ > 0 "Forgetting factor, λ, must be ]0;1]."

    T, D = size(x)

    w = zeros(T + 1, D)
    P = zeros(T + 1, D, D)
    μ = zeros(T)
    σ² = zeros(T)

    P[1, :, :] = Matrix(1000.0 * I, D, D)
    for t in 1:T
        # Predict
        μ[t] = predict(x[t, :], w[t, :])
        
        # Update
        w[t + 1, :], P[t + 1, :, :] = update(y[t], x[t, :], w[t, :], P[t, :, :], μ[t], λ)
    end
    return (predictions = μ, variances = σ², coefficients = w, covariance = P)
end

function RLS(y::FIVector,
             x::FIVector,
             λ::FI = 0.99)
    RLS(y, reshape(x, :, 1), λ)
end

function predict(x, w)
    return dot(x, w)
end

function update(y, x, w, P, μ, λ)
    K = P * x / (λ + x' * P * x)
    w = w + (y - μ) * K
    P = (I - K * x') * P / λ
    return (w, P)
end

predict(x::FIMatrix, w::Vector{<:AbstractFloat}) = x * w
predict(x::FIVector, w::Vector{<:AbstractFloat}) = dot(x, w)