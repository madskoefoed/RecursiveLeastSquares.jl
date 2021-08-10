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
        μ[t] = RLS_predict(x[t, :], w[t, :])
        
        # Update
        #K = P[t, :, :] * x[t, :] / (λ + x[t, :]' * P[t, :, :] * x[t, :])
        #w[t + 1, :] = w[t, :] + (y[t] - μ[t]) * K
        #P[t + 1, :, :] = (I - K * x[t, :]') * P[t, :, :] / λ
        w[t + 1, :], P[t + 1, :, :] = RLS_update(y[t], x[t, :], w[t, :], P[t, :, :], μ[t], λ)
    end
    return (predictions = μ, variances = σ², coefficients = w, covariance = P)
end

function RLS(y::FIVector,
             x::FIVector,
             λ::FI = 0.99)
    RLS(y, reshape(x, :, 1), λ)
end

function RLS_predict(x, w)
    return dot(x, w)
end

function RLS_update(y, x, w, P, μ, λ)
    K = P * x / (λ + x' * P * x)
    w = w + (y - μ) * K
    P = (I - K * x') * P / λ
    return (w, P)
end

