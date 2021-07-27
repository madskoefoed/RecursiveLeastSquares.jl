function RLS(y::AbstractVector{TYPE}, x::AbstractMatrix{TYPE}; λ = 0.99) where TYPE <: AbstractFloat

    @assert λ <= 1 && λ > 0 "Forgetting factor, λ, must be ]0;1]."

    T, D = size(x)

    w = zeros(T + 1, D)
    P = zeros(T + 1, D, D)
    pred = zeros(T)
    err  = zeros(T)

    P[1, :, :] = Matrix(1000.0 * I, D, D)

    for t in 1:T
        # Predict
        pred[t] = dot(x[t, :], w[t, :])
        err[t]  = y[t] - pred[t]
        
        # Update
        K = P[t, :, :] * x[t, :] / (λ + x[t, :]' * P[t, :, :] * x[t, :])
        w[t + 1, :] = w[t, :] + err[t] * K
        P[t + 1, :, :] = (I - K * x[t, :]') * P[t, :, :] / λ
    end
    return (coefficients = w, covariance = P, errors = err, predictions = pred)
end

T = 1000;
x = hcat(ones(T), sin.((1:T) * 0.01));
b = hcat(repeat([1, 1, 1, 1], inner = 250), repeat([-1], 1000));
y = vec(sum(x .* b; dims = 2)) .+ randn(T) * 0.25;

model = RLS(y, x; λ = 0.98)

using Plots, StatsPlots
plot(model.coefficients, color = [:black :red])
plot!(b, color = [:black :red], linestyle = :dash)
plot(y)
plot!(model.predictions)