function GP(y::Vector{TYPE}, x::Matrix{TYPE}, kernel_struct::Kernel; s2n = 1.0) where TYPE <: AbstractFloat

    @assert s2n > 0 "Output noise, s2n, must be positive."

    T, D = size(x)
    pred = zeros(T)
    err  = zeros(T)

    # Initialization
    K = kernel(x, kernel_struct)

    P = inv(K + I*s2n)
    μ  = K * P * y
    Σ = K - K * P * K

    return (μ = μ, Σ = Σ, errors = err, predictions = pred)
end