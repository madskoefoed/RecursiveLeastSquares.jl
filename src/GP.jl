function GP(y::FIVector,
            x::FIMatrix,
            z::FIMatrix,
            kernel_struct::Kernel,
            s2n::FI)

    @assert s2n > 0 "Output noise, s2n, must be positive."
    T, D = size(x)

    # Kernel
    K = kernel(x, kernel_struct)    # Training
    k = kernel(x, z, kernel_struct) # Training vs test
    # Cholesky
    L = cholesky(K + I*s2n).L
    # Predictive mean
    α = L' \ (L \ y)
    μ = k' * α
    # Predictive variance
    v = L \ k
    Σ = kernel(z, z, kernel_struct) - v' * v
    # Log-likelihood
    LL = -dot(y, α)/2 - sum(diag(L)) - T/2 * log(2π)

    return (predictions = μ, variances = diag(Σ), loglikelihood = LL)
end

function GP(y::FIVector,
            x::FIVector,
            z::FIVector,
            kernel_struct::Kernel,
            s2n::FI = 1.0)
    GP(y, reshape(x, :, 1), reshape(z, :, 1), kernel_struct, s2n)
end