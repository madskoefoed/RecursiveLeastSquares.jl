function GP(y::REALVEC,
            x::REALMAT,
            z::REALMAT,
            K::Kernel,
            s2n::Real)

    @assert s2n > 0 "Output noise, s2n, must be positive."
    T, D = size(x)

    # Kernel
    kbb = kernel(x, K)    # Training
    kbt = kernel(x, z, K) # Training vs test
    # Cholesky
    L = cholesky(kbb + I*s2n).L
    # Predictive mean
    α = L' \ (L \ y)
    μ = kbt' * α
    # Predictive variance
    v = L \ kbt
    Σ = kernel(z, z, kbb) - v' * v
    # Log-likelihood
    LL = -dot(y, α)/2 - sum(diag(L)) - T/2 * log(2π)

    return (predictions = μ, covariance = Σ, loglikelihood = LL)
end

function GP(y::REALVEC,
            x::REALVEC,
            z::REALVEC,
            K::Kernel,
            s2n::Real = 1.0)
    GP(y, reshape(x, :, 1), reshape(z, :, 1), K, s2n)
end