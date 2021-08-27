function RGP(y::REALVEC,
             x::REALMAT,
             K::Kernel,
             M::Integer = size(y),
             s2n::Real = 1)

    @assert M <= length(y) && M > 0 "Budget size, M, must be [1;T]."
    @assert s2n > 0 "Output noise, s2n, must be positive."

    T, D = size(x)

    basis = collect(1:M)

    # Run standard GP
    μ, C, _ = GP(y[basis], x[basis, :], x[basis, :], K, s2n)

    m = zeros(M)
    #C = diagm(ones(D) * 1_000)

    for t in 1:T

        # Calculate kernel
        ktt = kernel(x[1:t, :], x[1:t, :], K)
        kbt = kernel(x[basis, :], x[1:t, :], K)
        kbb = kernel(x[basis, :], x[basis, :], K)
        # Calculate gain matrix J
        J = inv(kbb) * kbt
        # Calculate mean
        #μ = m .+ J .* (μ - m)
        # calculate covariance matrices
        B = ktt - J' * kbt
        println("Size of J: $(size(J))")
        println("Size of C: $(size(C))")
        println("Size of B: $(size(B))")
        CP = B + J' * C * J
#        C = [C (C * J'); (J * C) ]


    end

    return (predictions = μ, variances = 0, loglikelihood = 0)
end

#function RGP(y::REALVEC,
#             x::REALVEC,
#             K::Kernel,
#             M::Integer = size(y),
#             s2n::Real = 1)
#   RGP(y, reshape(x, :, 1), K, s2n)
#end