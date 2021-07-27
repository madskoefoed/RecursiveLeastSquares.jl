function kernel(x::Matrix{TYPE}, x₀::Matrix{TYPE}, kerneltype::Kernel) where TYPE <: AbstractFloat
    N1, D1 = size(x)
    N2, D2 = size(x₀)
    @assert D1 == D2 "The number of columns of x and x₀ do not match."
    # Dispatch to kernel
    K = kernel_calculation(x, x₀, kerneltype::Kernel)
    return K
end
kernel(x::Matrix{TYPE}, kerneltype::Kernel) where TYPE <: AbstractFloat = kernel(x, x, kerneltype)
kernel(x::Vector{TYPE}, kerneltype::Kernel) where TYPE <: AbstractFloat = kernel(reshape(x, 1, length(x)), reshape(x, 1, length(x)), kerneltype)

# Radial Basis Function
function kernel_calculation(x, x₀, k::RBF)
    K = zeros(N1, N2)
    for i1 in 1:N1
        for i2 in 1:N2
            K[i1, i2] = k.σᶠ * exp(-sum((x[i1, :] .- x₀[i2, :]).^2)/(2*k.l^2))
        end
    end
    return K
end

# Linear
function kernel_calculation(x, x₀, k::Linear)
    K = [dot(x[i1, :], x[i2, :]) for i1 in 1:N1, i2 in 1:N2]
    return K * k.σᶠ
end