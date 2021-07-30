function kernel(x::Matrix{TYPE}, x₀::Matrix{TYPE}, kernel_struct::Kernel) where TYPE <: AbstractFloat
    @assert size(x, 2) == size(x₀, 2) "The number of columns of x and x₀ do not match."
    # Dispatch to kernel
    K = kernel_calculation(x, x₀, kernel_struct::Kernel)
    return K
end
kernel(x::Matrix{TYPE}, kernel_struct::Kernel) where TYPE <: AbstractFloat = kernel(x, x, kernel_struct)
kernel(x::Vector{TYPE}, kernel_struct::Kernel) where TYPE <: AbstractFloat = kernel(reshape(x, 1, length(x)), reshape(x, 1, length(x)), kernel_struct)

# White Noise
function kernel_calculation(x, x₀, k::WhiteNoise)
    K = [i1 == i2 ? 1.0 : 0.0 for i1 in 1:size(x, 1), i2 in 1:size(x₀, 1)]
    return K * k.σ
end

# Linear
function kernel_calculation(x, x₀, k::Linear)
    K = [dot(x[i1, :], x[i2, :]) for i1 in 1:size(x, 1), i2 in 1:size(x₀, 1)]
    return K * k.σ
end

# Radial Basis Function (Squared Exponential)
function kernel_calculation(x, x₀, k::RBF)
    K = zeros(size(x, 1), size(x₀, 1))
    for i1 in 1:size(x, 1)
        for i2 in 1:size(x₀, 1)
            K[i1, i2] = k.σ * exp(-sum((x[i1, :] .- x₀[i2, :]).^2)/(2*k.l^2))
        end
    end
    return K
end

# Rational Quadratic
function kernel_calculation(x, x₀, k::RationalQuadratic)
    K = zeros(size(x, 1), size(x₀, 1))
    for i1 in 1:size(x, 1)
        for i2 in 1:size(x₀, 1)
            K[i1, i2] = k.σ * (1.0 + sum((x[i1, :] .- x₀[i2, :])^2)/(2 * k.a * k.l^2))^-k.a
        end
    end
    return K
end

# Periodic
function kernel_calculation(x, x₀, k::Periodic)
    K = zeros(size(x, 1), size(x₀, 1))
    for i1 in 1:size(x, 1)
        for i2 in 1:size(x₀, 1)
            K[i1, i2] = k.σ * exp(-2 * sin(sum(abs.(x[i1, :] .- x₀[i2, :])))^2/k.l^2)
        end
    end
    return K
end

# Locally Periodic
function kernel_calculation(x, x₀, k::LocallyPeriodic)
    K = zeros(size(x, 1), size(x₀, 1))
    for i1 in 1:size(x, 1)
        for i2 in 1:size(x₀, 1)
            K[i1, i2] = k.σ * exp(-2 * sin(sum(abs.(x[i1, :] .- x₀[i2, :])))^2/k.l^2) * exp(-sum((x[i1, :] .- x[i2, :].^2)/(2*k.l^2)))
        end
    end
    return K
end