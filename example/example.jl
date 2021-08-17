M = 6;
c = 1e-5;
λ = 0.99

# Function
f(x) = sin.(x) .* x

# Training
x = [1, 3, 5, 6, 7, 8, 9, 10];
y = f(x);

# Test
X = collect(range(0, 10, length = 301));
Y = f(X);

# Kernel
k_krls = RBF(10, 1);
#k_gp   = RBF(2, 1);

#m_rls  = RLS(y, x, λ);
m_krls = KRLS(y, x, k_krls, M, λ, c, "B2P");
#m_gp   = GP(y, x, x, k_gp, c);

using Plots
plot(X, Y, label = "f(x)", color = "black", legend = :topleft)
scatter!(x, y, label = "Observations", color = "red")
plot!(x, m_krls.predictions, label = "KRLS", color = "green")
scatter!(x[m_krls.basis], y[m_krls.basis], label = "Dictionary", color = "blue")

ŷ = predictive_mean(m_krls.μ, m_krls.Q, reshape(X, :, 1), m_krls.xb, k_krls)
plot!(X, ŷ, label = "Prediction", color = "pink")