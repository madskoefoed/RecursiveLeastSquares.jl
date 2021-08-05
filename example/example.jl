M = 5;
c = 1e-5;
λ = 1.0

# Function
f(x) = sin.(x) .* x
# Training
x = [1, 3, 5, 6, 7, 8];
y = f(x);
# Test
X = collect(range(0, 10, length = 1000));
Y = f(X);

# Kernel
k_krls = RBF(2.0^-5, 1);
k_gp   = RBF(2.0^-8, 1);

m_rls  = RLS(y, x, λ);
m_krls = KRLS(y, x, k_krls, M, λ, c);
m_gp   = GP(y, x, x, k_gp, c);

using Plots
plot(X, Y, label = "f(x)", color = "black", legend = :topleft)
scatter!(x, y, label = "Observations", color = "red")
plot!(x, m_krls.predictions, label = "KRLS", color = "green")
plot!(x, m_gp.predictions, label = "GP", color = "red")