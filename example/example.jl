
T = 200;
σ = 0.1
x = hcat(ones(T), 1:T);
b = [1, -1];
z = x[:, 1] - sin.(x[:, 2] * 0.25);
y = z + rand(Normal(0.0, σ), T);
k = RBF(1.0, 1.0);

m_rls  = RLS(y, x; λ = 0.99);
m_krls = KRLS(y, x, k; λ = 0.98, s2n = sqrt(0.25));
m_gp   = GP(y, x, k; s2n = sqrt(0.25));

using Plots
scatter(y, label = "Measurement", color = "grey")
plot!(z, label = "Signal", color = "black")
#plot!(m_rls.predictions, label = "RLS", color = "red")
plot!(m_krls.predictions, label = "KRLS", color = "green")
plot!(m_gp.μ, label = "GP", color = "blue")