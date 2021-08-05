σ = 0.1;
# Training
t = 200;
x = collect(1:t)/t;
b = -0.25;
z = sin.(b * x * t);
y = z + rand(Normal(0.0, σ), t);
# Kernel
k_krls = RBF(2.0^-5, 2.0^2);
k_gp   = RBF(2.0^-8, 2.0^8);

m_rls  = RLS(y, x, 0.98);
m_krls = KRLS(y, x, k_krls, t, 0.98, sqrt(0.1));
m_gp   = GP(y, x, x, k_gp, sqrt(0.1));

using Plots
scatter(x, y, label = "Measurement", color = "grey")
plot!(x, z, label = "Signal", color = "black")
#plot!(x, m_rls.predictions, label = "RLS", color = "red")
plot!(x, m_krls.predictions, label = "KRLS", color = "green")
plot!(x, m_gp.predictions, label = "GP", color = "red")

findmin([sum((y .- KRLS(y, x, RBF(2.0^i, 2.0^j), 0.98, sqrt(0.25)).predictions).^2) for i = -10:8, j = -10:8])
findmin([sum((y .- GP(y, x, x, RBF(2.0^i, 2.0^j), sqrt(0.25)).predictions).^2) for i = -10:8, j = -10:8])

plot([sqrt(mean((y .- m_krls.predictions)[i-100:i].^2)) for i in 101:t])
plot!([sqrt(mean((y .- m_gp.predictions)[i-100:i].^2)) for i in 101:t])
hline!([σ])