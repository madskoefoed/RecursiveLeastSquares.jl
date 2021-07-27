
T = 1000;
x = hcat(ones(T), 1:T);
b = hcat(repeat([1, 1, 1, 1], inner = 250), repeat([-1], 1000));
y = vec(sum(sin.((x) * 0.01) .* b; dims = 2)) .+ randn(T) * 0.25;
k = RBF(1.0, 5.0);

m_rls  = RLS(y, x; λ = 0.98)
m_krls = KRLS(y, x, k; λ = 0.98, s2n = 0.1)

using Plots
plot(y)
plot!(model.predictions)