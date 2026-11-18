using Quipnet
using Plots
using Random
using Optim
using LinearAlgebra

# Setup
N = 16
L = 2π
T = 0.3
nsteps = 15
n_exp = 8
modes = (2, 2)

x = collect(0:N-1) .* (L / N)
fs = SplitStepSchrodinger((x, x); nsteps=nsteps)
input = FourierInput(modes, (L, L))
basis = FourierBasis(modes, (L, L))
noise = GaussianNoise(σw = 0, σμ = 1e-12)

rng = MersenneTwister(103)
# Random potential representable by the basis
θtrue = 0.05 .* randn(rng, dim(basis))
Vtrue = evaluate_basis(basis, θtrue, (x, x))

ws = [randn(rng, ComplexF64, prod(2 .* modes .+ 1)) for _ in 1:n_exp]

solver = TikhonovSolver(1e-8;
    options = Optim.Options(iterations = 50, g_tol = 1e-5))

function fit_measurement(B, Vtrue, ws)
    problem = InverseProblem(fs, input, B, basis, noise, ws, [T])
    problem = generate_data(problem, Vtrue; σμ = 1e-12)
    return fit!(solver, problem)
end

println("Fitting PointDensity...")
res_point = fit_measurement(PointDensity(), Vtrue, ws)
err_point = relative_error(res_point.V, Vtrue)
println("PointDensity error: ", err_point)

println("Fitting MomentumDensity...")
res_mom = fit_measurement(MomentumDensity(), Vtrue, ws)
err_mom = relative_error(res_mom.V, Vtrue)
println("MomentumDensity error: ", err_mom)

# Plotting
clims = extrema(Vtrue)
p1 = heatmap(x, x, Vtrue;
    title = "true potential", aspect_ratio = :equal, color = :viridis)

p2 = heatmap(x, x, res_point.V;
    title = "PointDensity, err=$(round(err_point, sigdigits=2))", 
    aspect_ratio = :equal, color = :viridis, clims=clims)

p3 = heatmap(x, x, res_mom.V;
    title = "MomentumDensity, err=$(round(err_mom, sigdigits=2))", 
    aspect_ratio = :equal, color = :viridis, clims=clims)

p_bar = bar(["PointDensity", "MomentumDensity"], [err_point, err_mom];
    title = "Error Comparison", ylabel = "Relative L² Error",
    legend = false, color = [:blue, :red])

plot(p1, p2, p3, p_bar;
    layout = (2, 2), size = (800, 800),
    plot_title = "PointDensity vs MomentumDensity")

out = joinpath(@__DIR__, "measurement_comparison_2d.png")
savefig(out)
println("Saved plot to ", out)
