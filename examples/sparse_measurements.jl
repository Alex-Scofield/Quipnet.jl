using Quipnet
using Plots
using Random
using Optim

N = 24
L = 2π
T = 0.3
nsteps = 15
n_exp = 8
modes = (2, 2)

x = collect(0:N-1) .* (L / N)
fs = SplitStepSchrodinger((x, x); nsteps=nsteps)
input = FourierInput(modes, (L, L))
basis = FourierBasis(modes, (L, L))
noise = GaussianNoise(σw = 0, σμ = 1.0)

rng = MersenneTwister(103)
θtrue = 0.01 .* randn(rng, dim(basis))
ws = [randn(rng, ComplexF64, 25) for _ in 1:n_exp]
Vtrue = evaluate_basis(basis, θtrue, (x, x))

sensor_sets = [
    ("1 sensor", [CartesianIndex(12, 12)]),
    ("4 sensors (line)", [
        CartesianIndex(6, 6), CartesianIndex(6, 10),
        CartesianIndex(6, 14), CartesianIndex(6, 18),
    ]),
    ("9 sensors", [
        CartesianIndex(i, j) for i in (4, 12, 20) for j in (4, 12, 20)
    ]),
]

solver = TikhonovSolver(1e-8;
    options = Optim.Options(iterations = 300, g_tol = 1e-10))

function fit_measurement(B)
    problem = InverseProblem(fs, input, B, basis, noise, ws, [T])
    problem = generate_data(problem, Vtrue; σμ = 0.0)
    return fit!(solver, problem)
end

results = map(sensor_sets) do (label, inds)
    result = fit_measurement(PointDensity(inds))
    (label = label, indices = inds, result = result,
        error = relative_error(result.V, Vtrue))
end

full = fit_measurement(PointDensity())
full_error = relative_error(full.V, Vtrue)

for item in results
    println(rpad(item.label, 10), " error = ", item.error,
        ", converged = ", item.result.converged,
        ", iterations = ", item.result.iterations)
end
println(rpad("full grid", 10), " error = ", full_error,
    ", converged = ", full.converged,
    ", iterations = ", full.iterations)

clims = extrema(Vtrue)
p1 = heatmap(x, x, Vtrue;
    title = "true potential", xlabel = "x", ylabel = "y",
    aspect_ratio = :equal, color = :viridis, colorbar = false, clims = clims)

p2 = heatmap(x, x, full.V;
    title = "full density, error = $(round(full_error, sigdigits = 2))",
    xlabel = "x", ylabel = "y", aspect_ratio = :equal,
    color = :viridis, colorbar = false, clims = clims)

sparse = results[2]
p3 = heatmap(x, x, sparse.result.V;
    title = "4 sensors (line), error = $(round(sparse.error, sigdigits = 2))",
    xlabel = "x", ylabel = "y", aspect_ratio = :equal,
    color = :viridis, colorbar = false, clims = clims)
scatter!(p3,
    [x[ci[1]] for ci in sparse.indices],
    [x[ci[2]] for ci in sparse.indices];
    marker = :circle, markersize = 7, markercolor = :white,
    markerstrokecolor = :black, label = "")

labels = [item.label for item in results]
errors = [item.error for item in results]
push!(labels, "full grid")
push!(errors, full_error)
p4 = bar(labels, errors;
    title = "reconstruction error", ylabel = "relative L² error",
    yscale = :log10, ylims = (1e-7, 2), color = :gray,
    legend = false, xrotation = 20)

plot(p1, p2, p3, p4;
    layout = (2, 2), size = (1000, 850),
    plot_title = "Sparse density measurements")

out = joinpath(@__DIR__, "..", "inverse_sparse_demo.png")
savefig(out)
println("saved ", out)
