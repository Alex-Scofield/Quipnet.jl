using Quipnet
using Plots
using Random
using Optim

N = 64
L = 2π
T = 0.3
nsteps = 15
n_exp = 8
modes = (2, 2)
measurement_noise = 1e-5

x = collect(0:N-1) .* (L / N)
fs = SplitStepSchrodinger((x, x); nsteps=nsteps)
input = FourierInput(modes, (L, L))
basis = FourierBasis(modes, (L, L))
noise = GaussianNoise(σw = 0, σμ = 1.0)

rng = MersenneTwister(103)
θlow = 0.01 .* randn(rng, dim(basis))
ws = [randn(rng, ComplexF64, 25) for _ in 1:n_exp]
Vlow = evaluate_basis(basis, θlow, (x, x))
Vhigh = 0.004 .* cos.(5 .* x) .* cos.(5 .* x') .+
        0.002 .* sin.(4 .* x) .* cos.(3 .* x')
Vtrue = Vlow + Vhigh

sensors = [
    CartesianIndex(16, 16),
    CartesianIndex(48, 16),
    CartesianIndex(32, 48),
]

solver = TikhonovSolver(1e-8;
    options = Optim.Options(iterations = 300, g_tol = 1e-10))

function fit_measurement(B, seed)
    problem = InverseProblem(fs, input, B, basis, noise, ws, [T])
    problem = generate_data(problem, Vtrue;
        σμ = measurement_noise, rng = MersenneTwister(seed))
    return fit!(solver, problem)
end

full = fit_measurement(PointDensity(), 201)
sparse = fit_measurement(PointDensity(sensors), 202)
full_error = relative_error(full.V, Vtrue)
sparse_error = relative_error(sparse.V, Vtrue)

println("measurement noise = ", measurement_noise)
println("full grid  error = ", full_error,
    ", converged = ", full.converged, ", iterations = ", full.iterations)
println("3 sensors  error = ", sparse_error,
    ", converged = ", sparse.converged, ", iterations = ", sparse.iterations)

clims = extrema(Vtrue)
p1 = heatmap(x, x, Vtrue;
    title = "true potential (modes ≤ 5)", xlabel = "x", ylabel = "y",
    aspect_ratio = :equal, color = :viridis, colorbar = false, clims = clims)

p2 = heatmap(x, x, full.V;
    title = "full density (basis ≤ 2), error = $(round(full_error, sigdigits = 2))",
    xlabel = "x", ylabel = "y", aspect_ratio = :equal,
    color = :viridis, colorbar = false, clims = clims)

p3 = heatmap(x, x, sparse.V;
    title = "3 sensors (triangle, noisy), error = $(round(sparse_error, sigdigits = 2))",
    xlabel = "x", ylabel = "y", aspect_ratio = :equal,
    color = :viridis, colorbar = false, clims = clims)
scatter!(p3,
    [x[ci[1]] for ci in sensors],
    [x[ci[2]] for ci in sensors];
    marker = :circle, markersize = 8, markercolor = :white,
    markerstrokecolor = :black, label = "")

full_residual = full.V - Vtrue
sparse_residual = sparse.V - Vtrue
full_scale = maximum(abs, full_residual)
sparse_scale = maximum(abs, sparse_residual)

p4 = heatmap(x, x, full_residual;
    title = "full - truth (model error)", xlabel = "x", ylabel = "y",
    aspect_ratio = :equal, color = :RdBu, colorbar = false,
    clims = (-full_scale, full_scale))

p5 = heatmap(x, x, sparse_residual;
    title = "3 sensors - truth", xlabel = "x", ylabel = "y",
    aspect_ratio = :equal, color = :RdBu, colorbar = false,
    clims = (-sparse_scale, sparse_scale))

p6 = bar(["3 sensors", "full grid"], [sparse_error, full_error];
    title = "reconstruction error", ylabel = "relative L² error",
    yscale = :log10, ylims = (1e-8, 1), color = :gray,
    legend = false, xrotation = 20)

plot(p1, p2, p3, p4, p5, p6;
    layout = (2, 3), size = (1500, 850),
    plot_title = "3 sensors, 64×64 mesh, σμ = $(measurement_noise); truth has modes 3–5")

out = joinpath(@__DIR__, "..", "inverse_sparse_triangle_demo.png")
savefig(out)
println("saved ", out)
