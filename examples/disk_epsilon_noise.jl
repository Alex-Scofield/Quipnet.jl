# Regularized reconstruction of a discontinuous disk potential from noisy
# full-grid density data: error vs Tikhonov ε, and the best reconstruction.
#
#   * true V: +V0 on a disk of radius R centered in the box, 0 outside
#   * input:  FourierInput packets with random coefficients
#   * data:   PointDensity() at every grid point, plus 5%-of-RMS measurement
#             noise on every experiment (one seeded draw, reused for all ε);
#             fits use unit noise weights, so ε is in unweighted-misfit units
#   * sweep:  coarse decade scan, then two half-decade refinements around the
#             coarse minimum, plus an unregularized (ε = 0) reference

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Quipnet
using Plots
using Random
using Optim

L = 2π
N = 32
T = 0.4
nsteps = 20
n_exp = 8
in_modes = (3, 3)
R = π / 3
V0 = 1.0
M = 10
noise_fraction = 0.05

# Discontinuous truth: constant inside the disk, zero outside.
disk(x; center = (π, π), radius = R, inside = V0) =
    ifelse.((reshape(x, :, 1) .- center[1]) .^ 2 .+
            (reshape(x, 1, :) .- center[2]) .^ 2 .<= radius^2, inside, 0.0)

x = collect(0:N-1) .* (L / N)
grids = (x, x)
fs = SplitStepSchrodinger(grids; nsteps = nsteps)
input = FourierInput(in_modes, (L, L))
basis = FourierBasis(M, (L, L))
B = PointDensity()
Vtrue = disk(x)

rng = MersenneTwister(7)
ws = [randn(rng, ComplexF64, prod(2 .* in_modes .+ 1)) for _ in 1:n_exp]

# Noise level calibrated to the clean measurement spread.
clean = InverseProblem(fs, input, B, basis,
    GaussianNoise(σw = 0, σμ = 1.0), ws, [T])
clean = generate_data(clean, Vtrue; σμ = 0.0)
σ_noise = noise_fraction * sqrt(sum(abs2, reduce(vcat, clean.measurements)) /
                                length(reduce(vcat, clean.measurements)))

# One noisy realization, reused for every ε so the sweep is paired.
problem = InverseProblem(fs, input, B, basis,
    GaussianNoise(σw = 0, σμ = 1.0), ws, [T])
problem = generate_data(problem, Vtrue; σμ = σ_noise,
    rng = MersenneTwister(11))

function run_fit(problem, Vtrue, ε; iterations, g_tol)
    solver = TikhonovSolver(ε;
        options = Optim.Options(iterations = iterations, g_tol = g_tol))
    res = fit!(solver, problem)
    return (ε = ε, err = relative_error(res.V, Vtrue),
        iterations = res.iterations, converged = res.converged, res = res)
end

# Coarse scan, then quarter-decade refinement around the coarse minimum.
coarse_eps = 10.0 .^ (-6:0.5:0)
coarse = [run_fit(problem, Vtrue, ε; iterations = 800, g_tol = 1e-8)
          for ε in coarse_eps]
ic = argmin([item.err for item in coarse])
fine_eps = 10.0 .^ (log10(coarse[ic].ε) .+ (-0.25, 0.25))
fine = [run_fit(problem, Vtrue, ε; iterations = 400, g_tol = 1e-10)
        for ε in fine_eps]
unreg = run_fit(problem, Vtrue, 0.0; iterations = 600, g_tol = 1e-8)

results = sort(vcat(coarse, fine); by = item -> item.ε)
best = results[argmin([item.err for item in results])]

println("sensors = ", length(x)^2, ", experiments = ", n_exp,
    ", σ_noise = ", round(σ_noise, sigdigits = 3))
println(rpad("ε", 10), rpad("error", 10), rpad("iterations", 12), "converged")
for item in vcat(results, [unreg])
    println(rpad(round(item.ε, sigdigits = 3), 10),
        rpad(round(item.err, sigdigits = 4), 10),
        rpad(item.iterations, 12), item.converged)
end
println("best ε = ", best.ε, ", error = ", best.err)

p_true = heatmap(x, x, Vtrue;
    title = "true V (disk, R = π/3)", aspect_ratio = :equal,
    color = :viridis, xlabel = "x", ylabel = "y", clims = (0, V0))

p_best = heatmap(x, x, best.res.V;
    title = "best reconstruction, ε = $(round(best.ε, sigdigits = 2)), " *
            "err = $(round(best.err, sigdigits = 3))",
    aspect_ratio = :equal, color = :viridis, xlabel = "x", ylabel = "y",
    clims = (0, V0))

p_err = plot([item.ε for item in results], [item.err for item in results];
    xscale = :log10, marker = :circle, linewidth = 2, color = :royalblue,
    label = "coarse", xlabel = "regularization ε",
    ylabel = "relative L² error",
    title = "error vs ε ($(round(Int, 100 * noise_fraction))% noise)")
scatter!(p_err, [item.ε for item in fine], [item.err for item in fine];
    marker = :diamond, markersize = 6, color = :royalblue, label = "refined")
vline!(p_err, [best.ε]; linestyle = :dash, color = :gray,
    label = "ε* = $(round(best.ε, sigdigits = 2))")
hline!(p_err, [unreg.err]; linestyle = :dot, color = :red,
    label = "ε = 0 (err = $(round(unreg.err, sigdigits = 3)))")

plot(p_true, p_best, p_err; layout = (1, 3), size = (1500, 450),
    margin = 5Plots.mm,
    plot_title = "Disk reconstruction with measurement noise (N = $(N), M = $(M))")

out = joinpath(@__DIR__, "disk_epsilon_noise.png")
savefig(out)
println("saved ", out)
