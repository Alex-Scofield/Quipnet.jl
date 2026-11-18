using Quipnet
using Plots
using Optim
using Random

# 3D hydrogen probe: full-grid density observations and no regularization
# (ε = 0). Sweep the number of hydrogen placements: more experiments give more
# measurements, so the reconstruction error should fall and then plateau.

L, N, T, nsteps, a = 6.0, 24, 2.0, 10, 0.5
dx = L / N
x = collect(0:N-1) .* dx
grids = (x, x, x)

fs = hydrogen_solver(grids; nsteps=nsteps)
input = PlacementInput(softening=a)
basis = GridBasis(grids)
noise = GaussianNoise(σw=0, σμ=1.0)

# Smooth, band-limited real potential: a sum of Fourier modes up to |k|∞ = 5,
# each with its own coefficient.
wavevectors = ((1, 0, 2), (0, 1, 1), (2, 2, 0), (3, -1, 2), (5, 3, -4))
coeffs = (0.7, -0.5, 0.35, 0.25, -0.15)
Vtrue = zeros(N, N, N)
for i in 1:N, j in 1:N, k in 1:N
    xi, yj, zk = x[i], x[j], x[k]
    for (m, A) in zip(wavevectors, coeffs)
        Vtrue[i, j, k] += A * cos(2π * (m[1] * xi + m[2] * yj + m[3] * zk) / L)
    end
end

# Full-grid density observation: |ψ|² at every grid point.
B = PointDensity()
nprobe = prod(map(length, grids))

# Candidate hydrogen positions. The sweep uses nested subsets of this pool,
# taken in a fixed shuffled order so the curve is reproducible.
pool = [[rx, ry, rz]
    for rx in range(1.5, 4.5, length=4)
        for ry in range(1.5, 4.5, length=4)
            for rz in range(1.5, 4.5, length=4)]
order = shuffle(Xoshiro(1234), eachindex(pool))
counts = [1, 2, 4, 8, 16, 32, 64]

solver = TikhonovSolver(0.0;
    options=Optim.Options(iterations=50, g_tol=0.0))

println("GridBasis dim = ", dim(basis), ", probes = ", nprobe)
println(rpad("positions", 12), rpad("measurements", 14), rpad("error", 10),
    rpad("iterations", 12), "converged")

results = map(counts) do n
    positions = pool[order[1:n]]
    problem = InverseProblem(fs, input, B, basis, noise, positions, [T])
    problem = generate_data(problem, Vtrue; σμ=0.0)
    result = fit!(solver, problem)
    err = relative_error(result.V, Vtrue)
    println(rpad(n, 12), rpad(nprobe * n, 14),
        rpad(round(err, sigdigits = 4), 10), rpad(result.iterations, 12),
        result.converged)
    (n = n, err = err, result = result)
end

best = results[argmin([item.err for item in results])]
println("best: ", best.n, " positions, error = ", best.err)

p1 = plot([item.n for item in results], [item.err for item in results];
    xscale = :log2, xticks = (counts, string.(counts)),
    marker = :circle, linewidth = 2, legend = false,
    xlabel = "number of hydrogen positions", ylabel = "relative L² error",
    title = "error vs positions (ε = 0)")

kc = N ÷ 2
slice(V) = V[:, :, kc]
clims = extrema(Vtrue)

p2 = heatmap(x, x, slice(Vtrue);
    title = "true V (z-mid)", aspect_ratio = :equal, clims = clims,
    color = :viridis, xlabel = "x", ylabel = "y")
p3 = heatmap(x, x, slice(best.result.V);
    title = "reconstruction, $(best.n) positions (ε = 0)",
    aspect_ratio = :equal, clims = clims, color = :viridis,
    xlabel = "x", ylabel = "y")

plot(p1, p2, p3; layout = (1, 3), size = (1500, 450),
    plot_title = "3D hydrogen probe: full-grid density, no regularization " *
                  "(dim = $(dim(basis)))")

out = joinpath(@__DIR__, "hydrogen_error_vs_positions.png")
savefig(out)
println("saved ", out)
