# Reconstruct a discontinuous disk potential from full-grid density data with a
# truncated Fourier basis.
#
#   * true V: +V0 on a disk of radius R centered in the box, 0 outside
#   * input:  FourierInput packets with random coefficients (unchanged by M)
#   * data:   PointDensity() at every grid point, noiseless
#   * sweep 1: relative error vs the number of Fourier modes M (fixed N)
#   * sweep 2: fit time vs grid size N (fixed M, best of 5 after a warm-up)

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Quipnet
using Plots
using Random
using Optim
using LinearAlgebra

# Stable timings: one BLAS thread and a GC sweep before each timed fit.
blas_threads = BLAS.get_num_threads()
BLAS.set_num_threads(1)

L = 2π
T = 0.4
nsteps = 20
n_exp = 8
in_modes = (3, 3)
R = π / 3
V0 = 1.0
ε = 1e-6

# Discontinuous truth: constant inside the disk, zero outside.
disk(x; center = (π, π), radius = R, inside = V0) =
    ifelse.((reshape(x, :, 1) .- center[1]) .^ 2 .+
            (reshape(x, 1, :) .- center[2]) .^ 2 .<= radius^2, inside, 0.0)

input = FourierInput(in_modes, (L, L))
B = PointDensity()
noise = GaussianNoise(σw = 0, σμ = 1.0)

rng = MersenneTwister(7)
ws = [randn(rng, ComplexF64, prod(2 .* in_modes .+ 1)) for _ in 1:n_exp]

solver = TikhonovSolver(ε;
    options = Optim.Options(iterations = 300, g_tol = 1e-10))

function setup(N, M)
    x = collect(0:N-1) .* (L / N)
    grids = (x, x)
    fs = SplitStepSchrodinger(grids; nsteps = nsteps)
    basis = FourierBasis(M, (L, L))
    Vtrue = disk(x)
    problem = InverseProblem(fs, input, B, basis, noise, ws, [T])
    problem = generate_data(problem, Vtrue; σμ = 0.0)
    return (; x, grids, fs, basis, Vtrue, problem)
end

# Fit timed as the best of `reps` runs after one warm-up fit (JIT + FFT plans).
function timed_fit!(solver, problem; reps = 5)
    fit!(solver, problem)
    best = Inf
    res = nothing
    for _ in 1:reps
        GC.gc()
        t = @elapsed res = fit!(solver, problem)
        best = min(best, t)
    end
    return best, res
end

# ---------------------------------------------------------------------------
# Sweep 1: error vs number of Fourier modes (fixed resolution)
# ---------------------------------------------------------------------------
Nref = 64
Ms = 1:10
resM = map(Ms) do M
    s = setup(Nref, M)
    res = fit!(solver, s.problem)
    Φ = basis_matrix(s.basis, s.grids)
    Vproj = evaluate_basis(s.basis, Φ \ vec(s.Vtrue), s.grids)
    (M = M, dim = dim(s.basis),
        err = relative_error(res.V, s.Vtrue),
        floor = relative_error(Vproj, s.Vtrue),
        iterations = res.iterations, converged = res.converged,
        res = res, s = s)
end

println("--- error vs Fourier modes (N = ", Nref, ", ε = ", ε, ") ---")
for item in resM
    println(rpad("M = $(item.M)", 8), rpad("dim = $(item.dim)", 12),
        rpad("fit err = $(round(item.err, sigdigits = 4))", 24),
        rpad("floor = $(round(item.floor, sigdigits = 4))", 24),
        "iterations = ", item.iterations)
end

# ---------------------------------------------------------------------------
# Sweep 2: convergence time vs grid size (fixed basis)
# ---------------------------------------------------------------------------
Mfix = 6
Ns = [16, 24, 32, 48, 64]
resN = map(Ns) do N
    s = setup(N, Mfix)
    t, res = timed_fit!(solver, s.problem)
    (N = N, dim = dim(s.basis), time = t,
        err = relative_error(res.V, s.Vtrue),
        iterations = res.iterations, converged = res.converged, s = s)
end

println("--- time vs N (M = ", Mfix, ", ε = ", ε, "; best of 5) ---")
for item in resN
    println(rpad("N = $(item.N)", 8), rpad("dim = $(item.dim)", 12),
        rpad("time = $(round(item.time, digits = 3)) s", 20),
        rpad("err = $(round(item.err, sigdigits = 4))", 22),
        "iterations = ", item.iterations)
end

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
best = resM[argmin([item.err for item in resM])]
xb = best.s.x

p_true = heatmap(xb, xb, best.s.Vtrue;
    title = "true V (disk, R = π/3)", aspect_ratio = :equal,
    color = :viridis, xlabel = "x", ylabel = "y", clims = (0, V0))

p_recon = heatmap(xb, xb, best.res.V;
    title = "reconstruction, M = $(best.M), N = $(Nref)",
    aspect_ratio = :equal, color = :viridis, xlabel = "x", ylabel = "y",
    clims = (0, V0))

p_modes = plot(Ms, [item.err for item in resM];
    marker = :circle, linewidth = 2, label = "fit", color = :royalblue,
    xlabel = "Fourier modes M", ylabel = "relative L² error",
    title = "error vs modes (N = $(Nref))")
plot!(p_modes, Ms, [item.floor for item in resM];
    linestyle = :dash, linewidth = 2, label = "truncation floor",
    color = :gray)

xt = (Ns, string.(Ns))

p_time = plot([item.N for item in resN], [item.time for item in resN];
    xscale = :log2, yscale = :log10, xticks = xt,
    marker = :circle, linewidth = 2, label = "fit",
    xlabel = "grid size N", ylabel = "fit time (s)",
    title = "time vs N (M = $(Mfix))")
tlast = resN[end].time
plot!(p_time, Ns, [tlast * (N / resN[end].N)^2 for N in Ns];
    linestyle = :dash, color = :gray, label = "∝ N²")

p_nerr = plot([item.N for item in resN], [item.err for item in resN];
    xscale = :log2, xticks = xt, marker = :circle, linewidth = 2,
    legend = false, xlabel = "grid size N", ylabel = "relative L² error",
    title = "error vs N (M = $(Mfix))")

p_iter = plot([item.N for item in resN], [item.iterations for item in resN];
    xscale = :log2, xticks = xt, marker = :circle, linewidth = 2,
    legend = false, xlabel = "grid size N", ylabel = "LBFGS iterations",
    title = "iterations vs N (M = $(Mfix))")

plot(p_true, p_recon, p_modes, p_time, p_nerr, p_iter;
    layout = (2, 3), size = (1500, 950), margin = 6Plots.mm,
    plot_title = "Disk potential: Fourier truncation (ε = $(ε))")

out = joinpath(@__DIR__, "disk_fourier_modes.png")
savefig(out)
println("saved ", out)

BLAS.set_num_threads(blas_threads)
