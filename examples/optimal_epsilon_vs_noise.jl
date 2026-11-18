# Optimal Tikhonov regularization ε* vs measurement noise level
# for reconstructing a discontinuous disk potential.

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

# Clean reference to compute baseline RMS
clean = InverseProblem(fs, input, B, basis,
    GaussianNoise(σw = 0, σμ = 1.0), ws, [T])
clean = generate_data(clean, Vtrue; σμ = 0.0)
clean_rms = sqrt(sum(abs2, reduce(vcat, clean.measurements)) /
                 length(reduce(vcat, clean.measurements)))

noise_fractions = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

function run_fit(problem, Vtrue, ε; iterations, g_tol)
    solver = TikhonovSolver(ε;
        options = Optim.Options(iterations = iterations, g_tol = g_tol))
    res = fit!(solver, problem)
    return (ε = ε, err = relative_error(res.V, Vtrue),
        iterations = res.iterations, converged = res.converged)
end

optimal_epsilons = Float64[]
actual_noise_levels = Float64[]

println("Noise Fraction | σ_noise | Best ε | Min Error")
println("-"^50)

for nf in noise_fractions
    σ_noise = nf * clean_rms
    push!(actual_noise_levels, σ_noise)

    # Noisy realization
    problem = InverseProblem(fs, input, B, basis,
        GaussianNoise(σw = 0, σμ = 1.0), ws, [T])
    problem = generate_data(problem, Vtrue; σμ = σ_noise,
        rng = MersenneTwister(11))

    # Coarse scan, then refinement
    coarse_eps = 10.0 .^ (-6:0.5:0)
    coarse = [run_fit(problem, Vtrue, ε; iterations = 600, g_tol = 1e-8)
              for ε in coarse_eps]
    ic = argmin([item.err for item in coarse])
    
    # Refinement around coarse minimum
    min_coarse_eps = coarse[ic].ε
    fine_eps = 10.0 .^ (log10(min_coarse_eps) .+ (-0.25, 0.25))
    fine = [run_fit(problem, Vtrue, ε; iterations = 400, g_tol = 1e-10)
            for ε in fine_eps]

    results = vcat(coarse, fine)
    best = results[argmin([item.err for item in results])]

    push!(optimal_epsilons, best.ε)
    println(rpad("$(Int(100*nf))%", 15), rpad(round(σ_noise, sigdigits = 3), 10), rpad(round(best.ε, sigdigits = 3), 10), round(best.err, sigdigits = 4))
end

# Plotting optimal epsilon vs noise level
p = plot(actual_noise_levels, optimal_epsilons;
    xscale = :log10, yscale = :log10,
    marker = :circle, linewidth = 2, color = :darkblue,
    xlabel = "Noise Level (σ_noise)",
    ylabel = "Optimal Regularization ε*",
    title = "Optimal Tikhonov ε* vs Measurement Noise Level")

out = joinpath(@__DIR__, "optimal_epsilon_vs_noise.png")
savefig(p, out)
println("\nSaved plot to ", out)
