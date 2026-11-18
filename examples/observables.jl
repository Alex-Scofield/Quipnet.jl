# Compare position-density and momentum-density measurements in the same
# inverse problem. All parameters are held fixed (potential basis, ε, inputs,
# observation times, optimizer, and noise seed); only the measurement map
# changes. The two truths stress complementary information:
#   * a localized barrier, where position data is more informative;
#   * a delocalized periodic grating, where momentum data is more informative.
# The composite measurement stacks both observables.

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Quipnet
using FFTW
using Plots
using Random
using Optim
using Statistics

# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------
L = 2π
N = 24
T = 0.5
nsteps = 25
n_exp = 32
epsilon = 1e-6
x = collect(0:N-1) .* (L / N)
dx = x[2] - x[1]

fs = SplitStepSchrodinger((x,); nsteps = nsteps)
input = FourierInput(3, L)
basis = GridBasis((x,))
noise_model = GaussianNoise(σw = 0, σμ = 1.0)

rng = MersenneTwister(101)
ws = [randn(rng, ComplexF64, 7) for _ in 1:n_exp]

truths = [
    ("barrier", 0.10 .* (abs.(x .- π) .<= 0.25π + 1e-12)),
    ("cosine", 0.10 .* cos.(x)),
]

observables = [
    (name = "position", B = PointDensity(), color = :royalblue),
    (name = "momentum", B = MomentumDensity(), color = :darkorange),
    (name = "position+momentum",
        B = CompositeMeasurement(PointDensity(), MomentumDensity()),
        color = :seagreen),
]
obs_colors = Dict(o.name => o.color for o in observables)

noise_levels = [0.0, 1e-3, 1e-2]
noise_labels = ["0", "1e-3", "1e-2"]

# Density data cannot identify the additive constant of the potential.
align_mean(V, Vref) = V .+ (mean(Vref) - mean(V))

# ---------------------------------------------------------------------------
# Reconstructions: noiseless and a small measurement-noise sweep
# ---------------------------------------------------------------------------
results = Dict{Tuple{String,String,Int},InverseResult}()
for (tname, Vtrue) in truths
    for (li, level) in enumerate(noise_levels)
        for obs in observables
            problem = InverseProblem(fs, input, obs.B, basis,
                noise_model, ws, [T])
            problem = generate_data(problem, Vtrue;
                σμ = level, rng = MersenneTwister(400 + 10li))
            solver = TikhonovSolver(epsilon;
                options = Optim.Options(iterations = 300, g_tol = 1e-11))
            results[(tname, obs.name, li)] = fit!(solver, problem)
        end
    end
end

for (tname, Vtrue) in truths
    println("--- ", tname, " (ε=", epsilon, ") ---")
    for (li, level) in enumerate(noise_levels), obs in observables
        res = results[(tname, obs.name, li)]
        println(rpad(obs.name, 20), " σμ=", rpad(noise_labels[li], 6),
            " err=", round(relative_error(res.V, Vtrue), sigdigits = 3),
            " converged=", res.converged,
            " iterations=", res.iterations)
    end
end

# ---------------------------------------------------------------------------
# Figure 1: the two measurement types for one experiment
# ---------------------------------------------------------------------------
k = fftshift(2π .* fftfreq(N, 1 / dx))
snapshot_panels = Plots.Plot[]
for (tname, Vtrue) in truths
    ψT = propagate(fs, apply_input(input, ws[1], (x,)), Vtrue, T)
    pos = abs2.(ψT)
    mom = fftshift(abs2.(fft(ψT)) ./ N)
    ymax = 1.1 * max(maximum(pos), maximum(mom))
    p1 = plot(x, pos;
        title = "$tname: position density", xlabel = "x",
        ylabel = "|ψ(T)|²", ylims = (0, ymax),
        color = obs_colors["position"], legend = false, linewidth = 2)
    p2 = plot(k, mom;
        title = "$tname: momentum density", xlabel = "k",
        ylabel = "|ψ̂(T)|² / N", ylims = (0, ymax),
        color = obs_colors["momentum"], legend = false, linewidth = 2)
    push!(snapshot_panels, p1, p2)
end

p_snapshot = plot(snapshot_panels...;
    layout = (length(truths), 2),
    size = (1100, 700),
    plot_title = "Measurement data for one experiment (noiseless)",
    bottom_margin = 5Plots.mm, left_margin = 5Plots.mm)

out_snapshot = joinpath(@__DIR__, "..", "observables_measurements.png")
savefig(p_snapshot, out_snapshot)
println("saved ", out_snapshot)

# ---------------------------------------------------------------------------
# Figure 2: noiseless reconstructions for every observable
# ---------------------------------------------------------------------------
recon_panels = Plots.Plot[]
for (ri, (tname, Vtrue)) in enumerate(truths)
    for (ci, obs) in enumerate(observables)
        res = results[(tname, obs.name, 1)]
        fit = align_mean(res.V, Vtrue)
        err = relative_error(res.V, Vtrue)
        lo = min(minimum(Vtrue), minimum(fit))
        hi = max(maximum(Vtrue), maximum(fit))
        pad = 0.1 * (hi - lo)

        p = plot(x, Vtrue;
            label = "true", color = :black, linestyle = :dash, linewidth = 2,
            ylims = (lo - pad, hi + pad),
            legend = (ci == 1 ? :topright : false), legendfontsize = 8,
            title = "$tname — $(obs.name)\n" *
                "err = $(round(err, sigdigits = 3))",
            xlabel = ri == length(truths) ? "x" : "", ylabel = "V")
        plot!(p, x, fit;
            label = obs.name, color = obs.color, linewidth = 2)
        push!(recon_panels, p)
    end
end

p_recon = plot(recon_panels...;
    layout = (length(truths), length(observables)),
    size = (1600, 780),
    plot_title = "Noiseless reconstructions at equal ε",
    bottom_margin = 6Plots.mm, left_margin = 6Plots.mm,
    top_margin = 2Plots.mm)

out_recon = joinpath(@__DIR__, "..", "observables_reconstruction.png")
savefig(p_recon, out_recon)
println("saved ", out_recon)

# ---------------------------------------------------------------------------
# Figure 3: relative error versus measurement noise
# ---------------------------------------------------------------------------
noise_panels = Plots.Plot[]
for (tname, Vtrue) in truths
    p = plot(;
        xlabel = "measurement noise σμ", ylabel = "relative L² error",
        yscale = :log10, ylims = (1e-5, 10),
        xticks = (1:length(noise_levels), noise_labels),
        title = "$(tname) (ε = $(epsilon))", legend = :topleft)

    for obs in observables
        errs = [relative_error(results[(tname, obs.name, li)].V, Vtrue)
                for li in eachindex(noise_levels)]
        plot!(p, 1:length(noise_levels), errs;
            label = obs.name, color = obs.color,
            marker = :circle, markersize = 5, linewidth = 2)
    end

    push!(noise_panels, p)
end

p_noise = plot(noise_panels...;
    layout = (1, length(truths)),
    size = (1100, 470),
    plot_title = "Observable comparison under measurement noise",
    bottom_margin = 5Plots.mm, left_margin = 5Plots.mm)

out_noise = joinpath(@__DIR__, "..", "observables_noise.png")
savefig(p_noise, out_noise)
println("saved ", out_noise)
