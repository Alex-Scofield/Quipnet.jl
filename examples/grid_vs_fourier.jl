# Compare the Fourier and grid potential bases at equal regularization ε:
#   * a smooth potential whose high-order Fourier coefficients are small
#   * a localized potential barrier with slowly decaying coefficients
# The script reconstructs both truths at ε ∈ {1e-4, 1e-6, 1e-8}, plots the
# noiseless error versus ε, and sweeps measurement noise at fixed ε = 1e-4.

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
x = collect(0:N-1) .* (L / N)

fs = SplitStepSchrodinger((x,); nsteps = nsteps)
input = FourierInput(3, L)
measurement = PointDensity()
noise_model = GaussianNoise(σw = 0, σμ = 1.0)

rng = MersenneTwister(101)
ws = [randn(rng, ComplexF64, 7) for _ in 1:n_exp]

# Truths: a smooth Gaussian bump and a localized rectangular barrier.
Vsmooth = 0.10 .* exp.(-((x .- π) ./ 0.6) .^ 2)
Vbarrier = 0.10 .* (abs.(x .- π) .<= 0.25π + 1e-12)
truths = [("smooth", Vsmooth), ("barrier", Vbarrier)]

# Fourier and grid bases, compared at the same regularization strength.
bases = [
    (name = "Fourier", basis = FourierBasis(8, L)),
    (name = "Grid", basis = GridBasis((x,))),
]
base_colors = Dict("Fourier" => :royalblue, "Grid" => :darkorange)

epsilons = [1e-4, 1e-6, 1e-8]
eps_labels = ["1e-4", "1e-6", "1e-8"]
noise_eps = epsilons[1]
noise_eps_label = eps_labels[1]

noise_levels = [0.0, 1e-4, 1e-3, 1e-2]
noise_labels = ["0", "1e-4", "1e-3", "1e-2"]

# Gauge-align a reconstruction to the truth for plotting (density data cannot
# identify the additive constant).
align_mean(V, Vref) = V .+ (mean(Vref) - mean(V))

# ---------------------------------------------------------------------------
# Reconstructions: noiseless for every ε and a noise sweep at fixed ε
# ---------------------------------------------------------------------------
results = Dict{Tuple{String,String,Float64,Int},InverseResult}()
for (tname, Vtrue) in truths
    for (bname, basis) in bases
        for eps in epsilons
            problem = InverseProblem(fs, input, measurement, basis,
                noise_model, ws, [T])
            problem = generate_data(problem, Vtrue; σμ = 0.0,
                rng = MersenneTwister(400))
            solver = TikhonovSolver(eps;
                options = Optim.Options(iterations = 300, g_tol = 1e-11))
            results[(tname, bname, eps, 1)] = fit!(solver, problem)
        end
        for li in 2:length(noise_levels)
            problem = InverseProblem(fs, input, measurement, basis,
                noise_model, ws, [T])
            problem = generate_data(problem, Vtrue;
                σμ = noise_levels[li],
                rng = MersenneTwister(400 + 10li))
            solver = TikhonovSolver(noise_eps;
                options = Optim.Options(iterations = 300, g_tol = 1e-11))
            results[(tname, bname, noise_eps, li)] = fit!(solver, problem)
        end
    end
end

for (tname, Vtrue) in truths
    println("--- ", tname, " (noiseless) ---")
    for (ei, eps) in enumerate(epsilons), (bname, _) in bases
        res = results[(tname, bname, eps, 1)]
        println(rpad(bname, 10), " ε=", rpad(eps_labels[ei], 6),
            " err=", round(relative_error(res.V, Vtrue), sigdigits = 3),
            " converged=", res.converged,
            " iterations=", res.iterations)
    end
    println("--- ", tname, " (noise sweep, ε=", noise_eps_label, ") ---")
    for li in eachindex(noise_levels), (bname, _) in bases
        res = results[(tname, bname, noise_eps, li)]
        println(rpad(bname, 10), " σμ=", rpad(noise_labels[li], 6),
            " err=", round(relative_error(res.V, Vtrue), sigdigits = 3),
            " converged=", res.converged,
            " iterations=", res.iterations)
    end
end

# ---------------------------------------------------------------------------
# Figure 1: Fourier spectra of the two truths
# ---------------------------------------------------------------------------
dx = x[2] - x[1]
k = fftshift(2π .* fftfreq(N, 1 / dx))
spec_smooth = fftshift(abs.(fft(Vsmooth)) ./ N)
spec_barrier = fftshift(abs.(fft(Vbarrier)) ./ N)

p_spec = plot(k, max.(spec_smooth, 1e-16);
    label = "smooth", color = :royalblue, linewidth = 2,
    xlabel = "wavenumber k", ylabel = "|Fourier coefficient|",
    yscale = :log10, ylims = (1e-8, 1e-1),
    title = "Fourier spectra of the true potentials")
plot!(p_spec, k, max.(spec_barrier, 1e-16);
    label = "barrier", color = :darkorange, linewidth = 2)
vline!(p_spec, [-8, 8];
    label = "Fourier basis cutoff", color = :gray, linestyle = :dash)

out_spec = joinpath(@__DIR__, "..", "grid_vs_fourier_spectra.png")
savefig(p_spec, out_spec)
println("saved ", out_spec)

# ---------------------------------------------------------------------------
# Figure 2: noiseless reconstructions at each ε
# ---------------------------------------------------------------------------
recon_panels = Plots.Plot[]
for (ri, (tname, Vtrue)) in enumerate(truths)
    all_fits = [align_mean(results[(tname, bname, eps, 1)].V, Vtrue)
                for eps in epsilons for (bname, _) in bases]
    lo = minimum(minimum.([Vtrue; all_fits]))
    hi = maximum(maximum.([Vtrue; all_fits]))
    pad = 0.1 * (hi - lo)
    ylims = (lo - pad, hi + pad)

    for (ci, eps) in enumerate(epsilons)
        resF = results[(tname, "Fourier", eps, 1)]
        resG = results[(tname, "Grid", eps, 1)]
        errF = relative_error(resF.V, Vtrue)
        errG = relative_error(resG.V, Vtrue)

        p = plot(x, Vtrue;
            label = "true", color = :black, linestyle = :dash, linewidth = 2,
            ylims = ylims, legend = (ci == 1 ? :topright : false),
            legendfontsize = 8,
            title = "ε = $(eps_labels[ci])\n" *
                "Fourier $(round(errF, sigdigits = 2))  ·  " *
                "grid $(round(errG, sigdigits = 2))",
            xlabel = ri == length(truths) ? "x" : "", ylabel = "V")
        plot!(p, x, align_mean(resF.V, Vtrue);
            label = "Fourier", color = base_colors["Fourier"],
            linewidth = 2)
        plot!(p, x, align_mean(resG.V, Vtrue);
            label = "Grid", color = base_colors["Grid"], linewidth = 2)
        push!(recon_panels, p)
    end
end

p_recon = plot(recon_panels...;
    layout = (length(truths), length(epsilons)),
    size = (1600, 820),
    plot_title = "Noiseless reconstructions at equal regularization ε",
    bottom_margin = 6Plots.mm, left_margin = 6Plots.mm,
    top_margin = 2Plots.mm)

out_recon = joinpath(@__DIR__, "..", "grid_vs_fourier_reconstruction.png")
savefig(p_recon, out_recon)
println("saved ", out_recon)

# ---------------------------------------------------------------------------
# Figure 3: noiseless relative error versus ε
# ---------------------------------------------------------------------------
eps_sorted = sort(epsilons)
eps_panels = Plots.Plot[]
for (tname, Vtrue) in truths
    p = plot(;
        xlabel = "regularization ε", ylabel = "relative L² error",
        xscale = :log10, yscale = :log10, ylims = (1e-5, 1.0),
        title = tname, legend = :topright)
    for (bname, _) in bases
        errs = [relative_error(results[(tname, bname, eps, 1)].V, Vtrue)
                for eps in eps_sorted]
        plot!(p, eps_sorted, errs;
            label = bname, color = base_colors[bname],
            marker = :circle, markersize = 5, linewidth = 2)
    end
    push!(eps_panels, p)
end

p_eps = plot(eps_panels...;
    layout = (1, length(truths)),
    size = (1100, 470),
    plot_title = "Noiseless error versus regularization strength",
    bottom_margin = 5Plots.mm, left_margin = 5Plots.mm)

out_eps = joinpath(@__DIR__, "..", "grid_vs_fourier_eps.png")
savefig(p_eps, out_eps)
println("saved ", out_eps)

# ---------------------------------------------------------------------------
# Figure 4: relative error versus measurement noise at fixed ε
# ---------------------------------------------------------------------------
noise_panels = Plots.Plot[]
for (tname, Vtrue) in truths
    p = plot(;
        xlabel = "measurement noise σμ", ylabel = "relative L² error",
        yscale = :log10, ylims = (1e-4, 10),
        xticks = (1:length(noise_levels), noise_labels),
        title = "$(tname) (ε = $(noise_eps_label))", legend = :topleft)

    for (bname, _) in bases
        errs = [relative_error(results[(tname, bname, noise_eps, li)].V, Vtrue)
                for li in eachindex(noise_levels)]
        plot!(p, 1:length(noise_levels), errs;
            label = bname, color = base_colors[bname],
            marker = :circle, markersize = 5, linewidth = 2)
    end

    push!(noise_panels, p)
end

p_noise = plot(noise_panels...;
    layout = (1, length(truths)),
    size = (1100, 470),
    plot_title = "Noise resilience at fixed ε = $(noise_eps_label)",
    bottom_margin = 5Plots.mm, left_margin = 5Plots.mm)

out_noise = joinpath(@__DIR__, "..", "grid_vs_fourier_noise.png")
savefig(p_noise, out_noise)
println("saved ", out_noise)
