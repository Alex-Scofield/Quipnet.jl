using Quipnet
using Plots
using Random
using Optim
using LinearAlgebra

# Setup for Complex Potential - full grid measurements
L = 10.0
T = 0.1
nsteps = 5
n_exp = 3
Ns = 6:2:24 # Range of grid sizes (N x N)

errors_point = []
errors_mom = []

# Using an updated solver configuration
ε = 2e-6 # Slightly increased regularization for better convergence
solver = TikhonovSolver(ε;
    options = Optim.Options(iterations = 5000, g_tol = 1e-5))

for N in Ns
    println("Running N = ", N)
    
    # Use explicit local scope
    local dx = L / N
    local x = (collect(0:N-1) .- N/2) .* dx
    local grids = (x, x)
    
    local fs = SplitStepSchrodinger(grids; nsteps=nsteps)
    local input = FourierInput((4, 4), (L, L))
    local basis = GridBasis(grids)
    local noise = GaussianNoise(σw = 0, σμ = 1e-4)

    # Complex Potential: Sum of two Gaussians (sampled continuously)
    local Vtrue = [0.5 * (exp(-((xi - L/4)^2 + (yj - L/4)^2) / 1.0) +
                          exp(-((xi + L/4)^2 + (yj + L/4)^2) / 1.0)) for xi in x, yj in x]

    local rng = MersenneTwister(103)
    local ws = [randn(rng, ComplexF64, prod(2 .* (4, 4) .+ 1)) for _ in 1:n_exp]

    function fit_measurement(B_type)
        # Using Colon() for full grid measurements
        local B = B_type(Colon())
        local problem = InverseProblem(fs, input, B, basis, noise, ws, [T])
        problem = generate_data(problem, Vtrue; σμ = 1e-4)
        local result = fit!(solver, problem)
        
        println("  Convergence (", B_type, "): ", result.converged)
        return relative_error(result.V, Vtrue)
    end

    push!(errors_point, fit_measurement(PointDensity))
    push!(errors_mom, fit_measurement(MomentumDensity))
end

# Plotting
title_str = "Error vs N (Full Grid Measurements, ε = $ε)"
p = plot(Ns, errors_point; label="PointDensity (Full Grid)", marker=:o, title=title_str, xlabel="N", ylabel="Relative L² Error", yscale=:log10)
plot!(p, Ns, errors_mom; label="MomentumDensity (Full Grid)", marker=:o)

out = joinpath(@__DIR__, "error_vs_n_fullgrid_2d.png")
savefig(p, out)
println("Saved plot to ", out)
