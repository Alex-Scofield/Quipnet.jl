using Quipnet
using Plots
using Random
using Optim
using LinearAlgebra

# Setup
L = 10.0
T = 0.1
nsteps = 5
n_exp = 3
modes = (10, 10) # 10 Fourier modes
σ_noise = 1e-4   # Fixed noise level

Ns = [8, 12, 16]
epsilons = 10.0 .^ (-10:-2)

all_errors = []

# Define potential consistently
function get_Vtrue(x)
    return [0.5 * (exp(-((xi - L/4)^2 + (yj - L/4)^2) / 1.0) +
                   exp(-((xi + L/4)^2 + (yj + L/4)^2) / 1.0)) for xi in x, yj in x]
end

for N in Ns
    println("Running N = ", N)
    
    local dx = L / N
    local x = (collect(0:N-1) .- N/2) .* dx
    local grids = (x, x)
    
    local fs = SplitStepSchrodinger(grids; nsteps=nsteps)
    local input = FourierInput(modes, (L, L))
    local basis = GridBasis(grids)
    local noise = GaussianNoise(σw = 0, σμ = σ_noise)
    
    local Vtrue = get_Vtrue(x)
    
    local rng = MersenneTwister(103)
    local ws = [randn(rng, ComplexF64, prod(2 .* modes .+ 1)) for _ in 1:n_exp]
    
    local errors_eps = []
    for eps in epsilons
        local solver = TikhonovSolver(eps;
            options = Optim.Options(iterations = 2000, g_tol = 1e-5))
        
        local B = PointDensity(Colon())
        local problem = InverseProblem(fs, input, B, basis, noise, ws, [T])
        problem = generate_data(problem, Vtrue; σμ = σ_noise)
        local result = fit!(solver, problem)
        
        push!(errors_eps, relative_error(result.V, Vtrue))
    end
    push!(all_errors, errors_eps)
end

# Plotting
title_str = "Error vs ε (PointDensity)\nModes: $(modes), Noise (σμ): $(σ_noise)"
p = plot(title=title_str, xlabel="ε (Tikhonov Regularization)", ylabel="Relative L² Error", xscale=:log10, yscale=:log10)

for (i, N) in enumerate(Ns)
    plot!(p, epsilons, all_errors[i], label="N = $N", marker=:o)
end

out = joinpath(@__DIR__, "error_vs_epsilon.png")
savefig(p, out)
println("Saved plot to ", out)
