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
N = 16 # Slightly smaller N to improve speed
modes = (6, 6) # Increased modes
noise_levels = 10.0 .^ (-8:1)
epsilons = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e0]

# Grid setup
dx = L / N
x = (collect(0:N-1) .- N/2) .* dx
grids = (x, x)

fs = SplitStepSchrodinger(grids; nsteps=nsteps)
input = FourierInput(modes, (L, L))
basis = GridBasis(grids)

# Define potential barrier (Smoother Gaussian)
x_grid = [xi for xi in x, yj in x]
y_grid = [yj for xi in x, yj in x]
Vtrue = 1.0 .* exp.(-(x_grid.^2 .+ y_grid.^2) ./ 8.0) # Smoother: denominator 8.0

# Plotting setup
p = plot(title="Error vs Noise (Smoother Barrier, N=$(N), Modes=$(modes))",
         xlabel="Noise Level (σμ)", ylabel="Relative L² Error",
         xscale=:log10, yscale=:log10)

rng = MersenneTwister(103)

for eps in epsilons
    println("Running ε = ", eps)
    local errors = []
    local solver = TikhonovSolver(eps; options = Optim.Options(iterations = 200, g_tol = 1e-5))
    
    for σ_noise in noise_levels
        local noise = GaussianNoise(σw = 0, σμ = σ_noise)
        local ws = [randn(rng, ComplexF64, prod(2 .* modes .+ 1)) for _ in 1:n_exp]
        
        # Use PointDensity
        local B = PointDensity(Colon())
        local problem = InverseProblem(fs, input, B, basis, noise, ws, [T])
        problem = generate_data(problem, Vtrue; σμ = σ_noise)
        local result = fit!(solver, problem)
        
        push!(errors, relative_error(result.V, Vtrue))
    end
    plot!(p, noise_levels, errors, label="ε = $eps", marker=:o)
end

out = joinpath(@__DIR__, "error_vs_noise_barrier.png")
savefig(p, out)
println("Saved plot to ", out)
