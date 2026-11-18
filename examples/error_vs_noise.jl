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
N = 16
eps = 1e-6
noise_levels = 10.0 .^ (-6:-1)

errors = []

# Grid setup
dx = L / N
x = (collect(0:N-1) .- N/2) .* dx
grids = (x, x)

fs = SplitStepSchrodinger(grids; nsteps=nsteps)
input = FourierInput(modes, (L, L))
basis = GridBasis(grids)

# Define potential expressible in Fourier modes
f_basis = FourierBasis(modes, (L, L))
rng = MersenneTwister(103)
θtrue = 0.05 .* randn(rng, dim(f_basis))
Vtrue = evaluate_basis(f_basis, θtrue, (x, x))

solver = TikhonovSolver(eps; options = Optim.Options(iterations = 2000, g_tol = 1e-5))

for σ_noise in noise_levels
    println("Running Noise = ", σ_noise)
    local noise = GaussianNoise(σw = 0, σμ = σ_noise)
    local ws = [randn(rng, ComplexF64, prod(2 .* modes .+ 1)) for _ in 1:n_exp]
    
    local B = PointDensity(Colon())
    local problem = InverseProblem(fs, input, B, basis, noise, ws, [T])
    problem = generate_data(problem, Vtrue; σμ = σ_noise)
    local result = fit!(solver, problem)
    
    println("  Convergence: ", result.converged)
    push!(errors, relative_error(result.V, Vtrue))
end

# Plotting
p = plot(noise_levels, errors; label="Error", marker=:o,
        title="Error vs Noise (Modes: $(modes), ε = $(eps))",
        xlabel="Noise Level (σμ)", ylabel="Relative L² Error",
        xscale=:log10, yscale=:log10)

out = joinpath(@__DIR__, "error_vs_noise.png")
savefig(p, out)
println("Saved plot to ", out)
