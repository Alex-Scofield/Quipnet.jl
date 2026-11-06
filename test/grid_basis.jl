using Quipnet
using Test
using Random
using Optim

@testset "Grid basis: point values" begin
    L = 2π
    N = 8
    x = collect(0:N-1) .* (L / N)
    b = GridBasis((x,))
    @test dim(b) == N
    @test GridBasis(x) isa GridBasis

    θ = randn(MersenneTwister(1), N)
    V = evaluate_basis(b, θ, (x,))
    @test V ≈ θ
    @test evaluate_basis(b, θ) ≈ θ
    @test basis_matrix(b, (x,)) * θ ≈ θ

    N2 = 6
    x2 = collect(0:N2-1) .* (L / N2)
    b2 = GridBasis((x2, x2))
    @test dim(b2) == N2^2
    θ2 = randn(MersenneTwister(2), N2^2)
    V2 = evaluate_basis(b2, θ2, (x2, x2))
    @test size(V2) == (N2, N2)
    @test vec(V2) == θ2
    @test basis_matrix(b2, (x2, x2)) * θ2 ≈ θ2

    @test_throws ArgumentError basis_matrix(b, (x[1:end-1],))
    @test_throws ArgumentError basis_matrix(b, (x, x))
    @test_throws ArgumentError evaluate_basis(b2, θ2, (x2[1:end-1], x2))
    @test_throws ArgumentError evaluate_basis(b, θ[1:end-1], (x,))
    @test_throws ArgumentError evaluate_basis(b, θ, (0.3,))
    @test_throws ArgumentError evaluate_basis(b, θ, 0.3)
    @test_throws ArgumentError GridBasis((0.3,))
end

@testset "Grid basis: analytic adjoint" begin
    L, N = 2π, 12
    x = collect(0:N-1) .* (L / N)
    fs = SplitStepSchrodinger((x,); nsteps=5)
    A = FourierInput(1, L)
    B = PointDensity(1:2:N)
    basis = GridBasis((x,))
    noise = GaussianNoise(σw = 0, σμ = 1.0)

    rng = MersenneTwister(31)
    ws = [randn(rng, ComplexF64, 3) for _ in 1:3]
    θtrue = 0.02 .* randn(rng, N)
    Vtrue = evaluate_basis(basis, θtrue, (x,))
    problem = InverseProblem(fs, A, B, basis, noise, ws, [0.3])
    problem = generate_data(problem, Vtrue; σμ = 0.0)

    θ = θtrue .+ 0.005 .* randn(rng, N)
    _, g = loss_and_grad(problem, θ; ε = 1e-3)
    h = 1e-6
    gfd = similar(g)
    for k in eachindex(θ)
        e = zeros(N)
        e[k] = h
        Jp, _ = loss_and_grad(problem, θ .+ e; ε = 1e-3)
        Jm, _ = loss_and_grad(problem, θ .- e; ε = 1e-3)
        gfd[k] = (Jp - Jm) / (2h)
    end
    @test g ≈ gfd rtol = 1e-4 atol = 1e-6
end

@testset "Grid basis: recovery and gauge" begin
    L, N = 2π, 16
    x = collect(0:N-1) .* (L / N)
    fs = SplitStepSchrodinger((x,); nsteps=12)
    A = FourierInput(1, L)
    B = PointDensity()
    basis = GridBasis((x,))
    noise = GaussianNoise(σw = 0, σμ = 1.0)

    rng = MersenneTwister(33)
    ws = [randn(rng, ComplexF64, 3) for _ in 1:8]
    Vtrue = 0.05 .* cos.(x) .+ 0.03 .* sin.(2 .* x)

    problem = InverseProblem(fs, A, B, basis, noise, ws, [0.3])
    problem = generate_data(problem, Vtrue; σμ = 0.0)
    solver = TikhonovSolver(1e-8;
        options = Optim.Options(iterations = 200, g_tol = 1e-10))
    result = fit!(solver, problem)
    @test result.converged
    @test relative_error(result.V, Vtrue) < 1e-2

    shifted = generate_data(
        InverseProblem(fs, A, B, basis, noise, ws, [0.3]),
        Vtrue .+ 0.37; σμ = 0.0)
    result_shifted = fit!(solver, shifted)
    @test relative_error(result_shifted.V, Vtrue) < 1e-2
end
