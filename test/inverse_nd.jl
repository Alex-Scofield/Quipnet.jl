using Quipnet
using Test
using Random
using Optim

@testset "Inverse: 2D Fourier basis and input" begin
    L = 2π
    b = FourierBasis((1, 1), (L, L))
    @test dim(b) == 8

    x = collect(0:15) .* (L / 16)
    θ = collect(1.0:8)
    Φ = basis_matrix(b, (x, x))
    V = evaluate_basis(b, θ, (x, x))
    @test size(Φ) == (256, 8)
    @test V ≈ reshape(Φ * θ, 16, 16)
    @test evaluate_basis(b, θ, (0.3, 0.4)) ≈
          (basis_matrix(b, ([0.3], [0.4])) * θ)[1]

    A = FourierInput((1, 1), (L, L))
    w = randn(MersenneTwister(1), ComplexF64, 9)
    ψ = apply_input(A, w, (x, x))
    dx = x[2] - x[1]
    @test sum(abs2, ψ) * dx^2 ≈ 1.0

    B = PointDensity([CartesianIndex(1, 2), CartesianIndex(3, 4)])
    @test apply_measurement(B, ψ) ≈ [abs2(ψ[1, 2]), abs2(ψ[3, 4])]
    r = [0.2, -0.1]
    g = Quipnet.measurement_adjoint(B, ψ, r, 2.0)
    @test g[1, 2] ≈ r[1] * ψ[1, 2]
    @test g[3, 4] ≈ r[2] * ψ[3, 4]
    @test count(!iszero, g) == 2

    w .= 0
    w[9] = 1.0
    ψ = apply_input(A, w, (x, x))
    ω = 2π / L
    ψref = cis.(ω .* x) .* cis.(ω .* x')
    ψref ./= sqrt(sum(abs2, ψref) * dx^2)
    @test ψ ≈ ψref
end

@testset "Inverse: 2D analytic adjoint" begin
    L, N = 2π, 16
    x = collect(0:N-1) .* (L / N)
    fs = SplitStepSchrodinger((x, x); nsteps=8)
    A = FourierInput((1, 1), (L, L))
    B = PointDensity()
    basis = FourierBasis((1, 1), (L, L))
    noise = GaussianNoise(σw = 0, σμ = 1.0)

    rng = MersenneTwister(22)
    ws = [randn(rng, ComplexF64, 9) for _ in 1:3]
    θtrue = [0.05, -0.03, 0.02, 0.01, -0.02, 0.04, 0.01, -0.03]
    Vtrue = evaluate_basis(basis, θtrue, (x, x))
    problem = InverseProblem(fs, A, B, basis, noise, ws, [0.25])
    problem = generate_data(problem, Vtrue; σμ = 0.0)

    θ = θtrue .+ 0.01 .* randn(rng, 8)
    _, g = loss_and_grad(problem, θ)
    h = 1e-6
    gfd = similar(g)
    for k in eachindex(θ)
        e = zeros(length(θ))
        e[k] = h
        Jp, _ = loss_and_grad(problem, θ .+ e)
        Jm, _ = loss_and_grad(problem, θ .- e)
        gfd[k] = (Jp - Jm) / (2h)
    end
    @test g ≈ gfd rtol = 1e-4 atol = 1e-6
end

@testset "Inverse: 2D recovery" begin
    L, N = 2π, 24
    x = collect(0:N-1) .* (L / N)
    fs = SplitStepSchrodinger((x, x); nsteps=15)
    A = FourierInput((1, 1), (L, L))
    B = PointDensity()
    basis = FourierBasis((1, 1), (L, L))
    noise = GaussianNoise(σw = 0, σμ = 1.0)

    rng = MersenneTwister(21)
    ws = [randn(rng, ComplexF64, 9) for _ in 1:8]
    θtrue = [0.05, -0.03, 0.02, 0.01, -0.02, 0.04, 0.01, -0.03]
    Vtrue = evaluate_basis(basis, θtrue, (x, x))
    problem = InverseProblem(fs, A, B, basis, noise, ws, [0.3])
    problem = generate_data(problem, Vtrue; σμ = 0.0)

    solver = TikhonovSolver(1e-8;
        options = Optim.Options(iterations = 100, g_tol = 1e-10))
    result = fit!(solver, problem)
    @test result.converged
    @test relative_error(result.V, Vtrue) < 1e-3
end

@testset "Inverse: 3D Fourier basis and input" begin
    L = 2π
    b = FourierBasis((1, 1, 1), (L, L, L))
    @test dim(b) == 26

    x = collect(0:7) .* (L / 8)
    θ = 0.01 .* collect(1.0:26)
    Φ = basis_matrix(b, (x, x, x))
    V = evaluate_basis(b, θ, (x, x, x))
    @test size(Φ) == (512, 26)
    @test V ≈ reshape(Φ * θ, 8, 8, 8)

    A = FourierInput((1, 1, 1), (L, L, L))
    w = randn(MersenneTwister(2), ComplexF64, 27)
    ψ = apply_input(A, w, (x, x, x))
    dx = x[2] - x[1]
    @test sum(abs2, ψ) * dx^3 ≈ 1.0
end

@testset "Inverse: 3D analytic adjoint" begin
    L, N = 2π, 8
    x = collect(0:N-1) .* (L / N)
    fs = SplitStepSchrodinger((x, x, x); nsteps=4)
    A = FourierInput((1, 1, 1), (L, L, L))
    B = PointDensity()
    basis = FourierBasis((1, 1, 1), (L, L, L))
    noise = GaussianNoise(σw = 0, σμ = 1.0)

    rng = MersenneTwister(42)
    ws = [randn(rng, ComplexF64, 27) for _ in 1:4]
    θtrue = 0.02 .* randn(rng, 26)
    Vtrue = evaluate_basis(basis, θtrue, (x, x, x))
    problem = InverseProblem(fs, A, B, basis, noise, ws, [0.2])
    problem = generate_data(problem, Vtrue; σμ = 0.0)

    θ = θtrue .+ 0.005 .* randn(rng, 26)
    _, g = loss_and_grad(problem, θ)
    h = 1e-6
    gfd = similar(g)
    for k in eachindex(θ)
        e = zeros(length(θ))
        e[k] = h
        Jp, _ = loss_and_grad(problem, θ .+ e)
        Jm, _ = loss_and_grad(problem, θ .- e)
        gfd[k] = (Jp - Jm) / (2h)
    end
    @test g ≈ gfd rtol = 1e-4 atol = 1e-6
end

@testset "Inverse: 3D recovery" begin
    L, N = 2π, 12
    x = collect(0:N-1) .* (L / N)
    fs = SplitStepSchrodinger((x, x, x); nsteps=8)
    A = FourierInput((1, 1, 1), (L, L, L))
    B = PointDensity()
    basis = FourierBasis((1, 1, 1), (L, L, L))
    noise = GaussianNoise(σw = 0, σμ = 1.0)

    rng = MersenneTwister(32)
    ws = [randn(rng, ComplexF64, 27) for _ in 1:8]
    θtrue = [0.03 * sin(0.7 * k) for k in 1:26]
    Vtrue = evaluate_basis(basis, θtrue, (x, x, x))
    problem = InverseProblem(fs, A, B, basis, noise, ws, [0.2])
    problem = generate_data(problem, Vtrue; σμ = 0.0)

    solver = TikhonovSolver(1e-8;
        options = Optim.Options(iterations = 100, g_tol = 1e-10))
    result = fit!(solver, problem)
    @test result.converged
    @test relative_error(result.V, Vtrue) < 1e-3
end
