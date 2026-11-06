using Quipnet
using Test
using Random
using Optim

@testset "Inverse: Fourier basis and maps" begin
    L = 2π
    b = FourierBasis(2, L)
    @test dim(b) == 4

    x = collect(0:63) .* (L / 64)
    θ = [0.1, -0.2, 0.3, 0.05]
    V = evaluate_basis(b, θ, x)
    @test V ≈ basis_matrix(b, x) * θ
    @test evaluate_basis(b, θ, 0.3) ≈
          sum(basis_matrix(b, [0.3])[1, :] .* θ)

    A = FourierInput(1, L)
    ψ = apply_input(A, [1.0 + 0im, 0.0im, 0.0im], (x,))
    @test sum(abs2, ψ) * (x[2] - x[1]) ≈ 1.0

    B = PointDensity(1:3:length(x))
    μ = apply_measurement(B, ψ)
    @test μ ≈ abs2.(ψ)[1:3:end]
end

@testset "Inverse: analytic adjoint" begin
    L, N = 2π, 32
    x = collect(0:N-1) .* (L / N)
    fs = SplitStepSchrodinger((x,); nsteps=8)
    A = FourierInput(1, L)
    B = PointDensity(1:2:N)          # exercise the sensor-subset adjoint
    basis = FourierBasis(2, L)
    noise = GaussianNoise(σw = 0, σμ = 0.3)

    rng = MersenneTwister(11)
    w = randn(rng, ComplexF64, 3)
    θtrue = [0.05, -0.03, 0.02, 0.01]
    Vtrue = evaluate_basis(basis, θtrue, x)

    problem = InverseProblem(fs, A, B, basis, noise, [w], [0.4])
    problem = generate_data(problem, Vtrue; σμ = 0.0)

    θ = [0.01, 0.0, -0.01, 0.0]
    _, g = loss_and_grad(problem, θ; ε = 1e-3)
    h = 1e-6
    gfd = similar(g)
    for k in eachindex(θ)
        e = zeros(length(θ))
        e[k] = h
        Jp, _ = loss_and_grad(problem, θ .+ e; ε = 1e-3)
        Jm, _ = loss_and_grad(problem, θ .- e; ε = 1e-3)
        gfd[k] = (Jp - Jm) / (2h)
    end
    @test g ≈ gfd rtol = 1e-4 atol = 1e-6
end

@testset "Inverse: recovery (r = 1)" begin
    L, N = 2π, 96
    x = collect(0:N-1) .* (L / N)
    fs = SplitStepSchrodinger((x,); nsteps=30)
    A = FourierInput(2, L)
    B = PointDensity()
    basis = FourierBasis(2, L)
    noise = GaussianNoise(σw = 0, σμ = 1.0)

    rng = MersenneTwister(7)
    ws = [randn(rng, ComplexF64, 5) for _ in 1:8]
    θtrue = [0.05, -0.08, 0.04, 0.03]
    Vtrue = evaluate_basis(basis, θtrue, x)

    problem = InverseProblem(fs, A, B, basis, noise, ws, [0.5])
    problem = generate_data(problem, Vtrue; σμ = 0.0)
    solver = TikhonovSolver(1e-8;
        options = Optim.Options(iterations = 100, g_tol = 1e-10))
    result = fit!(solver, problem)

    @test result.converged
    @test relative_error(result.V, Vtrue) < 1e-3

    # Gauge: V and V + c give the same density data.
    shifted = generate_data(
        InverseProblem(fs, A, B, basis, noise, ws, [0.5]),
        Vtrue .+ 0.37; σμ = 0.0)
    result_shifted = fit!(solver, shifted)
    @test relative_error(result_shifted.V, Vtrue) < 1e-3
end

@testset "Inverse: r = 2 batch" begin
    L, N = 2π, 64
    x = collect(0:N-1) .* (L / N)
    fs = SplitStepSchrodinger((x,); nsteps=20)
    A1 = FourierInput(2, L)
    A2 = BatchInput((A1, A1))
    B = PointDensity()
    basis = FourierBasis(2, L)
    noise = GaussianNoise(σw = 0, σμ = 1.0)

    rng = MersenneTwister(9)
    ws = [(randn(rng, ComplexF64, 5), randn(rng, ComplexF64, 5))
          for _ in 1:4]
    θtrue = [0.05, -0.08, 0.04, 0.03]
    Vtrue = evaluate_basis(basis, θtrue, x)

    problem = InverseProblem(fs, A2, B, basis, noise, ws, [0.25, 0.5])
    problem = generate_data(problem, Vtrue; σμ = 0.0)

    ψ0s = apply_input(A2, ws[1], (x,))
    ψ2 = Quipnet.apply_forward(problem, ψ0s, Vtrue)
    ψa = propagate(fs, apply_input(A1, ws[1][1], (x,)), Vtrue, 0.25)
    ψb = propagate(fs, apply_input(A1, ws[1][2], (x,)), Vtrue, 0.5)
    @test ψ2[1] ≈ ψa
    @test ψ2[2] ≈ ψb

    solver = TikhonovSolver(1e-8;
        options = Optim.Options(iterations = 100, g_tol = 1e-10))
    result = fit!(solver, problem)
    @test result.converged
    @test relative_error(result.V, Vtrue) < 1e-3
end

@testset "Inverse: noise increases error" begin
    L, N = 2π, 64
    x = collect(0:N-1) .* (L / N)
    fs = SplitStepSchrodinger((x,); nsteps=20)
    A = FourierInput(2, L)
    B = PointDensity()
    basis = FourierBasis(2, L)
    noise = GaussianNoise(σw = 0, σμ = 1.0)

    rng = MersenneTwister(13)
    ws = [randn(rng, ComplexF64, 5) for _ in 1:6]
    θtrue = [0.05, -0.08, 0.04, 0.03]
    Vtrue = evaluate_basis(basis, θtrue, x)
    solver = TikhonovSolver(1e-8;
        options = Optim.Options(iterations = 100, g_tol = 1e-10))

    exact = generate_data(
        InverseProblem(fs, A, B, basis, noise, ws, [0.5]),
        Vtrue; σμ = 0.0)
    noisy = generate_data(
        InverseProblem(fs, A, B, basis, noise, ws, [0.5]),
        Vtrue; σμ = 1e-2)

    err_exact = relative_error(fit!(solver, exact).V, Vtrue)
    err_noisy = relative_error(fit!(solver, noisy).V, Vtrue)
    @test err_noisy > err_exact
end

@testset "Inverse: step policies" begin
    L, N = 2π, 32
    x = collect(0:N-1) .* (L / N)
    w = [1.0 + 0im, 0.2im, -0.1 + 0.0im]
    ψ = apply_input(FourierInput(1, L), w, (x,))
    V0 = zeros(N)

    fs_steps = SplitStepSchrodinger((x,); nsteps=10)
    fs_dt = SplitStepSchrodinger((x,); dt=0.05)
    @test propagate(fs_dt, ψ, V0, 0.5) ≈ propagate(fs_steps, ψ, V0, 0.5)

    fs_free = FreeSchrodinger((x,))
    @test propagate(fs_free, ψ, V0, 0.5) ≈
          free_schrod_periodic(ψ, x, 0.5)
end
