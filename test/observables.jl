using Quipnet
using Test
using Random
using Optim
using FFTW

@testset "Observables: momentum density basics" begin
    L, N = 2π, 16
    x = collect(0:N-1) .* (L / N)
    ψ = ones(ComplexF64, N) ./ sqrt(N)

    B = MomentumDensity()
    μ = apply_measurement(B, ψ)
    @test length(μ) == N
    @test sum(μ) ≈ sum(abs2, ψ)
    @test argmax(μ) == 1          # zero momentum sits at FFT index 1

    Bs = MomentumDensity(shift = true)
    @test apply_measurement(Bs, ψ) ≈ fftshift(μ)

    Bsub = MomentumDensity([1, 3, 5])
    @test apply_measurement(Bsub, ψ) ≈ μ[[1, 3, 5]]

    # A plane wave puts all weight on a single momentum mode.
    m = 3
    ψk = cis.(2π * m .* x ./ L) ./ sqrt(N)
    μk = apply_measurement(MomentumDensity(), ψk)
    @test argmax(μk) == m + 1
    @test maximum(μk) ≈ 1.0
end

@testset "Observables: momentum adjoint" begin
    L, N = 2π, 16
    x = collect(0:N-1) .* (L / N)
    fs = SplitStepSchrodinger((x,); nsteps=4)
    A = FourierInput(1, L)
    basis = FourierBasis(2, L)
    noise = GaussianNoise(σw = 0, σμ = 0.3)

    rng = MersenneTwister(21)
    w = randn(rng, ComplexF64, 3)
    θtrue = [0.05, -0.03, 0.02, 0.01]
    Vtrue = evaluate_basis(basis, θtrue, x)
    θ = [0.01, 0.0, -0.01, 0.0]

    for B in (MomentumDensity(), MomentumDensity(shift = true),
              CompositeMeasurement(PointDensity(), MomentumDensity()),
              CompositeMeasurement(PointDensity(1:2:N),
                  MomentumDensity(shift = true)))
        problem = InverseProblem(fs, A, B, basis, noise, [w], [0.2])
        problem = generate_data(problem, Vtrue; σμ = 0.0)
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
end

@testset "Observables: composite layout" begin
    L, N = 2π, 16
    x = collect(0:N-1) .* (L / N)
    rng = MersenneTwister(5)
    ψ1 = randn(rng, ComplexF64, N)
    ψ2 = randn(rng, ComplexF64, N)

    B = CompositeMeasurement(PointDensity(), MomentumDensity())
    μ1 = apply_measurement(B, ψ1)
    μ2 = apply_measurement(B, ψ2)
    @test length(μ1) == 2N
    @test μ1 ≈ vcat(apply_measurement(PointDensity(), ψ1),
                    apply_measurement(MomentumDensity(), ψ1))
    @test apply_measurement(B, (ψ1, ψ2)) ≈ vcat(μ1, μ2)

    @test_throws ArgumentError CompositeMeasurement()
    @test_throws ArgumentError CompositeMeasurement((PointDensity(), 1))
end
