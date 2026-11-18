using Quipnet
using Test
using Optim

@testset "Hydrogen: ground state" begin
    L, N = 24.0, 48
    dx = L / N
    x = collect(0:N-1) .* dx
    grids = (x, x, x)
    c = (12.0, 12.0, 12.0)

    ψ = hydrogen_ground_state(grids; center=c)
    @test size(ψ) == (N, N, N)
    @test argmax(ψ) == CartesianIndex(25, 25, 25)
    @test sum(abs2, ψ) * dx^3 ≈ 1 rtol = 1e-10

    raw = hydrogen_ground_state(grids; center=c, normalize=false)
    @test raw[25, 25, 25] ≈ 1 / sqrt(π)
    @test raw[27, 25, 25] ≈ exp(-1) / sqrt(π)
    @test raw[25, 27, 25] ≈ exp(-1) / sqrt(π)
    @test raw[25, 25, 27] ≈ exp(-1) / sqrt(π)

    @test hydrogen_energy(1) ≈ -0.5
    @test hydrogen_energy(2) ≈ -0.125
end

@testset "Hydrogen: proton potential" begin
    L, N = 24.0, 48
    dx = L / N
    x = collect(0:N-1) .* dx
    grids = (x, x, x)
    c = (12.0, 12.0, 12.0)
    a = 0.5

    VH = hydrogen_potential(grids; center=c, softening=a)
    @test size(VH) == (N, N, N)
    @test all(isfinite, VH)
    @test VH[25, 25, 25] ≈ -1 / a
    @test VH[27, 25, 25] ≈ -1 / sqrt(1 + a^2)
    @test abs(VH[1, 25, 25]) < 1e-12
    @test all(isfinite, hydrogen_potential(grids))

    @test_throws ArgumentError hydrogen_potential(grids; softening=0.0)
    @test_throws ArgumentError hydrogen_potential((x,); center=[1.0, 2.0])
end

@testset "Hydrogen: propagation" begin
    L, N = 24.0, 48
    dx = L / N
    x = collect(0:N-1) .* dx
    grids = (x, x, x)
    c = (12.0, 12.0, 12.0)
    a = 0.2

    ψ = ComplexF64.(hydrogen_ground_state(grids; center=c))
    VH = hydrogen_potential(grids; center=c, softening=a)

    fs = hydrogen_solver(grids; nsteps=4)
    ψfree = propagate(fs, ψ, zeros(size(ψ)), 0.5)
    @test ψfree ≈ free_schrod_periodic(ψ, grids, 0.5) rtol = 1e-9

    fs = hydrogen_solver(grids; nsteps=20)
    ψt = propagate(fs, ψ, VH, 0.05)
    @test sum(abs2, ψt) * dx^3 ≈ 1 rtol = 1e-8
    overlap = sum(conj.(ψ) .* ψt) * dx^3
    @test abs(overlap) > 0.95
end

@testset "Hydrogen: placement input and background" begin
    L, N = 20.0, 64
    dx = L / N
    x = collect(0:N-1) .* dx
    grids = (x,)
    A = PlacementInput(softening=0.3)

    ψA = apply_input(A, [10.0], grids)
    @test ψA ≈ ComplexF64.(hydrogen_ground_state(grids; center=[10.0]))
    @test sum(abs2, ψA) * dx ≈ 1 rtol = 1e-10

    Vbg = Quipnet.experiment_potential(A, [10.0], grids)
    @test Vbg ≈ hydrogen_potential(grids; center=[10.0], softening=0.3)
    @test Quipnet.experiment_potential(FourierInput(1, L), [1.0 + 0im], grids) ===
        nothing
end

@testset "Hydrogen: inverse plumbing and adjoint" begin
    L, N = 20.0, 64
    dx = L / N
    x = collect(0:N-1) .* dx
    grids = (x,)
    fs = hydrogen_solver(grids; nsteps=6)
    A = PlacementInput(softening=0.3)
    B = PointDensity(1:4:N)
    basis = FourierBasis(2, L)
    noise = GaussianNoise(σw=0, σμ=0.3)

    ws = [[10.0], [12.0]]
    θtrue = [0.02, -0.01, 0.015, 0.005]
    Vtrue = evaluate_basis(basis, θtrue, x)

    problem = InverseProblem(fs, A, B, basis, noise, ws, [0.4])
    problem = generate_data(problem, Vtrue; σμ=0.0)
    @test problem.measurements[1] != problem.measurements[2]

    Vtot = Quipnet._total_potential(problem, Vtrue, ws[1])
    @test Vtot ≈ Vtrue .+ hydrogen_potential(grids; center=ws[1], softening=0.3)

    θ = [0.01, 0.0, -0.01, 0.0]
    _, g = loss_and_grad(problem, θ; ε=1e-3)
    h = 1e-6
    gfd = similar(g)
    for k in eachindex(θ)
        e = zeros(length(θ))
        e[k] = h
        Jp, _ = loss_and_grad(problem, θ .+ e; ε=1e-3)
        Jm, _ = loss_and_grad(problem, θ .- e; ε=1e-3)
        gfd[k] = (Jp - Jm) / (2h)
    end
    @test g ≈ gfd rtol = 1e-4 atol = 1e-6
end
