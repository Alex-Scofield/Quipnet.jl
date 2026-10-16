using Quipnet
using Test

@testset "Free Schrödinger Equation" begin
    xs = range(0, 2π, 100)
    @test free_schrod_periodic(x->0, xs, 10.) == zeros(100)
end

@testset "Gaussian packet vs analytic solution" begin
    N, σ, x₀ = 256, 4.0, 128.0
    x = collect(0.0:N-1)
    ψ₀ = x -> exp(-(x - x₀)^2 / (4σ^2))
    ψ_exact(t) = (a = σ^2 + im*t/2; σ/sqrt(a) .* exp.(-(x .- x₀).^2 ./ (4a)))
    for t in (0.0, 1.0, 10.0, 50.0)
        @test free_schrod_periodic(ψ₀, x, t) ≈ ψ_exact(t) rtol=1e-10
    end
end

@testset "Boosted Gaussian (arbitrary momentum k₀) vs analytic" begin
    # Galilean boost: if φ solves the free SE, so does
    #   ψ(x,t) = e^{i(k₀x − k₀²t/2)} φ(x − k₀t, t)
    # so the packet drifts at the group velocity k₀ while its envelope is
    # otherwise unchanged. k₀ is deliberately NOT a grid wavenumber (2πm/N):
    # the Gaussian envelope keeps the method spectrally accurate at the
    # boundaries, so the comparison still holds to ~1e-10 — hence the looser
    # rtol. Tests here must stay wrap-free: x₀ + k₀t inside [0, 255].
    N, σ, x₀ = 256, 4.0, 128.0
    x = collect(0.0:N-1)
    φ₀ = x -> exp(-(x - x₀)^2 / (4σ^2))
    function ψ_boosted(k₀, t)
        a = σ^2 + im*t/2
        sqrt(σ^2 / a) .* exp.(im .* (k₀ .* x .- (k₀^2 .* t) ./ 2)) .*
            exp.(-(x .- x₀ .- k₀ .* t).^2 ./ (4a))
    end
    cases = ((0.5, 25.0), (-1.3, 25.0), (2.0, 20.0), (1.1, 50.0))

    @testset "complex solution vs analytic" begin
        for k₀ in (-1.7, -0.6, 0.4, 1.1, 2.0), t in (0.0, 1.0, 5.0, 10.0, 20.0)
            got = free_schrod_periodic(x -> exp(im * k₀ * x) * φ₀(x), x, t)
            @test got ≈ ψ_boosted(k₀, t) rtol=1e-8
        end
    end

    @testset "density rigidity (envelope moves rigidly at k₀)" begin
        for (k₀, t) in cases
            got = free_schrod_periodic(x -> exp(im * k₀ * x) * φ₀(x), x, t)
            @test abs2.(got) ≈ abs2.(ψ_boosted(k₀, t)) rtol=1e-8
        end
    end

    @testset "group velocity ⟨x⟩ = x₀ + k₀t" begin
        for (k₀, t) in cases
            d = abs2.(free_schrod_periodic(x -> exp(im * k₀ * x) * φ₀(x), x, t))
            @test sum(x .* d) / sum(d) ≈ x₀ + k₀ * t rtol=1e-8
        end
    end
end

@testset "2D free Gaussian vs analytic" begin
    # 2D propagator factorises, so no extra normalisation beyond (σ²/a).
    N, σ = 128, 4.0
    c = (N - 1) / 2
    x = collect(0.0:N-1)
    ψ₀ = (x, y) -> exp(-((x - c)^2 + (y - c)^2) / (4σ^2))
    ψ_exact(t) = (a = σ^2 + im*t/2;
        (σ^2 / a) .* exp.(-((x .- c).^2 .+ (x' .- c).^2) ./ (4a)))
    for t in (0.0, 1.0, 5.0, 10.0, 20.0)
        @test free_schrod_periodic(ψ₀, (x, x), t) ≈ ψ_exact(t) rtol=1e-10
    end
end

@testset "2D boosted Gaussian: group velocity in both axes" begin
    N, σ = 128, 4.0
    c = (N - 1) / 2
    x = collect(0.0:N-1)
    y = collect(0.0:N-1)
    kx, ky = 0.7, -1.1
    ψ₀ = (x, y) -> exp(im * (kx * x + ky * y)) *
        exp(-((x - c)^2 + (y - c)^2) / (4σ^2))
    for t in (0.0, 5.0, 10.0)
            d = abs2.(free_schrod_periodic(ψ₀, (x, y), t))
        @test sum(x .* d) / sum(d) ≈ c + kx * t rtol=1e-8
        @test sum(y' .* d) / sum(d) ≈ c + ky * t rtol=1e-8
    end
end

@testset "3D free Gaussian vs analytic" begin
    N, σ = 96, 4.0
    c = (N - 1) / 2
    x = collect(0.0:N-1)
    r2 = reshape((x .- c).^2, N, 1, 1) .+
         reshape((x .- c).^2, 1, N, 1) .+
         reshape((x .- c).^2, 1, 1, N)
    ψ₀ = (x, y, z) -> exp(-((x - c)^2 + (y - c)^2 + (z - c)^2) / (4σ^2))
    ψ_exact(t) = (a = σ^2 + im*t/2;
        (σ^2 / a)^(3 / 2) .* exp.(-r2 ./ (4a)))
    for t in (0.0, 1.0, 5.0, 10.0)
        @test free_schrod_periodic(ψ₀, (x, x, x), t) ≈ ψ_exact(t) rtol=1e-10
    end
end

@testset "1D non-unit spacing (dx = 0.5) vs analytic" begin
    # dx previously assumed to be 1; the physical step must change the physics.
    dx, N, σ = 0.5, 256, 2.0
    x = dx .* collect(0.0:N-1)
    xc = (N ÷ 2) * dx
    ψ₀ = x -> exp(-(x - xc)^2 / (4σ^2))
    ψ_exact(t) = (a = σ^2 + im*t/2; sqrt(σ^2 / a) .* exp.(-(x .- xc).^2 ./ (4a)))
    for t in (0.0, 1.0, 10.0)
        @test free_schrod_periodic(ψ₀, x, t) ≈ ψ_exact(t) rtol=1e-10
    end
end

@testset "2D anisotropic spacing (dx ≠ dy) vs analytic" begin
    dx, dy, N, σ = 0.5, 0.8, 128, 2.0
    x = dx .* collect(0.0:N-1)
    y = dy .* collect(0.0:N-1)
    xc, yc = (N ÷ 2) * dx, (N ÷ 2) * dy
    ψ₀ = (x, y) -> exp(-((x - xc)^2 + (y - yc)^2) / (4σ^2))
    ψ_exact(t) = (a = σ^2 + im*t/2;
        (σ^2 / a) .* exp.(-((x .- xc).^2 .+ (y' .- yc).^2) ./ (4a)))
    for t in (0.0, 1.0, 10.0)
        @test free_schrod_periodic(ψ₀, (x, y), t) ≈ ψ_exact(t) rtol=1e-10
    end
end

@testset "group velocity with non-unit spacing" begin
    dx, N, σ, k₀ = 0.5, 256, 2.0, 0.9
    x = dx .* collect(0.0:N-1)
    xc = (N ÷ 2) * dx
    ψ₀ = x -> exp(im * k₀ * x) * exp(-(x - xc)^2 / (4σ^2))
    for t in (0.0, 5.0, 10.0)
        d = abs2.(free_schrod_periodic(ψ₀, x, t))
        @test sum(x .* d) / sum(d) ≈ xc + k₀ * t rtol=1e-8
    end
end

@testset "Schrödinger with potential (split-step)" begin
    N, σ, x₀ = 256, 4.0, 128.0
    x = collect(0.0:N-1)
    ψ₀ = x -> exp(-(x - x₀)^2 / (4σ^2))

    @testset "V = 0 reduces exactly to the free propagator" begin
        for t in (0.0, 5.0, 20.0)
            @test schrod_periodic(ψ₀, x -> 0.0, x, t; nsteps=50) ≈
                free_schrod_periodic(ψ₀, x, t) rtol=1e-10
        end
    end

    @testset "constant potential is a global phase e^{-iVt}" begin
        V0 = 0.7
        for t in (1.0, 10.0)
            @test schrod_periodic(ψ₀, x -> V0, x, t; nsteps=30) ≈
                exp(-im * V0 * t) .* free_schrod_periodic(ψ₀, x, t) rtol=1e-10
        end
    end

    @testset "array inputs agree with callable inputs" begin
        ψarr = ψ₀.(x)
        @test schrod_periodic(ψarr, fill(0.3, N), x, 5.0; nsteps=40) ≈
            schrod_periodic(ψ₀, x -> 0.3, x, 5.0; nsteps=40)
    end

    @testset "unitarity: L2 norm preserved at finite Δt" begin
        V = x -> 0.5 * (x - x₀)^2 / N   # smooth, modest potential
        for nsteps in (1, 7, 100)
            ψt = schrod_periodic(ψ₀, V, x, 12.0; nsteps=nsteps)
            @test sum(abs2, ψt) ≈ sum(abs2, ψ₀.(x)) rtol=1e-10
        end
    end

    @testset "first-order convergence in Δt" begin
        # Smooth periodic potential on a periodic box; compare successive
        # refinements (no reference bias). For O(Δt) errors the difference
        # between consecutive solutions should halve each time ns doubles.
        xg = range(0, 2π, length=256)
        ψg = x -> exp(-(x - π)^2 / (4 * 0.3^2))
        Vg = x -> cos(x)
        sols = Dict(ns => schrod_periodic(ψg, Vg, xg, 4.0; nsteps=ns)
                    for ns in (64, 128, 256, 512, 1024))
        diffs = [sqrt(sum(abs2, sols[ns] - sols[2ns]))
                 for ns in (64, 128, 256, 512)]
        for i in 1:3
            @test 1.8 < diffs[i] / diffs[i+1] < 2.2
        end
    end
end

@testset "2D Schrödinger with potential" begin
    N, σ = 128, 4.0
    c = (N - 1) / 2
    x = collect(0.0:N-1)
    ψ₀ = (x, y) -> exp(-((x - c)^2 + (y - c)^2) / (4σ^2))
    V = (x, y) -> 0.1 * (cos(2π * (x - c) / N) + cos(2π * (y - c) / N))

    @test sum(abs2, schrod_periodic(ψ₀, V, (x, x), 5.0; nsteps=50)) ≈
        sum(abs2, ψ₀.(x, x')) rtol=1e-10

    @test schrod_periodic(ψ₀.(x, x'), V, (x, x), 5.0; nsteps=30) ≈
        schrod_periodic(ψ₀, V, (x, x), 5.0; nsteps=30)
end
