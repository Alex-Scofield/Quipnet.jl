# Kinetic phase e^{-i|k|²Δt/2} per FFT mode (E = |k|²/2). Axes must be
# uniformly spaced: each step dx sets k = 2π·fftfreq(N)/dx.
function _kinetic_phase(grids, dims, Δt)
    ksq = zeros(Float64, dims)
    for (d, g) in enumerate(grids)
        dx = g[2] - g[1]
        kd = 2π .* fftfreq(length(g)) ./ dx
        shape = ntuple(i -> i == d ? length(kd) : 1, length(grids))
        ksq .+= reshape(abs2.(kd), shape)
    end
    return exp.(-im .* ksq .* (Δt / 2))
end

# Evaluate f on the Cartesian product grid, or pass through a matching array.
function _sample_on_grid(f, grids, dims)
    if f isa AbstractArray
        size(f) == dims || throw(ArgumentError(
            "array of size $(size(f)) does not match grid size $dims"))
        return f
    end
    return reshape([f(c...) for c in Iterators.product(grids...)], dims)
end

"""
    free_schrod_periodic(ψ₀, x, t)
    free_schrod_periodic(ψ₀, (x, y), t)
    free_schrod_periodic(ψ₀, (x, y, z), t)

Evolve `ψ₀` under the free Schrödinger equation `i ∂ₜψ = -½∇²ψ` (ħ = m = 1)
with a spectral FFT propagator. Works in any number of dimensions.

`ψ₀` is a callable `ψ₀(x[, y, z])` evaluated on the grid, or an array already
sampled on the Cartesian product. Pass one grid in 1D or a tuple `(x, y, …)`.
Each axis must be uniformly spaced; its step sets `k = 2π·fftfreq(N)/dx`.
"""
free_schrod_periodic(ψ₀, x, t::Number) = free_schrod_periodic(ψ₀, (x,), t)

# Spectral free-particle propagator,  i ∂ₜψ = -½∇²ψ  (ħ = m = 1).
function free_schrod_periodic(ψ₀, grids::Tuple, t::Number)
    dims = map(length, grids)
    vals = _sample_on_grid(ψ₀, grids, dims)
    return ifft(_kinetic_phase(grids, dims, t) .* fft(vals))
end

"""
    schrod_periodic(ψ₀, V, x, t; nsteps)
    schrod_periodic(ψ₀, V, (x, y), t; nsteps)
    schrod_periodic(ψ₀, V, (x, y, z), t; nsteps)

Evolve `ψ₀` under `i ∂ₜψ = -½∇²ψ + Vψ` (ħ = m = 1) over `[0, t]` in `nsteps`
split-step (Lie–Trotter) steps. Works in any number of dimensions.

Each step applies `e^{-iΔt V}` in real space and `e^{-iΔt k²/2}` in reciprocal
space; the scheme is unitary and first-order in `Δt = t/nsteps`.

`ψ₀` and `V` are callables `f(x[, y, z])` or arrays sampled on the grid, with
the same grid and **periodic**-boundary assumptions as
[`free_schrod_periodic`](@ref).
"""
schrod_periodic(ψ₀, V, x, t::Number; nsteps) = schrod_periodic(ψ₀, V, (x,), t; nsteps=nsteps)

# Split-step (Lie–Trotter) propagator: potential exactly in real space,
# kinetic exactly in reciprocal space; unitary, first-order in Δt.
function schrod_periodic(ψ₀, V, grids::Tuple, t::Number; nsteps)
    nsteps ≥ 1 || throw(ArgumentError("nsteps must be ≥ 1"))
    dims = map(length, grids)
    ψ = ComplexF64.(_sample_on_grid(ψ₀, grids, dims))
    Vvals = _sample_on_grid(V, grids, dims)
    dt = t / nsteps
    Vfac = exp.(-im .* Vvals .* dt)
    Kfac = _kinetic_phase(grids, dims, dt)
    for _ in 1:nsteps
        ψ = Vfac .* ψ
        ψ = ifft(Kfac .* fft(ψ))
    end
    return ψ
end
