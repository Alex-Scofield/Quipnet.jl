# Neutral-hydrogen primitives (Z = 1, ħ = m = e = 1, a₀ = 1).

_box_center(grids::NTuple{D,Any}) where D =
    ntuple(d -> (first(grids[d]) + last(grids[d])) / 2, D)

function _grid_periods(grids::NTuple{D,Any}) where D
    return ntuple(d -> length(grids[d]) * (grids[d][2] - grids[d][1]), D)
end

_default_softening(grids::NTuple{D,Any}) where D =
    maximum(grids[d][2] - grids[d][1] for d in 1:D)

function _center_vector(center, D::Integer)
    if center isa Number
        D == 1 || throw(ArgumentError(
            "center must have $D component(s), got a scalar"))
        return [Float64(center)]
    end
    c = collect(Float64, center)
    length(c) == D || throw(ArgumentError(
        "center must have $D component(s), got $(length(c))"))
    return c
end

# Squared periodic minimum-image distance on the Cartesian product grid.
function _minimum_image_sq(grids::NTuple{D,Any}, center::Vector{Float64},
        dims) where D
    periods = _grid_periods(grids)
    rsq = zeros(Float64, dims)
    for d in 1:D
        p = periods[d]
        delta = grids[d] .- center[d]
        delta = delta .- p .* round.(delta ./ p)
        shape = ntuple(i -> i == d ? length(delta) : 1, D)
        rsq .+= reshape(abs2.(delta), shape)
    end
    return rsq
end

"""
    hydrogen_energy(n=1)

Energy of the neutral-hydrogen level `n`, `Eₙ = -1 / (2n²)`.
"""
function hydrogen_energy(n::Integer = 1)
    n ≥ 1 || throw(ArgumentError("n must be ≥ 1"))
    return -1 / (2 * n^2)
end

"""
    hydrogen_ground_state(grids; center=nothing, normalize=true)

Analytic ground state of neutral hydrogen, `ψ₁₀₀(x) = exp(-|x - center|)/√π`
(`ħ = m = e = 1`, `a₀ = 1`), sampled on the Cartesian grid `grids`. With
`normalize=true` the state has unit `L²` norm on the grid. Distances use the
periodic minimum image, so the state is consistent with the periodic solver.
"""
function hydrogen_ground_state(grids::NTuple{D,Any};
        center = _box_center(grids), normalize::Bool = true) where D
    all(g -> length(g) ≥ 2, grids) || throw(ArgumentError(
        "grids must have at least two points per axis"))
    c = _center_vector(center, D)
    dims = map(length, grids)
    rsq = _minimum_image_sq(grids, c, dims)
    ψ = exp.(-sqrt.(rsq)) ./ sqrt(π)
    if normalize
        volume = prod(grids[d][2] - grids[d][1] for d in 1:D)
        nrm = sqrt(sum(abs2, ψ) * volume)
        nrm > 0 || throw(ArgumentError("cannot normalize the zero state"))
        ψ ./= nrm
    end
    return ψ
end

"""
    hydrogen_potential(grids; center=nothing, softening=nothing, taper=nothing)

Softened proton Coulomb potential, `-1 / sqrt(|x - center|² + softening²)`,
sampled on the grid. `softening` defaults to the largest grid step. A compact
raised-cosine window reaches zero at half the box width, so the potential is
smooth and periodic; `taper` overrides that cutoff radius. Distances use the
periodic minimum image.
"""
function hydrogen_potential(grids::NTuple{D,Any};
        center = _box_center(grids),
        softening = nothing,
        taper = nothing) where D
    all(g -> length(g) ≥ 2, grids) || throw(ArgumentError(
        "grids must have at least two points per axis"))
    a = softening === nothing ? _default_softening(grids) : Float64(softening)
    a > 0 || throw(ArgumentError("softening must be > 0"))
    c = _center_vector(center, D)
    r2 = taper === nothing ? 0.5 * minimum(_grid_periods(grids)) : Float64(taper)
    r2 > 0 || throw(ArgumentError("taper must be > 0"))
    r1 = 0.7 * r2
    dims = map(length, grids)
    rsq = _minimum_image_sq(grids, c, dims)
    r = sqrt.(rsq)
    window = @. ifelse(r ≤ r1, 1.0,
        ifelse(r ≥ r2, 0.0, 0.5 * (1 + cos(π * (r - r1) / (r2 - r1)))))
    return -window ./ sqrt.(rsq .+ a^2)
end
