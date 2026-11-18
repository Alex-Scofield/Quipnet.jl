abstract type AbstractInputMap end

"""
    FourierInput(modes, L=2π; normalize=true)

Map complex Fourier coefficients `w` (modes `-modes…modes` on each axis) to a
periodic initial state sampled on a `D`-dimensional grid. Coefficients follow
`Iterators.product((-modes):modes, ...)` order, with the first axis varying
fastest. With `normalize=true` the state is scaled to unit `L²` norm.
"""
struct FourierInput{D}
    modes::NTuple{D,Int}
    L::NTuple{D,Float64}
    normalize::Bool
end

function FourierInput(modes::NTuple{D,Integer}, L::NTuple{D,Real};
        normalize::Bool = true) where D
    D ≥ 1 || throw(ArgumentError("at least one dimension is required"))
    all(m -> m ≥ 0, modes) || throw(ArgumentError("modes must be ≥ 0"))
    all(l -> l > 0, L) || throw(ArgumentError("L must be > 0"))
    return FourierInput{D}(Int.(modes), Float64.(L), normalize)
end

FourierInput(modes::Integer, L::Real = 2π; normalize::Bool = true) =
    FourierInput((Int(modes),), (Float64(L),); normalize = normalize)

FourierInput(modes::NTuple{D,Integer}, L::Real;
        normalize::Bool = true) where D =
    FourierInput(modes, ntuple(_ -> Float64(L), D); normalize = normalize)

FourierInput(modes::Integer, L::NTuple{D,Real};
        normalize::Bool = true) where D =
    FourierInput(ntuple(_ -> Int(modes), D), L; normalize = normalize)

FourierInput(modes::Integer, L::Real, D::Integer;
        normalize::Bool = true) =
    FourierInput(ntuple(_ -> Int(modes), D),
        ntuple(_ -> Float64(L), D); normalize = normalize)

function apply_input(A::FourierInput{D}, w::AbstractVector,
        grids::NTuple{D,Any}) where D
    ncoef = prod(2 .* A.modes .+ 1)
    length(w) == ncoef || throw(ArgumentError(
        "expected $ncoef coefficients, got $(length(w))"))

    dims = map(length, grids)
    ψ = zeros(ComplexF64, dims)
    ranges = map(m -> (-m):m, A.modes)
    idx = 1
    for ks in Iterators.product(ranges...)
        p = ntuple(d -> cis.(2π * ks[d] / A.L[d] .* grids[d]), D)
        ψk = reduce(kron, reverse(p))
        ψ .+= w[idx] .* reshape(ψk, dims)
        idx += 1
    end

    if A.normalize
        volume = prod(abs(grids[d][2] - grids[d][1]) for d in 1:D)
        nrm = sqrt(sum(abs2, ψ) * volume)
        nrm > 0 || throw(ArgumentError("cannot normalize the zero state"))
        ψ ./= nrm
    end
    return ψ
end

"""
    BatchInput(maps; split=identity)

Input map for `r > 1` trajectories. `split(w)` must return a tuple of
coefficient blocks, one per map in `maps`.
"""
struct BatchInput{M,F} <: AbstractInputMap
    maps::M
    split::F
end

BatchInput(maps; split = identity) = BatchInput(maps, split)

function apply_input(A::BatchInput, w, grids::Tuple)
    ws = A.split(w)
    length(ws) == length(A.maps) || throw(ArgumentError(
        "split returned $(length(ws)) blocks for $(length(A.maps)) maps"))
    return ntuple(i -> apply_input(A.maps[i], ws[i], grids), length(A.maps))
end

"""
    PlacementInput(; softening=nothing, taper=nothing, normalize=true)

Input map that places a neutral hydrogen atom at the per-experiment location
`w` (a length-`D` real vector of grid coordinates). The initial state is the
analytic 1s orbital centered at `w`; the matching proton Coulomb background is
returned by [`experiment_potential`](@ref). `softening` and `taper` are passed
through to [`hydrogen_potential`](@ref).
"""
struct PlacementInput <: AbstractInputMap
    softening::Union{Nothing,Float64}
    taper::Union{Nothing,Float64}
    normalize::Bool
end

PlacementInput(; softening = nothing, taper = nothing, normalize::Bool = true) =
    PlacementInput(softening === nothing ? nothing : Float64(softening),
        taper === nothing ? nothing : Float64(taper), normalize)

function apply_input(A::PlacementInput, w, grids::NTuple{D,Any}) where D
    c = _center_vector(w, D)
    return ComplexF64.(hydrogen_ground_state(grids; center = c,
        normalize = A.normalize))
end

"""
    experiment_potential(A, w, grids)

Fixed background potential contributed by the experiment setup for input `w`,
or `nothing`. [`PlacementInput`](@ref) returns the proton Coulomb potential
centered at `w`.
"""
experiment_potential(A, w, grids) = nothing

experiment_potential(A::PlacementInput, w, grids::NTuple{D,Any}) where D =
    hydrogen_potential(grids; center = _center_vector(w, D),
        softening = A.softening, taper = A.taper)

abstract type AbstractMeasurementMap end

"""
    PointDensity(indices=Colon())

Measurement map `ψ ↦ |ψ|²` at `indices`. Indices may be linear indices or
`CartesianIndex` values; `Colon()` means all grid points. For a tuple of states
the results are concatenated.
"""
struct PointDensity{I} <: AbstractMeasurementMap
    indices::I
end

PointDensity() = PointDensity(Colon())

function apply_measurement(B::PointDensity, ψ::AbstractArray)
    return abs2.(ψ)[B.indices]
end

function apply_measurement(B::PointDensity, ψs::Tuple)
    return vcat((apply_measurement(B, s) for s in ψs)...)
end

# Adjoint of B at the endpoint: ∂ℓ/∂ψ*, where ℓ = ‖B(ψ) - μ‖² / σ².
function measurement_adjoint(B::PointDensity, ψ::AbstractArray,
        r::AbstractVector, σ²::Real)
    g = zeros(ComplexF64, size(ψ))
    if B.indices isa Colon
        g .= 2 .* reshape(r, size(ψ)) .* ψ ./ σ²
    else
        idx = B.indices
        g[idx] .= 2 .* r .* ψ[idx] ./ σ²
    end
    return g
end

"""
    MomentumDensity(indices=Colon(); shift=false)

Measurement map `ψ ↦ |F(ψ)|² / N`, where `F` is the unnormalized FFT and
`N = length(ψ)` (the product of the grid dimensions). This is the unitary
momentum density: its sum equals `sum(abs2, ψ)`, the same total as
[`PointDensity`](@ref). `indices` selects momentum modes in FFT order; with
`shift=true`, `fftshift` is applied first and indices refer to the centered
(per-axis) momentum ordering. For a tuple of states the results are
concatenated.
"""
struct MomentumDensity{I} <: AbstractMeasurementMap
    indices::I
    shift::Bool
end

function MomentumDensity(indices = Colon(); shift::Bool = false)
    return MomentumDensity{typeof(indices)}(indices, shift)
end

function apply_measurement(B::MomentumDensity, ψ::AbstractArray)
    n = abs2.(fft(ψ)) ./ length(ψ)
    B.shift && (n = fftshift(n))
    return vec(n[B.indices])
end

function apply_measurement(B::MomentumDensity, ψs::Tuple)
    return vcat((apply_measurement(B, s) for s in ψs)...)
end

function _measurement_residual_array(B::MomentumDensity, ψ::AbstractArray,
        r::AbstractVector)
    rfull = zeros(Float64, size(ψ))
    if B.indices isa Colon
        rfull .= reshape(r, size(ψ))
    else
        rfull[B.indices] .= r
    end
    B.shift && (rfull = ifftshift(rfull))
    return rfull
end

# B = |Fψ|²/N, so ∂B/∂ψ* = Fψ/N and Fᴴ = N·ifft gives
# g = 2·ifft(r ⊙ Fψ)/σ².
function measurement_adjoint(B::MomentumDensity, ψ::AbstractArray,
        r::AbstractVector, σ²::Real)
    rfull = _measurement_residual_array(B, ψ, r)
    return 2 .* ifft(rfull .* fft(ψ)) ./ σ²
end

"""
    CompositeMeasurement(maps...)

Concatenate several measurement maps. `apply_measurement` returns the
concatenation of the component measurements, in order; `measurement_adjoint`
splits the residual by component and sums the component adjoints. For a tuple
of states the per-state blocks are concatenated in the same order as for
[`PointDensity`](@ref).
"""
struct CompositeMeasurement{M} <: AbstractMeasurementMap
    maps::M
end

function CompositeMeasurement(maps::Tuple)
    isempty(maps) && throw(ArgumentError(
        "at least one measurement map is required"))
    all(m -> m isa AbstractMeasurementMap, maps) || throw(ArgumentError(
        "all entries of a CompositeMeasurement must be measurement maps"))
    return CompositeMeasurement{typeof(maps)}(maps)
end

CompositeMeasurement(maps::AbstractVector) = CompositeMeasurement(Tuple(maps))
CompositeMeasurement(maps...) = CompositeMeasurement(maps)

function apply_measurement(B::CompositeMeasurement, ψ::AbstractArray)
    return vcat((apply_measurement(m, ψ) for m in B.maps)...)
end

function apply_measurement(B::CompositeMeasurement, ψs::Tuple)
    return vcat((apply_measurement(B, s) for s in ψs)...)
end

function measurement_adjoint(B::CompositeMeasurement, ψ::AbstractArray,
        r::AbstractVector, σ²::Real)
    g = zeros(ComplexF64, size(ψ))
    offset = 0
    for m in B.maps
        n = length(apply_measurement(m, ψ))
        offset + n ≤ length(r) || throw(ArgumentError(
            "residual is too short for the composite measurement"))
        g .+= measurement_adjoint(m, ψ, r[offset + 1:offset + n], σ²)
        offset += n
    end
    offset == length(r) || throw(ArgumentError(
        "residual length $(length(r)) does not match the composite " *
        "measurement length $offset"))
    return g
end
