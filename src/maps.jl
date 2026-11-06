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
