abstract type AbstractPotentialBasis end

"""
    FourierBasis(modes, L)

Real truncated Fourier basis on a `D`-dimensional periodic box with side
lengths `L`. For each axis `d`, the 1D functions are `1`, `cos(2πkx/L[d])`,
and `sin(2πkx/L[d])` for `k = 1…modes[d]`. The N-D basis is their tensor
product with the global constant removed, so

    dim(b) = ∏(1 + 2 * modes[d]) - 1.

In 1D this is `cos`, `sin` for `k = 1…modes`, with the constant excluded.
Basis coefficients are ordered with the first axis varying fastest.
"""
struct FourierBasis{D}
    modes::NTuple{D,Int}
    L::NTuple{D,Float64}
end

function FourierBasis(modes::NTuple{D,Integer}, L::NTuple{D,Real}) where D
    D ≥ 1 || throw(ArgumentError("at least one dimension is required"))
    all(m -> m ≥ 1, modes) || throw(ArgumentError("modes must be ≥ 1"))
    all(l -> l > 0, L) || throw(ArgumentError("L must be > 0"))
    return FourierBasis{D}(Int.(modes), Float64.(L))
end

FourierBasis(modes::Integer, L::Real) =
    FourierBasis((Int(modes),), (Float64(L),))

FourierBasis(modes::NTuple{D,Integer}, L::Real) where D =
    FourierBasis(modes, ntuple(_ -> Float64(L), D))

FourierBasis(modes::Integer, L::NTuple{D,Real}) where D =
    FourierBasis(ntuple(_ -> Int(modes), D), L)

FourierBasis(modes::Integer, L::Real, D::Integer) =
    FourierBasis(ntuple(_ -> Int(modes), D), ntuple(_ -> Float64(L), D))

dim(b::FourierBasis) = prod(1 .+ 2 .* b.modes) - 1

function _check_coefficients(b::FourierBasis, θ)
    length(θ) == dim(b) || throw(ArgumentError(
        "expected $(dim(b)) potential coefficients, got $(length(θ))"))
    return nothing
end

function _basis_1d_matrix(modes::Int, L::Float64, x::AbstractVector)
    B = Matrix{Float64}(undef, length(x), 1 + 2 * modes)
    B[:, 1] .= 1.0
    for k in 1:modes
        ω = 2π * k / L
        B[:, 2 * k] .= cos.(ω .* x)
        B[:, 2 * k + 1] .= sin.(ω .* x)
    end
    return B
end

"""
    basis_matrix(b, grids)

Return the `prod(length.(grids)) × dim(b)` matrix of the basis functions on
the Cartesian product grid `grids`. In 1D, `grids` may also be a vector.
"""
function basis_matrix(b::FourierBasis{D},
        grids::NTuple{D,Any}) where D
    Bs = ntuple(d -> _basis_1d_matrix(b.modes[d], b.L[d], grids[d]), D)
    Φ = reduce(kron, reverse(Bs))
    return Φ[:, 2:end]
end

basis_matrix(b::FourierBasis{1}, x::AbstractVector) =
    basis_matrix(b, (x,))

"""
    evaluate_basis(b, θ, grids)

Evaluate `V_θ` on the Cartesian product grid `grids`.
"""
function evaluate_basis(b::FourierBasis{D}, θ::AbstractVector,
        grids::NTuple{D,Any}) where D
    _check_coefficients(b, θ)
    Φ = basis_matrix(b, grids)
    return reshape(Φ * θ, map(length, grids))
end

function evaluate_basis(b::FourierBasis{1}, θ::AbstractVector,
        x::AbstractArray)
    _check_coefficients(b, θ)
    Φ = basis_matrix(b, vec(x))
    return reshape(Φ * θ, size(x))
end

function evaluate_basis(b::FourierBasis{1}, θ::AbstractVector,
        x::Number)
    _check_coefficients(b, θ)
    V = 0.0
    for k in 1:b.modes[1]
        ω = 2π * k / b.L[1]
        V += θ[2 * k - 1] * cos(ω * x) + θ[2 * k] * sin(ω * x)
    end
    return V
end

function evaluate_basis(b::FourierBasis{D}, θ::AbstractVector,
        xs::NTuple{D,T}) where {D,T<:Number}
    _check_coefficients(b, θ)
    Φ = basis_matrix(b, ntuple(d -> [xs[d]], D))
    return (Φ * θ)[1]
end

"""
    GridBasis(grids)

Pointwise potential basis on a Cartesian grid. The coefficients `θ` are the
values of the potential at the grid points, ordered `θ = vec(V)` with the
first axis varying fastest (the same ordering as `reshape` and the forward
solver). On the solver grid the basis matrix is the identity, so

    dim(b) = ∏ length(grids[d]).

`grids` must match the forward solver's grid shape; a mismatch raises an
`ArgumentError`. Evaluation is defined only on the grid, not at arbitrary
off-grid points. The basis includes the global constant, which density data
cannot identify: L2 regularization selects a mean, and
[`relative_error`](@ref) removes it by default.
"""
struct GridBasis{D,G} <: AbstractPotentialBasis
    grids::G
end

function GridBasis(grids::NTuple{D,Any}) where D
    D ≥ 1 || throw(ArgumentError("at least one dimension is required"))
    all(g -> g isa AbstractVector, grids) ||
        throw(ArgumentError("grids must be a tuple of vectors"))
    all(g -> length(g) ≥ 1, grids) ||
        throw(ArgumentError("grids must be non-empty"))
    return GridBasis{D,typeof(grids)}(grids)
end

GridBasis(x::AbstractVector) = GridBasis((x,))

dim(b::GridBasis) = prod(length.(b.grids))

function _check_coefficients(b::GridBasis, θ)
    length(θ) == dim(b) || throw(ArgumentError(
        "expected $(dim(b)) potential coefficients, got $(length(θ))"))
    return nothing
end

function _check_grid_shape(b::GridBasis, grids)
    all(g -> g isa AbstractVector, grids) || throw(ArgumentError(
        "grids must be a tuple of vectors"))
    length(grids) == length(b.grids) || throw(ArgumentError(
        "GridBasis has $(length(b.grids)) dimension(s), got " *
        "$(length(grids)) grid(s)"))
    map(length, grids) == map(length, b.grids) || throw(ArgumentError(
        "GridBasis grid shape $(map(length, b.grids)) does not match " *
        "solver grid shape $(map(length, grids))"))
    return nothing
end

function basis_matrix(b::GridBasis, grids::NTuple{D,Any}) where D
    _check_grid_shape(b, grids)
    return Diagonal(ones(Float64, dim(b)))
end

basis_matrix(b::GridBasis{1}, x::AbstractVector) = basis_matrix(b, (x,))

function evaluate_basis(b::GridBasis, θ::AbstractVector,
        grids::NTuple{D,Any}) where D
    _check_coefficients(b, θ)
    _check_grid_shape(b, grids)
    return reshape(collect(θ), map(length, grids))
end

evaluate_basis(b::GridBasis{1}, θ::AbstractVector, x::AbstractVector) =
    evaluate_basis(b, θ, (x,))

function evaluate_basis(b::GridBasis, θ::AbstractVector)
    _check_coefficients(b, θ)
    return reshape(collect(θ), map(length, b.grids))
end

function evaluate_basis(b::GridBasis, θ::AbstractVector,
        xs::NTuple{D,T}) where {D,T<:Number}
    throw(ArgumentError(
        "GridBasis evaluates only on the stored grid; off-grid points " *
        "are not supported"))
end

function evaluate_basis(b::GridBasis{1}, θ::AbstractVector, x::Number)
    throw(ArgumentError(
        "GridBasis evaluates only on the stored grid; off-grid points " *
        "are not supported"))
end
