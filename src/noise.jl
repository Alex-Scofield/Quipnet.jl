"""
    GaussianNoise(; σw=0.0, σμ=1.0)

Independent additive Gaussian noise with standard deviations `σw` on the
input coefficients and `σμ` on the measurements. v1 uses clean inputs
(`σw = 0`); the field is kept for forward compatibility.
"""
struct GaussianNoise
    σw::Float64
    σμ::Float64
end

function GaussianNoise(; σw::Real = 0.0, σμ::Real = 1.0)
    σw ≥ 0 || throw(ArgumentError("σw must be ≥ 0"))
    σμ > 0 || throw(ArgumentError("σμ must be > 0"))
    return GaussianNoise(Float64(σw), Float64(σμ))
end

function _perturb(w::AbstractArray{<:Complex}, σ, rng)
    return w .+ (σ / sqrt(2)) .* (randn(rng, size(w)) .+
                                 im .* randn(rng, size(w)))
end

function _perturb(w::AbstractArray{<:Real}, σ, rng)
    return w .+ σ .* randn(rng, size(w))
end

_perturb(w::Tuple, σ, rng) = map(x -> _perturb(x, σ, rng), w)
