"""
    relative_error(Vhat, Vtrue; modulo_constant=true)

Relative `L²` error between two potentials sampled on the same grid. Because
density measurements cannot resolve an additive constant, the default removes
each potential's mean before comparing.
"""
function relative_error(Vhat::AbstractArray, Vtrue::AbstractArray;
        modulo_constant::Bool = true)
    size(Vhat) == size(Vtrue) || throw(ArgumentError(
        "potential arrays must have the same size"))
    a = Float64.(Vhat)
    b = Float64.(Vtrue)
    if modulo_constant
        a = a .- mean(a)
        b = b .- mean(b)
    end
    denom = sqrt(sum(abs2, b))
    denom > 0 || throw(ArgumentError("true potential has zero norm"))
    return sqrt(sum(abs2, a .- b)) / denom
end

relative_error(result::InverseResult, Vtrue::AbstractArray; kwargs...) =
    relative_error(result.V, Vtrue; kwargs...)
