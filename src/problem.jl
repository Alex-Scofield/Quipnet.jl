"""
    InverseProblem(forward, input, measurement, basis, noise, inputs, times)

Batched inverse problem. `inputs[i]` is one experiment's input, `times` is the
`r`-vector of observation times, and `measurements[i]` is the corresponding
stacked measurement vector (filled by [`generate_data`](@ref)).
"""
struct InverseProblem{S,A,B,P,N,W,M,T}
    forward::S
    input::A
    measurement::B
    basis::P
    noise::N
    inputs::W
    measurements::M
    times::T
end

_times_vector(t::Number) = [Float64(t)]
_times_vector(t) = collect(Float64, t)

function InverseProblem(forward, input, measurement, basis, noise, inputs,
        times)
    ts = _times_vector(times)
    isempty(ts) && throw(ArgumentError("at least one observation time is required"))
    all(≥(0), ts) || throw(ArgumentError("observation times must be ≥ 0"))
    length(inputs) ≥ 1 || throw(ArgumentError("at least one experiment is required"))
    measurements = [Float64[] for _ in inputs]
    return InverseProblem(forward, input, measurement, basis, noise,
        inputs, measurements, ts)
end

"""
    apply_forward(problem, ψ0s, V, w=nothing)

Propagate each state in `ψ0s` for its corresponding time under `V` plus the
experiment background from [`experiment_potential`](@ref) for input `w`.
"""
function _total_potential(problem::InverseProblem, V, w)
    grids = forward_grids(problem.forward)
    Vvals = _sample_on_grid(V, grids, map(length, grids))
    w === nothing && return Vvals
    bg = experiment_potential(problem.input, w, grids)
    return bg === nothing ? Vvals : Vvals .+ bg
end

function apply_forward(problem::InverseProblem, ψ0s, V, w = nothing)
    ψs = ψ0s isa Tuple ? ψ0s : (ψ0s,)
    length(ψs) == length(problem.times) || throw(ArgumentError(
        "input map produced $(length(ψs)) states for $(length(problem.times)) times"))
    Vtot = _total_potential(problem, V, w)
    return ntuple(i -> propagate(problem.forward, ψs[i], Vtot, problem.times[i]),
        length(ψs))
end

"""
    evaluate_potential(problem, θ)

Evaluate the potential basis at the solver grid.
"""
function evaluate_potential(problem::InverseProblem, θ::AbstractVector)
    return evaluate_basis(problem.basis, θ, forward_grids(problem.forward))
end

"""
    generate_data(problem, V; σw=problem.noise.σw, σμ=problem.noise.σμ, rng)

Simulate the experiments under potential `V`, optionally perturb the inputs
and/or measurements, and return a new `InverseProblem` with the measurements.
"""
function generate_data(problem::InverseProblem, V;
        σw::Real = problem.noise.σw,
        σμ::Real = problem.noise.σμ,
        rng::AbstractRNG = Random.default_rng())
    σw ≥ 0 || throw(ArgumentError("σw must be ≥ 0"))
    σμ ≥ 0 || throw(ArgumentError("σμ must be ≥ 0"))
    measurements = Vector{Vector{Float64}}(undef, length(problem.inputs))
    for i in eachindex(problem.inputs)
        w = problem.inputs[i]
        σw > 0 && (w = _perturb(w, σw, rng))
        ψ0s = apply_input(problem.input, w, forward_grids(problem.forward))
        ψTs = apply_forward(problem, ψ0s, V, w)
        μ = apply_measurement(problem.measurement, ψTs)
        if σμ > 0
            μ = μ .+ σμ .* randn(rng, length(μ))
        end
        measurements[i] = μ
    end
    return InverseProblem(problem.forward, problem.input, problem.measurement,
        problem.basis, problem.noise, problem.inputs, measurements,
        problem.times)
end
