abstract type AbstractForwardSolver end

"""Exact free-particle propagator on a periodic grid."""
struct FreeSchrodinger{G} <: AbstractForwardSolver
    grids::G
end

"""
    SplitStepSchrodinger(grids; nsteps=nothing, dt=nothing)

Lie–Trotter split-step propagator on a periodic grid. Provide exactly one of
`nsteps` (fixed number of steps per call) or `dt` (step size, with
`ceil(T/dt)` steps per trajectory).
"""
struct SplitStepSchrodinger{G} <: AbstractForwardSolver
    grids::G
    nsteps::Union{Int,Nothing}
    dt::Union{Float64,Nothing}
end

function SplitStepSchrodinger(grids; nsteps=nothing, dt=nothing)
    if (nsteps === nothing) == (dt === nothing)
        throw(ArgumentError("provide exactly one of nsteps or dt"))
    end
    if nsteps !== nothing
        nsteps isa Integer || throw(ArgumentError("nsteps must be an integer"))
        nsteps ≥ 1 || throw(ArgumentError("nsteps must be ≥ 1"))
    end
    if dt !== nothing
        dt isa Real || throw(ArgumentError("dt must be a real number"))
        dt > 0 || throw(ArgumentError("dt must be > 0"))
    end
    return SplitStepSchrodinger(grids,
        nsteps === nothing ? nothing : Int(nsteps),
        dt === nothing ? nothing : Float64(dt))
end

forward_grids(fs::AbstractForwardSolver) = fs.grids

function _nsteps(fs::SplitStepSchrodinger, T)
    T ≥ 0 || throw(ArgumentError("T must be ≥ 0"))
    return fs.nsteps === nothing ? max(1, ceil(Int, T / fs.dt)) : fs.nsteps
end

"""
    propagate(fs, ψ₀, V, T)

Evolve `ψ₀` under the solver `fs` for time `T`, returning the final state.
`V` is a callable or an array sampled on the solver grid.
"""
function propagate(fs::FreeSchrodinger, ψ₀, V, T)
    T ≥ 0 || throw(ArgumentError("T must be ≥ 0"))
    return free_schrod_periodic(ψ₀, fs.grids, T)
end

function propagate(fs::SplitStepSchrodinger, ψ₀, V, T)
    n = _nsteps(fs, T)
    return schrod_periodic(ψ₀, V, fs.grids, T; nsteps=n)
end

# Internal: forward trajectory used by the discrete adjoint. The states are
# simulated, never measured; only the final B(ψ) is compared with data.
struct Trajectory{S,TS,ST,K,V}
    solver::S
    times::TS
    states::ST
    Kfac::K
    Vfac::V
    dt::Float64
end

function _propagate_with_history(
        fs::SplitStepSchrodinger, ψ₀, V, T)
    n = _nsteps(fs, T)
    dims = map(length, fs.grids)
    ψ = ComplexF64.(_sample_on_grid(ψ₀, fs.grids, dims))
    Vvals = _sample_on_grid(V, fs.grids, dims)
    Δt = T / n
    Kfac, Vfac = _splitstep_factors(fs.grids, dims, Vvals, Δt)
    states = Vector{typeof(ψ)}(undef, n + 1)
    states[1] = ψ
    for j in 1:n
        ψ = _step_forward(ψ, Kfac, Vfac)
        states[j+1] = ψ
    end
    times = collect(range(0.0, Float64(T), length=n+1))
    return Trajectory(fs, times, states, Kfac, Vfac, Δt)
end

function _propagate_with_history(fs::FreeSchrodinger, ψ₀, V, T)
    T ≥ 0 || throw(ArgumentError("T must be ≥ 0"))
    dims = map(length, fs.grids)
    ψ0 = ComplexF64.(_sample_on_grid(ψ₀, fs.grids, dims))
    ψ = free_schrod_periodic(ψ0, fs.grids, T)
    return Trajectory(fs, [0.0, Float64(T)], [ψ0, ψ], nothing, nothing,
        Float64(T))
end
