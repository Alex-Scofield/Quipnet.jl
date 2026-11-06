abstract type AbstractInverseSolver end

"""
    TikhonovSolver(ε; regularizer=L2Regularizer(), algorithm=Optim.LBFGS(),
                   options=Optim.Options())

Minimize the regularized data misfit over the potential coefficients with
L-BFGS and the analytic adjoint gradient.
"""
struct TikhonovSolver{R,A,O} <: AbstractInverseSolver
    ε::Float64
    regularizer::R
    algorithm::A
    options::O
end

function TikhonovSolver(ε::Real;
        regularizer::AbstractRegularizer = L2Regularizer(),
        algorithm = Optim.LBFGS(),
        options = Optim.Options())
    ε ≥ 0 || throw(ArgumentError("ε must be ≥ 0"))
    return TikhonovSolver(Float64(ε), regularizer, algorithm, options)
end

struct InverseResult
    θ::Vector{Float64}
    V::Any
    objective::Float64
    iterations::Int
    converged::Bool
end

"""Recover the reconstructed potential from an [`InverseResult`](@ref)."""
reconstruct(result::InverseResult) = result.V

"""
    fit!(solver, problem; θ0=zeros(dim(problem.basis)))

Run the reconstruction and return an [`InverseResult`](@ref).
"""
function fit!(solver::TikhonovSolver, problem::InverseProblem;
        θ0::AbstractVector = zeros(dim(problem.basis)))
    K = dim(problem.basis)
    length(θ0) == K || throw(ArgumentError(
        "expected $K initial coefficients, got $(length(θ0))"))

    function fg!(F, G, θ)
        J, g = loss_and_grad(problem, θ; ε = solver.ε,
            regularizer = solver.regularizer)
        G === nothing || copyto!(G, g)
        return F === nothing ? g : J
    end

    result = Optim.optimize(Optim.NLSolversBase.only_fg!(fg!),
        collect(Float64, θ0), solver.algorithm, solver.options)
    θ̂ = collect(Float64, Optim.minimizer(result))
    V = evaluate_potential(problem, θ̂)
    return InverseResult(θ̂, V, Optim.minimum(result),
        Optim.iterations(result), Optim.converged(result))
end
