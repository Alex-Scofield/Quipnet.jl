module Quipnet

using FFTW
using LinearAlgebra
using Optim
using Plots
using Random
using Statistics

include("solvers.jl")
include("forward.jl")
include("bases.jl")
include("maps.jl")
include("regularizers.jl")
include("noise.jl")
include("problem.jl")
include("adjoint.jl")
include("inverse_solvers.jl")
include("metrics.jl")
include("visualize.jl")

export free_schrod_periodic
export schrod_periodic
export animate_free_particle_periodic

export AbstractForwardSolver, FreeSchrodinger, SplitStepSchrodinger, propagate
export FourierBasis, GridBasis, dim, evaluate_basis, basis_matrix
export FourierInput, BatchInput, PointDensity
export apply_input, apply_measurement
export GaussianNoise, InverseProblem, generate_data, evaluate_potential
export L2Regularizer, TikhonovSolver, InverseResult, fit!, reconstruct
export loss_and_grad, relative_error

end
