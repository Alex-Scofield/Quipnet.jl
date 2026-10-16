module Quipnet

using FFTW
using Plots

include("solvers.jl")
include("visualize.jl")

export free_schrod_periodic
export schrod_periodic
export animate_free_particle_periodic

end
