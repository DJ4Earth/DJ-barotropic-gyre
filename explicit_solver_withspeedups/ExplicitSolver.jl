module ExplicitSolver

using Plots, SparseArrays, Parameters, UnPack
using JLD2, LinearAlgebra
using Enzyme, Checkpointing, Zygote

include("init_structs.jl")
include("init_params.jl")
include("build_grid.jl")
include("build_discrete_operators.jl")
include("advance.jl")
include("compute_time_deriv.jl")
include("main_energy_1_chkp.jl")

Enzyme.API.runtimeActivity!(true)

end