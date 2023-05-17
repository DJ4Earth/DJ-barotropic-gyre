# making an effort at the whole experiment here, done with Enzyme + checkpointing 
# once this is running and working will structure a main file with it. 

# This one will try to use Enzyme + checkpointing for an energy sensitivity
# similar to the Burgers equation

using Plots, SparseArrays, Parameters, UnPack
using JLD2, LinearAlgebra
using Enzyme, Checkpointing, Zygote

include("init_structs.jl")
include("init_params.jl")
include("build_grid.jl")
include("build_discrete_operators.jl")
include("advance.jl")
include("cost_func.jl")
include("compute_time_deriv.jl")

# This function will setup the structures needed to integrate the model. Comes with default values, but
# these can be specified if desired. 
function setup(; days = 30, nx = 20, ny = 20, Lx = 3840e3, Ly = 3840e3)

    grid = build_grid(Lx, Ly, nx, ny)
    params = def_params(grid)

    # building discrete operators
    grad = build_derivs(grid)            # discrete gradient operators
    interp = build_interp(grid, grad)    # discrete interpolation operators (travels between grids)
    advec = build_advec(grid)

    Nu = grid.Nu
    Nv = grid.Nv
    NT = grid.NT
    Nq = grid.Nq 

    T = days_to_seconds(days, params.dt)

    states_rhs = SWM_pde(Nu = Nu, 
        Nv = Nv,
        NT = NT, 
        Nq = Nq, 
        T = T
    )

    return grid, params, grad, interp, advec, states_rhs

end


function chkpt_maybe(
    chkpt_struct::SWM_pde,
    chkpt_scheme::Scheme,
    grid::Grid,
    params::Params,
    interp::Interps,
    grad::Derivatives,
    advec::Advection
)

    @checkpoint_struct chkpt_scheme chkpt_struct for t in 1:chkpt_struct.T

        advance(chkpt_struct, grid, params, interp, grad, advec)

        copyto!(chkpt_struct.u, chkpt_struct.u0)
        copyto!(chkpt_struct.v, chkpt_struct.v0)
        copyto!(chkpt_struct.eta, chkpt_struct.eta0)

    end

    energy = sum(chkpt_struct.u.^2 .+ chkpt_struct.v.^2) / (grid.nx * grid.ny)

    return energy

end

days = 10 
nx = 50
ny = 50

grid, gyre_params, grad_ops, interp_ops, advec_ops, chkpt_struct_outer = setup(days=days, nx=nx, ny=ny)

# snaps = 300
# verbose = 0
# revolve = Revolve{SWM_pde}(chkpt_struct_outer.T, snaps; verbose=verbose)

# denergy = Zygote.gradient(chkpt_maybe, 
#     chkpt_struct_outer, 
#     revolve, 
#     grid, 
#     gyre_params, 
#     interp_ops, 
#     grad_ops, 
#     advec_ops
# )

#### from here these are derivative checks ####

# gradient check with the results from checkpointing (only run for one single step) - passed

# du = denergy[1].u
# dv = denergy[1].v
# deta = denergy[1].eta

# steps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# use_to_check = du[88]

# chkpt_struct_new = SWM_pde(Nu = grid.Nu, 
# Nv = grid.Nv,
# NT = grid.NT, 
# Nq = grid.Nq, 
# T = 1
# )

# advance(chkpt_struct_new, grid, gyre_params, interp_ops, grad_ops, advec_ops)
# energy_to_check = energy(grid, chkpt_struct_new.u0, chkpt_struct_new.v0)

# diffs = []
# for s in steps 

#     chkpt_struct_new = SWM_pde(Nu = grid.Nu, 
#         Nv = grid.Nv,
#         NT = grid.NT, 
#         Nq = grid.Nq, 
#         T = 1
#     )

#     chkpt_struct_new.u[88] = s

#     advance(chkpt_struct_new, grid, gyre_params, interp_ops, grad_ops, advec_ops)

#     new_energy = energy(grid, chkpt_struct_new.u0, chkpt_struct_new.v0)

#     push!(diffs, (new_energy - energy_to_check) / s)

# end

# first check to see if the adjoints diverge without checkpointing - passed, seem to be fine 

# integrate forward so we have the final states 
# for t = 1:chkpt_struct_outer.T

#     advance(chkpt_struct_outer, grid, gyre_params, interp_ops, grad_ops, advec_ops)
#     copyto!(chkpt_struct_outer.u, chkpt_struct_outer.u0)
#     copyto!(chkpt_struct_outer.v, chkpt_struct_outer.v0)
#     copyto!(chkpt_struct_outer.eta, chkpt_struct_outer.eta0)

# end

# function for_enzyme(
#     chkpt_struct::SWM_pde,
#     grid::Grid,
#     params::Params,
#     interp::Interps,
#     grad::Derivatives,
#     advec::Advection
# )

#     for t = 1:chkpt_struct.T

#         advance(chkpt_struct, grid, params, interp, grad, advec)
#         copyto!(chkpt_struct.u, chkpt_struct.u0)
#         copyto!(chkpt_struct.v, chkpt_struct.v0)
#         copyto!(chkpt_struct.eta, chkpt_struct.eta0)

#     end

# end
 
# T = days_to_seconds(days, gyre_params.dt)

# ad_chkpt_struct = SWM_pde(Nu = grid.Nu, 
# Nv = grid.Nv,
# NT = grid.NT, 
# Nq = grid.Nq, 
# T = T
# )

# ad_chkpt_struct.eta[13] = 1.0

# state_for_checking = copy(chkpt_struct_outer.eta[13])

# autodiff(Reverse, for_enzyme, 
#     Duplicated(chkpt_struct_outer, ad_chkpt_struct),
#     grid,
#     gyre_params, 
#     interp_ops,
#     grad_ops, 
#     advec_ops
# )

# # check the derivative 
# use_to_check = ad_chkpt_struct.u[33]

# steps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# diffs = []
# new_eta = []
# for s in steps 

#     chkpt_struct_new1 = SWM_pde(Nu = grid.Nu, 
#         Nv = grid.Nv,
#         NT = grid.NT, 
#         Nq = grid.Nq, 
#         T = 50
#     )

#     chkpt_struct_new1.u[33] = s
    
#     for t = 1:50
#         advance(chkpt_struct_new1, grid, gyre_params, interp_ops, grad_ops, advec_ops)
#         copyto!(chkpt_struct_new1.u, chkpt_struct_new1.u0)
#         copyto!(chkpt_struct_new1.v, chkpt_struct_new1.v0)
#         copyto!(chkpt_struct_new1.eta, chkpt_struct_new1.eta0)
#     end
#     push!(new_eta, chkpt_struct_new1.eta[13])

#     push!(diffs, (chkpt_struct_new1.eta[13] - state_for_checking) / s)

# end

# chkpt_maybe(
#     chkpt_struct, 
#     revolve, 
#     grid, 
#     gyre_params, 
#     interp_ops, 
#     grad_ops, 
#     advec_ops
# )

# for checking that the forward integration still works 

# for t = 1:chkpt_struct.T
#     advance(chkpt_struct, grid, gyre_params, interp_ops, grad_ops, advec_ops)
#     copyto!(chkpt_struct.u, chkpt_struct.u0)
#     copyto!(chkpt_struct.v, chkpt_struct.v0)
#     copyto!(chkpt_struct.eta, chkpt_struct.eta0)
# end