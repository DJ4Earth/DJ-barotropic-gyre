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
include("temp.jl")

# This function will setup the structures needed to integrate the model. Comes with default values, but
# these can be specified if desired. 
# Now has an option to not start the integration from rest and instead specify the initial fields 
# u0, v0, eta0. If these are specified, the arguments nx and ny should also be adjusted to match the 
# dimensions of the initial conditions. I don't know how to force this to happen, so for now just need
# to remember 
function setup(;days = 1, nx = 30, ny = 30, Lx = 3840e3, Ly = 3840e3)

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

function setup(
    u0::Vector{Float64}, 
    v0::Vector{Float64}, 
    eta0::Vector{Float64},
    nx, 
    ny;
    days = 10, 
    Lx = 3840e3, 
    Ly = 3840e3
)

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

    states_rhs = SWM_pde(
        Nu = Nu, 
        Nv = Nv,
        NT = NT,
        Nq = Nq,
        T = T, 
        u = u0,
        v = v0, 
        eta = eta0
    )

    return grid, params, grad, interp, advec, states_rhs

end

function chkpt_integration(
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


function run_checkpointing(days_to_integrate, nx, ny, snaps)

    # @load "u_v_eta_nx128_ny128_10yr.jld2" u_v_eta_init_cond

    grid, gyre_params, grad_ops, interp_ops, advec_ops, chkpt_struct_outer = setup(nx = nx,
    ny = ny,
    days=days_to_integrate
    )

    snaps = snaps
    verbose = 0
    revolve = Revolve{SWM_pde}(chkpt_struct_outer.T, snaps; verbose=verbose)

    denergy = Zygote.gradient(chkpt_integration, 
        chkpt_struct_outer, 
        revolve, 
        grid, 
        gyre_params, 
        interp_ops, 
        grad_ops, 
        advec_ops
    )

    return denergy

end

days_to_integrate = 10
@time denergy = run_checkpointing(days_to_integrate, 10, 10, 2)

# gradient check with the results from checkpointing - passed

# du = denergy[1].u
# dv = denergy[1].v
# deta = denergy[1].eta

# steps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# use_to_check = du[88]

# T = days_to_seconds(days_to_integrate, gyre_params.dt)

# chkpt_struct_new = SWM_pde(Nu = grid.Nu, 
#     Nv = grid.Nv,
#     NT = grid.NT,
#     Nq = grid.Nq,
#     T = 1
# )

# for t = 1:T
#     advance(chkpt_struct_new, grid, gyre_params, interp_ops, grad_ops, advec_ops)
#     copyto!(chkpt_struct_new.u, chkpt_struct_new.u0)
#     copyto!(chkpt_struct_new.v, chkpt_struct_new.v0)
#     copyto!(chkpt_struct_new.eta, chkpt_struct_new.eta0)
# end
# energy_to_check = energy(grid, chkpt_struct_new.u0, chkpt_struct_new.v0)

# diffs = []
# for s in steps 

#     chkpt_struct_new = SWM_pde(Nu = grid.Nu, 
#         Nv = grid.Nv,
#         NT = grid.NT,
#         Nq = grid.Nq,
#         T = 1
#         #u_v_eta_init_cond.u,
#         #u_v_eta_init_cond.v, 
#         #u_v_eta_init_cond.eta
#     )

#     chkpt_struct_new.u[88] = s

#     for t = 1:T
#         advance(chkpt_struct_new, grid, gyre_params, interp_ops, grad_ops, advec_ops)
#         copyto!(chkpt_struct_new.u, chkpt_struct_new.u0)
#         copyto!(chkpt_struct_new.v, chkpt_struct_new.v0)
#         copyto!(chkpt_struct_new.eta, chkpt_struct_new.eta0)
#     end

#     new_energy = energy(grid, chkpt_struct_new.u0, chkpt_struct_new.v0)

#     push!(diffs, (new_energy - energy_to_check) / s)

# end
