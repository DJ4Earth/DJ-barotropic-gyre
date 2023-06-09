# using Plots, SparseArrays, Parameters, UnPack
# using JLD2, LinearAlgebra
# using Enzyme, Checkpointing, Zygote 

# include("init_structs.jl")
# include("init_params.jl")
# include("build_grid.jl")
# include("build_discrete_operators.jl")
# include("advance.jl")
# include("compute_time_deriv.jl")
# include("temp.jl")

# Enzyme.API.runtimeActivity!(false)

# This script will run an example of Enzyme + Checkpointing being used to compute a sensitivity 
# of the final energy to initial conditions. Five functions are defined. The first two set up the entire experiment by 
# building structures related to the grid, discrete operators, and the structure that contains all variables relevant to the derivatives. 
# The third function defined is the loop that will be checkpointed. It runs a full integration as well as 
# a final computation of the energy (the cost function evaluation). The last two functions puts the setup and checkpointing
# together for the whole experiment. 
#
# Example usage:
# (1) if there are no initial conditions to specify (starting the model from rest)
# include("ExplicitSolver.jl")
# days_to_integrate = 30 
# nx = 128
# ny = 128
# snaps = 80
# denergy = run_checkpointing_energyex(days_to_integrate, nx, ny, snaps)
# 
# (2) if there are non-zero initial condtions (starting from a spun-up state)
# include("ExplicitSolver.jl")
# @load "./initcond_plus_data/states_nx128_ny128_10year_060523.jld2" states_nx128_ny128_10year_060523
# u0 = states_nx128_ny128_10yr_060523.u 
# v0 = states_nx128_ny128_10yr_060523.v
# eta0 = states_nx128_ny128_10yr_060523.eta
# snaps = 80
# days_to_integrate = 30
# nx = 128 
# ny = 128
# denergy = run_checkpointing_energyex(u0, v0, eta0, days_to_integrate, nx, ny, snaps)

function setup_energy(days, 
    nx, 
    ny; 
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

    states_rhs = SWM_pde(Nu = Nu, 
        Nv = Nv,
        NT = NT, 
        Nq = Nq, 
        T = T
    )

    return grid, params, grad, interp, advec, states_rhs

end

function setup_energy(
    u0::Vector{Float64}, 
    v0::Vector{Float64}, 
    eta0::Vector{Float64},
    days,
    nx, 
    ny; 
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


function run_checkpointing_energyex(days, nx, ny, snaps)

    grid, gyre_params, grad_ops, interp_ops, advec_ops, chkpt_struct_outer = setup_energy(days, 
    nx, 
    ny
    )

    snaps = snaps
    verbose = 0
    revolve = Revolve{SWM_pde}(chkpt_struct_outer.T, snaps; verbose=1, gc=true, write_checkpoints=false)

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

function run_checkpointing_energyex(u0, v0, eta0, days, nx, ny, snaps)

    grid, gyre_params, grad_ops, interp_ops, advec_ops, chkpt_struct_outer = setup_energy(u0,
    v0, 
    eta0,
    days, 
    nx, 
    ny
    )

    snaps = snaps
    revolve = Revolve{SWM_pde}(chkpt_struct_outer.T, snaps; verbose=1, gc=true, write_checkpoints=false)

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


# gradient check with the results from checkpointing - passed

# nx = 128
# ny = 128
# days_to_integrate = 90
# snaps = 1
# @time denergy = run_checkpointing_energyex(days_to_integrate, nx, ny, snaps)

# @load "./initcond_plus_data/states_nx128_ny128_10year_060523.jld2" states_nx128_ny128_10year_060523
# u0 = states_nx128_ny128_10year_060523.u
# v0 = states_nx128_ny128_10year_060523.v
# eta0 = states_nx128_ny128_10year_060523.eta

# @time denergy = run_checkpointing_energyex(u0, v0, eta0, days_to_integrate, nx, ny, snaps)

# du = denergy[1].u
# dv = denergy[1].v
# deta = denergy[1].eta

# steps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# use_to_check = du[88]

# T = days_to_seconds(days_to_integrate, gyre_params.dt)

# grid_, gyre_params, grad_ops, interp_ops, advec_ops, chkpt_struct_outer = setup_energy(
#     days_to_integrate, 
#     nx, 
#     ny
# )

# chkpt_struct_new = SWM_pde(Nu = grid_.Nu, 
#     Nv = grid_.Nv,
#     NT = grid_.NT,
#     Nq = grid_.Nq,
#     T = T
# )

# for t = 1:T
#     advance(chkpt_struct_new, grid_, gyre_params, interp_ops, grad_ops, advec_ops)
#     copyto!(chkpt_struct_new.u, chkpt_struct_new.u0)
#     copyto!(chkpt_struct_new.v, chkpt_struct_new.v0)
#     copyto!(chkpt_struct_new.eta, chkpt_struct_new.eta0)
# end
# energy_to_check = energy(grid_, chkpt_struct_new.u0, chkpt_struct_new.v0)

# diffs = []
# for s in steps 

#     T = days_to_seconds(days_to_integrate, gyre_params.dt)

#     chkpt_struct_new = SWM_pde(Nu = grid_.Nu, 
#         Nv = grid_.Nv,
#         NT = grid_.NT,
#         Nq = grid_.Nq,
#         T = T
#     )

#     chkpt_struct_new.u[88] = s

#     for t = 1:T
#         advance(chkpt_struct_new, grid_, gyre_params, interp_ops, grad_ops, advec_ops)
#         copyto!(chkpt_struct_new.u, chkpt_struct_new.u0)
#         copyto!(chkpt_struct_new.v, chkpt_struct_new.v0)
#         copyto!(chkpt_struct_new.eta, chkpt_struct_new.eta0)
#     end

#     new_energy = energy(grid_, chkpt_struct_new.u0, chkpt_struct_new.v0)

#     push!(diffs, (new_energy - energy_to_check) / s)

# end
