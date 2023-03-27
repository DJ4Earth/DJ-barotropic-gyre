# This is my attempt at a Julia equivalent of Milan Kloewer's shallow water model (code found here: https://github.com/milankl/swm)
# We'll solve the following equations on an Arakawa C-grid setup:
#       u_t = f v - g eta_x + A_h (u_xx + u_yy) + F_x 
#       v_t = -f u - g eta_y + A_h (v_xx + v_yy)
#       eta_t = - (H u_x + H v_y)
# using a fully explicit solver. 
# Will add more about how the code is structured at a future point.... 

# This script is almost identical to main_barotropic_gyre, except here I'm attempting to add in 
# the Julia package Checkpointing, as I was running into memory issues applying Enzyme

using Plots, SparseArrays, Parameters
using JLD2
using Enzyme, Checkpointing, Zygote

include("init_structs.jl")
include("init_params.jl")
include("build_grid.jl")
include("build_discrete_operators.jl")
include("compute_time_deriv.jl")
include("advance.jl")

function setup(; nx = 10, ny = 10, Lx = 3840e3, Ly = 3840e3)

    grid = build_grid(Lx, Ly, nx, ny)
    params = def_params(grid)

    # building discrete operators 
    grad = build_derivs(grid)                # discrete gradient operators 
    interp = build_interp(grid, grad)    # discrete interpolation operators (travels between grids)
    advec = build_advec(grid)
    #rhs = RHS_terms(Nu = grid.Nu, Nv = grid.Nv, NT = grid.NT, Nq = grid.Nq)

    return grid, params, grad, interp, advec

end

function chkpt_maybe(chkpt_struct::SWM_pde, T::Int)
    grid, gyre_params, grad_ops, interp_ops, advec_ops = setup(nx=20, ny=20)
    @checkpoint_struct revolve chkpt_struct for j in 1:T

        advance2(chkpt_struct, grid, gyre_params, interp_ops, grad_ops, advec_ops)

        copyto!(chkpt_struct.u, chkpt_struct.u0)
        copyto!(chkpt_struct.v, chkpt_struct.v0)
        copyto!(chkpt_struct.eta, chkpt_struct.eta0)

    end
    return chkpt_struct.u
end

nx = 20
ny = 20 

deriv_struct = SWM_pde(Nu = (nx - 1) * ny, 
    Nv = (ny - 1) * nx, 
    NT = nx * ny, 
    Nq = (nx + 1) * (ny + 1)
)

Trun = 10
snaps = 3
verbose = 0
revolve = Revolve{SWM_pde}(Trun, snaps; verbose=verbose)

g = Zygote.jacobian(chkpt_maybe, deriv_struct, Trun)

# grid, gyre_params, grad_ops, interp_ops, advec_ops = setup(nx=20,ny=20)
# for j in 1:Trun
#     advance2(deriv_struct, grid, gyre_params, interp_ops, grad_ops, advec_ops)
# end

# state_m = vec_to_mat(deriv_struct.u, deriv_struct.v, deriv_struct.eta, grid)
