# This is my attempt at a Julia equivalent of Milan Kloewer's shallow water model (code found here: https://github.com/milankl/swm)
# We'll solve the following equations on an Arakawa C-grid setup:
#       u_t = f v - g eta_x + A_h (u_xx + u_yy) + F_x 
#       v_t = -f u - g eta_y + A_h (v_xx + v_yy)
#       eta_t = - (H u_x + H v_y)
# using a fully explicit solver. 
# Will add more about how the code is structured at a future point.... 

using Enzyme, Plots, SparseArrays, Parameters

include("init_structs.jl")
include("init_params.jl")
include("build_grid.jl")
include("build_discrete_operators.jl")
include("advance_c_grid.jl")
include("compute_time_deriv.jl")
include("temp.jl")

# setting up the grid parameters first


# Zanna / Bolton setup 
Lx = 3840e3                    # E-W length of the domain [meters]
Ly = 3840e3                    # N-S length of the domain [meters]
nx = 20                         # number of cells in the x-direction
ny = 20                        # number of cells in the y-direction

###### debugging 
dx = Lx / nx 
dy = Ly / ny 

NT = nx * ny 

Nx = nx 
Ny = ny 

Nu = (Nx - 1) * Ny 
Nv = (Ny - 1) * Nx 
Nq = (Nx + 1) * (Ny + 1)

x = (dx/2):dx:Lx
y = dy/2:dy:Ly

xu = x[1:end-1] .+ dx/2 
yu = copy(y)

xv = copy(x) 
yv = y[1:end-1] .+ dy/2 

xq = 0:dx:(Lx + dx/2)
yq = 0:dy:(Ly + dy/2)
########

# MIT GCM setup 
# Lx = 1200e3 
# Ly = 1200e3
# nx = 62 
# ny = 62 

# based on above values this returns more parameters related to the four grids 
grid_params = build_grid(Lx, Ly, nx, ny)

gyre_params = def_params(grid_params)

# building discrete operators 
grad_ops = build_derivs(grid_params)                # discrete gradient operators 
interp_ops = build_interp(grid_params, grad_ops)    # discrete interpolation operators (travels between grids)
advec_ops = build_advec(grid_params)
# rhs_terms = allocate(grid_params, grad_ops, interp_ops, advec_ops)

# starting from rest ---> all initial conditions are zero 

# how long to spinup the model for 
Tspinup_days = 500 # [days] 

# how long to run the model for after spinup
Trun_days = 10     # [days] 

Tspinup, Trun = days_to_seconds(Tspinup_days, Trun_days, gyre_params.dt)

uout = zeros(grid_params.Nu) 
vout = zeros(grid_params.Nv) 
etaout = zeros(grid_params.NT)

# uout = (collect(LinRange(0, grid_params.Nu - 1, grid_params.Nu)) ./ 1000).^2
# vout = (collect(LinRange(0, grid_params.Nv - 1, grid_params.Nv)) ./ 1000).^2
# etaout = (collect(LinRange(0, grid_params.NT - 1, grid_params.NT)) ./ 1000).^2

u_v_eta = gyre_vector(copy(uout), 
copy(vout), 
copy(etaout))

@time for t in 1:Tspinup
    advance(u_v_eta, gyre_params, interp_ops, grad_ops, advec_ops) 
end

u_v_eta_mat = vec_to_mat(u_v_eta.u, u_v_eta.v, u_v_eta.eta, grid_params)
heatmap(u_v_eta_mat.eta)

# u_v_eta_start = deepcopy(u_v_eta)

# @time for t in 1:Tspinup
#     advance(u_v_eta, gyre_params, interp_ops, grad_ops, advec_ops) 
# end
