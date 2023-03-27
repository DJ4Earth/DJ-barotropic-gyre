using Enzyme, Parameters, SparseArrays

include("init_structs.jl")
include("init_params.jl")
include("build_grid.jl")
include("build_discrete_operators.jl")
include("compute_time_deriv.jl")

function advance(u_v_eta, grid, rhs, params, interp, grad, advec) 

    nx = grid.nx 
    dt = params.dt

    # we now use RK4 as the timestepper, here I'm storing the coefficients needed for this 
    rk_a = [1/6, 1/3, 1/3, 1/6]
    rk_b = [1/2, 1/2, 1.]

    rhs.umid .= u_v_eta.u
    rhs.vmid .= u_v_eta.v
    rhs.etamid .= u_v_eta.eta

    rhs.u0 .= u_v_eta.u
    rhs.v0 .= u_v_eta.v
    rhs.eta0 .= u_v_eta.eta

    rhs.u1 .= u_v_eta.u
    rhs.v1 .= u_v_eta.v
    rhs.eta1 .= u_v_eta.eta

    for j in 1:4

        comp_u_v_eta_t(nx, rhs, params, interp, grad, advec)

        if j < 4
            rhs.u1 .= rhs.umid .+ rk_b[j] .* dt .* rhs.u_t
            rhs.v1 .= rhs.vmid .+ rk_b[j] .* dt .* rhs.v_t
            rhs.eta1 .= rhs.etamid .+ rk_b[j] .* dt .* rhs.eta_t
        end

        rhs.u0 .= rhs.u0 .+ rk_a[j] .* dt .* rhs.u_t
        rhs.v0 .= rhs.v0 .+ rk_a[j] .* dt .* rhs.v_t 
        rhs.eta0 .= rhs.eta0 .+ rk_a[j] .* dt .* rhs.eta_t 

    end

    @assert all(x -> x < 10.0, rhs.u0)
    @assert all(x -> x < 10.0, rhs.v0)
    @assert all(x -> x < 10.0, rhs.eta0)

    copyto!(u_v_eta.u, rhs.u0)
    copyto!(u_v_eta.v, rhs.v0)
    copyto!(u_v_eta.eta, rhs.eta0)

    return nothing 

end

function main(Trun, u_v_eta, grid, rhs, params, interp, grad, advec)

    for t in 1:Trun
        advance(u_v_eta, grid, rhs, params, interp, grad, advec) 
    end
    
    return nothing  
    
end

nx = 5             # grid resolution in x-direction
ny = 5             # grid resolution in y-direction

Lx = 3840e3                     # E-W length of the domain [meters]
Ly = 3840e3                     # N-S length of the domain [meters]

grid_params = build_grid(Lx, Ly, nx, ny)
gyre_params = def_params(grid_params)

# building discrete operators 
grad_ops = build_derivs(grid_params)                # discrete gradient operators 
interp_ops = build_interp(grid_params, grad_ops)    # discrete interpolation operators (travels between grids)
advec_ops = build_advec(grid_params)
rhs_terms = RHS_terms(Nu = grid_params.Nu, Nv = grid_params.Nv, NT = grid_params.NT, Nq = grid_params.Nq)

u_v_eta = gyre_vector(zeros(grid_params.Nu), zeros(grid_params.Nv), zeros(grid_params.NT))
ad_u_v_eta = gyre_vector(zeros(grid_params.Nu), zeros(grid_params.Nv), zeros(grid_params.NT))

ad_grad_ops = deepcopy(grad_ops)
ad_interp_ops = deepcopy(interp_ops)
ad_advec_ops = deepcopy(advec_ops)

# checking that the function does in fact run
main(1, u_v_eta, grid_params, rhs_terms, gyre_params, interp_ops, grad_ops, advec_ops)

autodiff(main,
    Const(1),
    Duplicated(u_v_eta, ad_u_v_eta),
    grid_params,
    rhs_terms,
    gyre_params,
    interp_ops,
    grad_ops,
    advec_ops
)