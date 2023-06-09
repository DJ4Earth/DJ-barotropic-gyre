# Contains two functions that ultimately do the same thing: they advance the model one step forward in time 
# using an RK4 timestep. The difference between the two comes from the function call itself, one takes 
# my structure RHS_terms, which just allocated space to terms on the RHS of the system. Say, anything that 
# appears in the computation of the time derivatives. The second version was edited to use with Checkpointing.jl, 
# I needed all of the terms that matter to the derivative to appear in a *single* structure, so keeping 
# RHS_terms separate from the states was no longer a good idea. 

function advance(states_rhs::SWM_pde, 
    grid::Grid, 
    params::Params, 
    interp::Interps, 
    grad::Derivatives, 
    advec::Advection
) 

nx = grid.nx 
(;dt, rk_a, rk_b) = params

states_rhs.umid .= states_rhs.u
states_rhs.vmid .= states_rhs.v
states_rhs.etamid .= states_rhs.eta

states_rhs.u0 .= states_rhs.u
states_rhs.v0 .= states_rhs.v
states_rhs.eta0 .= states_rhs.eta

states_rhs.u1 .= states_rhs.u
states_rhs.v1 .= states_rhs.v
states_rhs.eta1 .= states_rhs.eta

@inbounds for j in 1:4

    comp_u_v_eta_t(nx, states_rhs, params, interp, grad, advec)

    if j < 4
        states_rhs.u1 .= states_rhs.umid .+ rk_b[j] .* dt .* states_rhs.u_t
        states_rhs.v1 .= states_rhs.vmid .+ rk_b[j] .* dt .* states_rhs.v_t
        states_rhs.eta1 .= states_rhs.etamid .+ rk_b[j] .* dt .* states_rhs.eta_t
    end

    states_rhs.u0 .= states_rhs.u0 .+ rk_a[j] .* dt .* states_rhs.u_t
    states_rhs.v0 .= states_rhs.v0 .+ rk_a[j] .* dt .* states_rhs.v_t 
    states_rhs.eta0 .= states_rhs.eta0 .+ rk_a[j] .* dt .* states_rhs.eta_t 

end

# Diffusion and bottom friction as Euler forward
dissipative_terms!(nx, states_rhs, params, interp, grad, advec)
states_rhs.u0 .+= dt .* states_rhs.u_t
states_rhs.v0 .+= dt .* states_rhs.v_t

@assert all(x -> x < 7.0, states_rhs.u0)
@assert all(x -> x < 7.0, states_rhs.v0)
@assert all(x -> x < 7.0, states_rhs.eta0)

copyto!(states_rhs.u, states_rhs.u0)
copyto!(states_rhs.v, states_rhs.v0)
copyto!(states_rhs.eta, states_rhs.eta0)

return nothing 

end 

# This function needs to be given 
#           T - how many days to integrate the model for
#           nx, ny - how many grid cells in x and y directions, respectively
#           Lx - E-W length of the domain [meters], has a default but can change if wanted
#           Ly - N-S length of the domain [meters], also has a default 
# mostly keeping because I like having a function that just integrates for some amount
# of time 
function integrate(days, nx, ny; Lx = 3840e3, Ly = 3840e3)                 

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
    
    for t in 1:states_rhs.T

        advance(states_rhs, grid, params, interp, grad, advec)

        copyto!(states_rhs.u, states_rhs.u0)
        copyto!(states_rhs.v, states_rhs.v0)
        copyto!(states_rhs.eta, states_rhs.eta0)
        
    end
        
    return states_rhs

end

function integrate(u0, v0, eta0, days, nx, ny; Lx = 3840e3, Ly = 3840e3)                 

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
        T = T,
        u = u0,
        v = v0,
        eta = eta0
    )
    
    @btime for t in 1:T

        advance(states_rhs, grid, params, interp, grad, advec)
        copyto!(states_rhs.u, states_rhs.u0)
        copyto!(states_rhs.v, states_rhs.v0)
        copyto!(states_rhs.eta, states_rhs.eta0)
        
    end 
        
    return states_rhs

end

