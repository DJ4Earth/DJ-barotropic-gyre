# Contains two functions that ultimately do the same thing: they advance the model one step forward in time 
# using an RK4 timestep. The difference between the two comes from the function call itself, one takes 
# my structure RHS_terms, which just allocated space to terms on the RHS of the system. Say, anything that 
# appears in the computation of the time derivatives. The second version was edited to use with Checkpointing.jl, 
# I needed all of the terms that matter to the derivative to appear in a *single* structure, so keeping 
# RHS_terms separate from the states was no longer a good idea. 


function advance(u_v_eta::gyre_vector, 
        grid::Grid, 
        rhs::RHS_terms, 
        params::Params, 
        interp::Interps, 
        grad::Derivatives, 
        advec::Advection
    ) 

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

    @assert all(x -> x < 7.0, rhs.u0)
    @assert all(x -> x < 7.0, rhs.v0)
    @assert all(x -> x < 7.0, rhs.eta0)

    copyto!(u_v_eta.u, rhs.u0)
    copyto!(u_v_eta.v, rhs.v0)
    copyto!(u_v_eta.eta, rhs.eta0)

    return nothing 

end 

function advance(states_rhs::SWM_pde, 
        grid::Grid, 
        params::Params, 
        interp::Interps, 
        grad::Derivatives, 
        advec::Advection
    ) 

    nx = grid.nx 
    dt = params.dt

    # we now use RK4 as the timestepper, here I'm storing the coefficients needed for this 
    rk_a = [1/6, 1/3, 1/3, 1/6]
    rk_b = [1/2, 1/2, 1.]

    states_rhs.umid .= states_rhs.u
    states_rhs.vmid .= states_rhs.v
    states_rhs.etamid .= states_rhs.eta

    states_rhs.u0 .= states_rhs.u
    states_rhs.v0 .= states_rhs.v
    states_rhs.eta0 .= states_rhs.eta

    states_rhs.u1 .= states_rhs.u
    states_rhs.v1 .= states_rhs.v
    states_rhs.eta1 .= states_rhs.eta

    for j in 1:4

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

    @assert all(x -> x < 7.0, states_rhs.u0)
    @assert all(x -> x < 7.0, states_rhs.v0)
    @assert all(x -> x < 7.0, states_rhs.eta0)

    return nothing 

end 

# This function needs to be given 
#           T - how many days to integrate the model for
#           nx, ny - how many grid cells in x and y directions, respectively
#           Lx - E-W length of the domain [meters], has a default but can change if wanted
#           Ly - N-S length of the domain [meters], also has a default 
# mostly keeping because I like having a function that just integrates for some amount
# of time 
function integrate(T, nx, ny; Lx = 3840e3, Ly = 3840e3)                 
    
    grid_params = build_grid(Lx, Ly, nx, ny)
    gyre_params = def_params(grid_params)
    
    # building discrete operators 
    grad_ops = build_derivs(grid_params)                # discrete gradient operators 
    interp_ops = build_interp(grid_params, grad_ops)    # discrete interpolation operators (travels between grids)
    advec_ops = build_advec(grid_params)
    rhs_terms = RHS_terms(Nu = grid_params.Nu, Nv = grid_params.Nv, NT = grid_params.NT, Nq = grid_params.Nq)
    
    # starting from rest ---> all initial conditions are zero 
    
    Trun = days_to_seconds(T, gyre_params.dt)
    
    uout = zeros(grid_params.Nu) 
    vout = zeros(grid_params.Nv) 
    etaout = zeros(grid_params.NT)
    
    u = [zeros(grid_params.Nu)]
    v = [zeros(grid_params.Nv)]
    eta = [zeros(grid_params.NT)]

    u_v_eta = gyre_vector(uout, vout, etaout)
    
    @time for t in 1:Trun
        advance(u_v_eta, grid_params, rhs_terms, gyre_params, interp_ops, grad_ops, advec_ops)
        push!(u, copy(u_v_eta.u))
        push!(v, copy(u_v_eta.v))
        push!(eta, copy(u_v_eta.eta))
    end
    
    # u_v_eta_mat = vec_to_mat(u_v_eta.u, u_v_eta.v, u_v_eta.eta, grid_params)
    
    return u, v, eta
    
end

# ****IMPORTANT**** not yet sure if I'm moving between high and low res grids, need to check with Patrick
# and come back here if there are issues with how I did it

# This function needs to be given 
#           Trun - how many days to integrate the model for
#           nx_lowres, ny_lowres - grid resolution (number of cells in the x and y directions
#                    respectively) of the courser grid
#           Lx, Ly - size of the domain, have a default value but can set 
#                    manually if needed 
# Theoretically, we should only ever run this function *once*, from there 
# the data points will be stored as a JLD2 data file 
function create_data(days, nx_lowres, ny_lowres; scaling = 5, Lx = 3840e3, Ly = 3840e3)

    nx_highres = nx_lowres * scaling
    ny_highres = ny_lowres * scaling 

    grid_lowres = build_grid(Lx, Ly, nx_lowres, ny_lowres)
    grid_highres = build_grid(Lx, Ly, nx_highres, ny_highres)
    params = def_params(grid_highres)

    # building discrete operators
    grad = build_derivs(grid_highres)            # discrete gradient operators
    interp = build_interp(grid_highres, grad)    # discrete interpolation operators (travels between grids)
    advec = build_advec(grid_highres)

    Trun = days_to_seconds(days, params.dt)

    Nu = grid_highres.Nu
    Nv = grid_highres.Nv
    NT = grid_highres.NT
    Nq = grid_highres.Nq 

    u_v_eta_rhs = SWM_pde(Nu = Nu, 
        Nv = Nv,
        NT = NT, 
        Nq = Nq
    )

    data = zeros(grid_lowres.Nu + grid_lowres.Nv + grid_lowres.NT, Trun)

    temp_u = zeros(grid_lowres.Nu)
    temp_v = zeros(grid_lowres.Nv)
    temp_eta = zeros(grid_lowres.NT)

    # the steps where we want data in the high res model correspond to (roughly) scaling * t for t 
    # in the low res model. for simplicity I'm going to keep the times in the low res where I want to have 
    # data and then just scale them in the for loop to find corresponding high res data points  

    # for an initial effort I'm just going to assume data at every timestep for simplicity, 
    # and will be running pretty course resolution models for both the high and low res 

    temp_matrices = vec_to_mat(u_v_eta_rhs.u, u_v_eta_rhs.v, u_v_eta_rhs.eta, grid_highres)
    
    temp_u .= reshape(temp_matrices.u[scaling:scaling:end, scaling:scaling:end]', grid_lowres.Nu)
    temp_v .= reshape(temp_matrices.v[scaling:scaling:end, scaling:scaling:end]', grid_lowres.Nv)
    temp_eta .= reshape(temp_matrices.eta[scaling:scaling:end, scaling:scaling:end]', grid_lowres.NT)

    data[:, 1] .= [temp_u; temp_v; temp_eta]

    for t in 2:Trun

        advance(u_v_eta_rhs, grid_highres, params, interp, grad, advec) 

        temp_matrices = vec_to_mat(u_v_eta_rhs.u0, u_v_eta_rhs.v0, u_v_eta_rhs.eta0, grid_highres)
        temp_u .= reshape(temp_matrices.u[scaling:scaling:end, scaling:scaling:end]', grid_lowres.Nu)
        temp_v .= reshape(temp_matrices.v[scaling:scaling:end, scaling:scaling:end]', grid_lowres.Nv)
        temp_eta .= reshape(temp_matrices.eta[scaling:scaling:end, scaling:scaling:end]', grid_lowres.NT)
        data[:, t] .= [temp_u; temp_v; temp_eta]

        copyto!(u_v_eta_rhs.u, u_v_eta_rhs.u0)
        copyto!(u_v_eta_rhs.v, u_v_eta_rhs.v0)
        copyto!(u_v_eta_rhs.eta, u_v_eta_rhs.eta0)

    end
    
    return data
    
end