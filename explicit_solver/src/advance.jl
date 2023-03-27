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