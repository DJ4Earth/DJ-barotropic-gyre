# Contains one function advance, which just takes a single step forward in time. 
# This will differ from advance.jl: here I'm going to try and implement a C-grid scheme 

function advance(u_v_eta, params, interp_ops, grad_ops, advec_ops) 

    dt = params.dt

    # we now use RK4 as the timestepper, here I'm storing the coefficients needed for this 
    rk_a = [1/6, 1/3, 1/3, 1/6]
    rk_b = [1/2, 1/2, 1]

    umid = copy(u_v_eta.u)
    vmid = copy(u_v_eta.v)
    etamid = copy(u_v_eta.eta)

    u0 = copy(u_v_eta.u)
    v0 = copy(u_v_eta.v)
    eta0 = copy(u_v_eta.eta)

    u1 = copy(u_v_eta.u)
    v1 = copy(u_v_eta.v)
    eta1 = copy(u_v_eta.eta)

    for j in 1:4

        u_t, v_t, eta_t = comp_u_v_eta_t(u1, v1, eta1, params, interp_ops, grad_ops, advec_ops)

        if j < 4
            u1 .= umid + rk_b[j] * dt * u_t
            v1 .= vmid + rk_b[j] * dt * v_t
            eta1 .= etamid + rk_b[j] * dt * eta_t
        end

        u0 .= u0 + rk_a[j] * dt * u_t
        v0 .= v0 + rk_a[j] * dt * v_t 
        eta0 .= eta0 + rk_a[j] * dt * eta_t 

    end

    # @assert all(x -> x < 5.0, u0)
    # @assert all(x -> x < 5.0, v0)
    # @assert all(x -> x < 5.0, eta0)

    copyto!(u_v_eta.u, u0)
    copyto!(u_v_eta.v, v0)
    copyto!(u_v_eta.eta, eta0)

    return nothing 

end 