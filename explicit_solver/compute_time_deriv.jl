function comp_u_v_eta_t(u, v, eta, params, interp_ops, grad_ops, advec_ops) 

    h = eta .+ params.H 

    h_u = interp_ops.ITu * h
    h_v = interp_ops.ITv * h
    h_q = interp_ops.ITq * h

    U = u .* h_u 
    V = v .* h_v 

    kinetic = interp_ops.IuT * (u.^2) + interp_ops.IvT * (v.^2)

    # Kloewer defined new terms q and p corresponding to potential vorticity and 
    # Bernoulli potential respectively. To avoid errors in my mimic I'm following 
    # along and doing the same 
    q = (params.coriolis + grad_ops.Gvx * v - grad_ops.Guy * u) ./ h_q 
    p = 0.5 * kinetic + params.g * h

    # bottom friction
    kinetic_sq = (kinetic).^(1/2)
    bfric_u =  params.bottom_drag * ((interp_ops.ITu * kinetic_sq) .* u) ./ h_u
    bfric_v =  params.bottom_drag * ((interp_ops.ITv * kinetic_sq) .* v) ./ h_v

    # deal with the advection term 
    adv_u, adv_v = comp_advection(q, U, V, advec_ops)

    Mu =  params.A_h .* (grad_ops.LLu * u)
    Mv =  params.A_h .* (grad_ops.LLv * v) 

    rhs_u = adv_u - grad_ops.GTx * p + params.wind_stress ./ h_u - Mu - bfric_u

    rhs_v = adv_v - grad_ops.GTy * p - Mv - bfric_v 

    rhs_eta = - (grad_ops.Gux * U + grad_ops.Gvy * V) 

    return rhs_u, rhs_v, rhs_eta

end 

function comp_advection(q, U, V, advec_ops)

    AL1q = advec_ops.AL1 * q 
    AL2q = advec_ops.AL2 * q 

    # adv_u = advec_ops.Seur * ((advec_ops.ALeur * q) .* U) + advec_ops.Seul * ((advec_ops.ALeul * q) .* U) +
    # advec_ops.Sau * (AL1q[1:end-nx] .* V) + advec_ops.Sbu * (AL2q[nx+1:end] .* V) + 
    # advec_ops.Scu * (AL2q[1:end-nx] .* V) + advec_ops.Sdu * (AL1q[nx+1:end] .* V)
    
    # adv_v = advec_ops.Spvu * ((advec_ops.ALpvu * q) .* U) + advec_ops.Spvd * ((advec_ops.ALpvd * q) .* V) -
    # advec_ops.Sav * (AL1q[advec_ops.index_av] .* U) - advec_ops.Sbv * (AL2q[advec_ops.index_bv] .* U) - 
    # advec_ops.Scv * (AL2q[advec_ops.index_cv] .* U) - advec_ops.Sdv * (AL1q[advec_ops.index_dv] .* U)

    adv_u = advec_ops.Seur * ((advec_ops.ALeur * q) .* U) + advec_ops.Seul * ((advec_ops.ALeul * q) .* U) +
    advec_ops.Sau * (AL1q[1:end-nx] .* V) + advec_ops.Sbu * (AL2q[nx+1:end] .* V) + 
    advec_ops.Scu * (AL2q[1:end-nx] .* V) + advec_ops.Sdu * (AL1q[nx+1:end] .* V)
    
    adv_v = advec_ops.Spvu * ((advec_ops.ALpvu * q) .* V) + advec_ops.Spvd * ((advec_ops.ALpvd * q) .* V) -
    advec_ops.Sav * (AL1q[advec_ops.index_av] .* U) - advec_ops.Sbv * (AL2q[advec_ops.index_bv] .* U) - 
    advec_ops.Scv * (AL2q[advec_ops.index_cv] .* U) - advec_ops.Sdv * (AL1q[advec_ops.index_dv] .* U)

    return adv_u, adv_v

end