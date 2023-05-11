# Contains two functions: one that computes the time derivatives and another that 
# computes the advection term (needed for the time derivatives). Two versions of each function
# are defined, each just depends on what type of structure I'm passing for RHS 

function comp_u_v_eta_t(nx::Int, 
        rhs::RHS_terms, 
        params::Params, 
        interp::Interps, 
        grad::Derivatives, 
        advec::Advection
    )

    # unpack stuff
    u = rhs.u1
    v = rhs.v1
    eta = rhs.eta1

    (;ITu, ITv, ITq, IuT, IvT) = interp                 # interpolation operators
    (;GTx, GTy, Gux, Guy, Gvx, Gvy, LLu, LLv) = grad    # gradient operators
    (;h, h_u, h_v, h_q, U, V, p, q) = rhs               # diagnostic variables
    (;kinetic, kinetic_sq, Mu, Mv, nu_u, nu_v) = rhs
    (;GTx_p, GTy_p, Gux_U, Gvy_V, Gvx_v1, Guy_u1) = rhs
    (;adv_u, adv_v, bfric_u, bfric_v) = rhs
    (;IuT_u1, IvT_v1, ITu_ksq,ITv_ksq) = rhs
    (;u_t, v_t, eta_t) = rhs                            # tendencies
    (;H, coriolis, g, nu, bottom_drag, wind_stress) = params

    h .= eta .+ H 

    @inplacemul h_q = ITq * h
    @inplacemul h_u = ITu * h
    @inplacemul h_v = ITv * h
    
    # kinetic energy u² + v²
    u²_T, u² = IuT_u1, ITu_ksq      # reuse and rename arrays for u²
    v²_T, v² = IvT_v1, ITv_ksq      # and v², _T is on T-grid
    u² .= u.^2
    v² .= v.^2
    @inplacemul u²_T = IuT * u² 
    @inplacemul v²_T = IvT * v²
    kinetic .= u²_T .+ v²_T
    
    # Kloewer defined new terms q and p corresponding to potential vorticity and 
    # Bernoulli potential respectively. To avoid errors in my mimic I'm following 
    # along and doing the same 
    @inplacemul Guy_u1 = Guy * u
    @inplacemul Gvx_v1 = Gvx * v
    
    q .= (coriolis .+ Gvx_v1 .- Guy_u1) ./ h_q 
    p .= 0.5 .* kinetic .+ g .* h

    # bottom friction
    kinetic_sq .= sqrt.(kinetic)
    @inplacemul ITu_ksq = ITu * kinetic_sq
    @inplacemul ITv_ksq = ITv * kinetic_sq
    bfric_u .= bottom_drag .* ITu_ksq .* u ./ h_u
    bfric_v .= bottom_drag .* ITv_ksq .* v ./ h_v

    # deal with the advection term 
    # comp_advection(nx, rhs, advec)

    # diffusion term ν∇⁴(u,v)
    @inplacemul nu_u = ITu * nu
    @inplacemul nu_v = ITv * nu
    @inplacemul Mu = LLu * u
    @inplacemul Mv = LLv * v
    Mu .*= nu_u
    Mv .*= nu_v

    # bernoulli gradient ∇p = ∇(1/2(u²+v² + gh))
    @inplacemul GTx_p = GTx * p
    @inplacemul GTy_p = GTy * p

    # momentum equations
    u_t .= adv_u .- GTx_p .+ wind_stress ./ h_u .- Mu .- bfric_u
    v_t .= adv_v .- GTy_p .- Mv .- bfric_v 

    # continuity equations
    U .= u .* h_u                  # volume fluxes U,V
    V .= v .* h_v 
    @inplacemul Gux_U = Gux * U    # volume flux divergence dUdx + dVdy
    @inplacemul Gvy_V = Gvy * V
    @. eta_t = -(Gux_U + Gvy_V)

    return nothing
end 

function comp_u_v_eta_t(nx::Int, 
        rhs::SWM_pde, 
        params::Params, 
        interp::Interps, 
        grad::Derivatives, 
        advec::Advection
    ) 

    rhs.h .= rhs.eta1 .+ params.H 

    rhs.h_u .= interp.ITu * rhs.h
    rhs.h_v .= interp.ITv * rhs.h
    rhs.h_q .= interp.ITq * rhs.h

    rhs.U .= rhs.u1 .* rhs.h_u 
    rhs.V .= rhs.v1 .* rhs.h_v 

    rhs.kinetic .= interp.IuT * (rhs.u1.^2) .+ interp.IvT * (rhs.v1.^2)

    # Kloewer defined new terms q and p corresponding to potential vorticity and 
    # Bernoulli potential respectively. To avoid errors in my mimic I'm following 
    # along and doing the same 
    rhs.q .= (params.coriolis .+ grad.Gvx * rhs.v1 .- grad.Guy * rhs.u1) ./ rhs.h_q 
    rhs.p .= 0.5 .* rhs.kinetic .+ params.g .* rhs.h

    # bottom friction
    rhs.kinetic_sq .= sqrt.(rhs.kinetic)
    rhs.bfric_u .= params.bottom_drag .* ((interp.ITu * rhs.kinetic_sq) .* rhs.u1) ./ rhs.h_u
    rhs.bfric_v .= params.bottom_drag .* ((interp.ITv * rhs.kinetic_sq) .* rhs.v1) ./ rhs.h_v

    # deal with the advection term 
    comp_advection(nx, rhs, advec)

    # rhs.Mu .= params.A_h .* (grad.LLu * rhs.u1)
    # rhs.Mv .= params.A_h .* (grad.LLv * rhs.v1) 

    rhs.Mu .= (interp.ITu * params.nu) .* (grad.LLu * rhs.u1)
    rhs.Mv .= (interp.ITv * params.nu) .* (grad.LLv * rhs.v1) 

    rhs.u_t .= rhs.adv_u .- grad.GTx * rhs.p .+ params.wind_stress ./ rhs.h_u .- rhs.Mu .- rhs.bfric_u

    rhs.v_t .= rhs.adv_v .- grad.GTy * rhs.p .- rhs.Mv .- rhs.bfric_v 

    rhs.eta_t .= - (grad.Gux * rhs.U .+ grad.Gvy * rhs.V)

    return nothing

end 

function comp_advection(nx::Int, rhs::SWM_pde, advec::Advection)

    rhs.AL1q .= advec.AL1 * rhs.q 
    rhs.AL2q .= advec.AL2 * rhs.q 

    AL1q_au = @view rhs.AL1q[1:end-nx]
    AL2q_bu = @view rhs.AL2q[nx+1:end]
    AL2q_cu = @view rhs.AL2q[1:end-nx]
    AL1q_du = @view rhs.AL1q[nx+1:end]

    AL1q_av = @view rhs.AL1q[advec.index_av]
    AL2q_bv = @view rhs.AL2q[advec.index_bv]
    AL2q_cv = @view rhs.AL2q[advec.index_cv]
    AL1q_dv = @view rhs.AL1q[advec.index_dv]

    rhs.adv_u .= advec.Seur * (advec.ALeur * rhs.q .* rhs.U) .+ advec.Seul * (advec.ALeul * rhs.q .* rhs.U) .+
    advec.Sau * (AL1q_au .* rhs.V) .+ advec.Sbu * (AL2q_bu .* rhs.V) .+ 
    advec.Scu * (AL2q_cu .* rhs.V) .+ advec.Sdu * (AL1q_du .* rhs.V)

    rhs.adv_v .= advec.Spvu * ((advec.ALpvu * rhs.q) .* rhs.V) .+ advec.Spvd * ((advec.ALpvd * rhs.q) .* rhs.V) .-
    advec.Sav * (AL1q_av .* rhs.U) .- advec.Sbv * (AL2q_bv .* rhs.U) .- 
    advec.Scv * (AL2q_cv .* rhs.U) .- advec.Sdv * (AL1q_dv .* rhs.U)

    return nothing

end

function comp_advection(nx::Int, rhs::RHS_terms, advec::Advection)

    rhs.AL1q .= advec.AL1 * rhs.q 
    rhs.AL2q .= advec.AL2 * rhs.q 

    AL1q_au = @view rhs.AL1q[1:end-nx]
    AL2q_bu = @view rhs.AL2q[nx+1:end]
    AL2q_cu = @view rhs.AL2q[1:end-nx]
    AL1q_du = @view rhs.AL1q[nx+1:end]

    AL1q_av = @view rhs.AL1q[advec.index_av]
    AL2q_bv = @view rhs.AL2q[advec.index_bv]
    AL2q_cv = @view rhs.AL2q[advec.index_cv]
    AL1q_dv = @view rhs.AL1q[advec.index_dv]

    rhs.adv_u .= advec.Seur * (advec.ALeur * rhs.q .* rhs.U) .+ advec.Seul * (advec.ALeul * rhs.q .* rhs.U) .+
    advec.Sau * (AL1q_au .* rhs.V) .+ advec.Sbu * (AL2q_bu .* rhs.V) .+ 
    advec.Scu * (AL2q_cu .* rhs.V) .+ advec.Sdu * (AL1q_du .* rhs.V)

    rhs.adv_v .= advec.Spvu * ((advec.ALpvu * rhs.q) .* rhs.V) .+ advec.Spvd * ((advec.ALpvd * rhs.q) .* rhs.V) .-
    advec.Sav * (AL1q_av .* rhs.U) .- advec.Sbv * (AL2q_bv .* rhs.U) .- 
    advec.Scv * (AL2q_cv .* rhs.U) .- advec.Sdv * (AL1q_dv .* rhs.U)

    return nothing

end