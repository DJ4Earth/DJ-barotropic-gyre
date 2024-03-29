# Contains two functions, the first of which computes the dissipative terms, namely
# it computes the bottom friction and diffusion terms and applies these to the time derivatives
# The second function 

function dissipative_terms!(nx::Int, 
    rhs::SWM_pde, 
    params::Params, 
    interp::Interps, 
    grad::Derivatives, 
    advec::Advection
)

# unpack stuff
u = rhs.u0                          # calculate based on prognostics at
v = rhs.v0                          # t + dt of the non-dissipative RHS

(;ITu, ITv) = interp                # interpolation operators   
(;LLu, LLv) = grad                  # gradient operators
(;h_u, h_v) = rhs                   # diagnostic variables
(;kinetic, kinetic_sq, Mu, Mv, nu_u, nu_v) = rhs
(;bfric_u, bfric_v) = rhs
(;ITu_ksq,ITv_ksq) = rhs
(;u_t, v_t) = rhs                   # tendencies
(;nu, bottom_drag) = params

# bottom friction
kinetic_sq .= sqrt.(kinetic)
@inplacemul ITu_ksq = ITu * kinetic_sq
@inplacemul ITv_ksq = ITv * kinetic_sq
bfric_u .= bottom_drag .* ITu_ksq .* u ./ h_u
bfric_v .= bottom_drag .* ITv_ksq .* v ./ h_v

### Important: Need to apply Laplacian to nu for when nu varies spatially ##################
# diffusion term ν∇⁴(u,v)
@inplacemul nu_u = ITu * nu
@inplacemul nu_v = ITv * nu
@inplacemul Mu = LLu * u
@inplacemul Mv = LLv * v
Mu .*= nu_u
Mv .*= nu_v

# tendencies for bottom friction and diffusion
u_t .= .- Mu .- bfric_u
v_t .= .- Mv .- bfric_v 

return nothing
end

function comp_u_v_eta_t(nx::Int, 
    states_rhs::SWM_pde, 
    params::Params, 
    interp::Interps, 
    grad::Derivatives, 
    advec::Advection
) 

# unpack stuff
u = states_rhs.u1
v = states_rhs.v1
eta = states_rhs.eta1

(;ITu, ITv, ITq, IuT, IvT) = interp                 # interpolation operators
(;Iuv, Ivu, Iqu, Iqv) = interp       
(;GTx, GTy, Gux, Guy, Gvx, Gvy) = grad              # gradient operators
(;h, h_u, h_v, h_q, U, V, p, q) = states_rhs               # diagnostic variables
(;kinetic) = states_rhs
(;GTx_p, GTy_p, Gux_U, Gvy_V, Gvx_v1, Guy_u1) = states_rhs
(;adv_u, adv_v) = states_rhs
(;IuT_u1, IvT_v1, ITu_ksq, ITv_ksq) = states_rhs
(;u_t, v_t, eta_t) = states_rhs                            # tendencies
(;H, coriolis, g, wind_stress) = params

h .= eta .+ H 

@inplacemul h_q = ITq * h
@inplacemul h_u = ITu * h
@inplacemul h_v = ITv * h
U .= u .* h_u                  # volume fluxes U,V
V .= v .* h_v 

# kinetic energy u² + v²
u2_T, u2 = IuT_u1, ITu_ksq      # reuse and rename arrays for u²
v2_T, v2 = IvT_v1, ITv_ksq      # and v², _T is on T-grid
u2 .= u.^2
v2 .= v.^2
@inplacemul u2_T = IuT * u2
@inplacemul v2_T = IvT * v2
kinetic .= u2_T .+ v2_T

# Kloewer defined new terms q and p corresponding to potential vorticity and 
# Bernoulli potential respectively. To avoid errors in my mimic I'm following 
# along and doing the same 
@inplacemul Guy_u1 = Guy * u
@inplacemul Gvx_v1 = Gvx * v

q .= (coriolis .+ Gvx_v1 .- Guy_u1) ./ h_q 
p .= 0.5 .* kinetic .+ g .* h

# Arakawa and Lamb advection scheme
# comp_ALadvection(nx, states_rhs, advec)    

# Sadourny, 1975 enstrophy conserving advection scheme
V_u = ITu_ksq                # reuse and rename array
@inplacemul V_u = Ivu * V
@inplacemul adv_u = Iqu * q  # u-component qhv
adv_u .*= V_u

U_v = ITv_ksq                # reuse and rename array
@inplacemul U_v = Iuv * U         
@inplacemul adv_v = Iqv * q  # v-component -qhu
adv_v .*= .-U_v

# Bernoulli gradient ∇p = ∇(1/2(u²+v² + gh))
@inplacemul GTx_p = GTx * p
@inplacemul GTy_p = GTy * p

# momentum equations
u_t .= adv_u .- GTx_p .+ wind_stress ./ h_u
v_t .= adv_v .- GTy_p

# continuity equations
@inplacemul Gux_U = Gux * U    # volume flux divergence dUdx + dVdy
@inplacemul Gvy_V = Gvy * V
@. eta_t = -(Gux_U + Gvy_V)

return nothing

end 

function comp_ALadvection(nx::Int, rhs::SWM_pde, advec::Advection)

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
