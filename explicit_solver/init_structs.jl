# This script will contain the structures needed for the barotropic gyre model

mutable struct gyre_matrix
    u::Matrix{Float64}
    v::Matrix{Float64}
    eta::Matrix{Float64}
end

mutable struct gyre_vector
    u::Vector{Float64}
    v::Vector{Float64}
    eta::Vector{Float64}
end

# Combines the contents of gyre_vector with RHS_terms, explicitly needed for Checkpointing.jl
@with_kw mutable struct SWM_pde

    # To initialize struct just need to specify the following, the rest
    # of the entries will follow
    Nu::Int
    Nv::Int
    NT::Int
    Nq::Int

    u::Vector{Float64} = zeros(Nu)
    v::Vector{Float64} = zeros(Nv)
    eta::Vector{Float64} = zeros(NT)

    # Placeholder for cost function computation 
    J::Float64 = 0.0
    energy::Float64 = 0.0 

    # # interpolation operators to move the viscosity parameter to velocity grids 
    # Inu_u::Matrix{Float64} 
    # Inu_v::Matrix{Float64}

    # Placeholder for the scaling factor between the highres and lowres models 
    scaling::Float64 = 0.0

    # Placeholder for total steps to integrate for 
    T::Int = 0

    # since everything that matters to the derivative needs to live in a single structure, 
    # this will also contain all of the placeholders for terms on the RHS of the system

    # Appear in advance
    umid::Vector{Float64} = zeros(Nu)
    vmid::Vector{Float64} = zeros(Nv)
    etamid::Vector{Float64} = zeros(NT)

    u0::Vector{Float64} = zeros(Nu) 
    v0::Vector{Float64} = zeros(Nv)
    eta0::Vector{Float64} = zeros(NT)

    u1::Vector{Float64} = zeros(Nu)
    v1::Vector{Float64} = zeros(Nv)
    eta1::Vector{Float64} = zeros(NT)

    # Appear in comp_u_v_eta_t
    h::Vector{Float64} = zeros(NT)          # height of water columns [meters]

    h_u::Vector{Float64} = zeros(Nu)        # height of water columns interpolated to u-grid 
    h_v::Vector{Float64} = zeros(Nv)        # height of water columns interpolated to v-grid 
    h_q::Vector{Float64} = zeros(Nq)

    U::Vector{Float64} = zeros(Nu)
    V::Vector{Float64} = zeros(Nv)

    IuT_u1::Vector{Float64} = zeros(NT)
    IvT_v1::Vector{Float64} = zeros(NT)
    kinetic::Vector{Float64} = zeros(NT)

    kinetic_sq::Vector{Float64} = zeros(NT)

    Gvx_v1::Vector{Float64} = zeros(Nq)
    Guy_u1::Vector{Float64} = zeros(Nq)
    q::Vector{Float64} = zeros(Nq)
    p::Vector{Float64} = zeros(NT)

    ITu_ksq::Vector{Float64} = zeros(Nu)
    ITv_ksq::Vector{Float64} = zeros(Nv)
    bfric_u::Vector{Float64} = zeros(Nu)
    bfric_v::Vector{Float64} = zeros(Nv)

    LLu_u1::Vector{Float64} = zeros(Nu)
    LLv_v1::Vector{Float64} = zeros(Nv)
    Mu::Vector{Float64} = zeros(Nu)
    Mv::Vector{Float64} = zeros(Nv)

    GTx_p::Vector{Float64} = zeros(Nu)
    u_t::Vector{Float64} = zeros(Nu)
    GTy_p::Vector{Float64} = zeros(Nv)
    v_t::Vector{Float64} = zeros(Nv)
    Gux_U::Vector{Float64} = zeros(NT)
    Gvy_V::Vector{Float64} = zeros(NT)
    eta_t::Vector{Float64} = zeros(NT)
    adv_u::Vector{Float64} = zeros(Nu)
    adv_v::Vector{Float64} = zeros(Nv)

    # Appear in comp_advection 
    AL1q::Vector{Float64} = zeros(NT)
    AL2q::Vector{Float64} = zeros(NT)

end

# Parameters that appear in the model in various places
struct Params 
    dt::Float64                     # timestep
    g::Float64                      # gravity
    f0::Float64                     # Coriolis parameter
    beta::Float64                   # Coriolis parameter
    H::Float64                      # ocean depth 
    A_h::Float64                    # horizontal Laplacian viscosity 
    nu::Vector{Float64}             # matrix of nu values (not yet sure if this is the right idea, but its a start)
    ρ_c::Float64                    # reference density 
    bottom_drag::Float64            # bottom drag coefficient
    wind_stress::Vector{Float64}    # wind stress values on the grid 
    coriolis::Vector{Float64}       # coriolis values on the grid 
    rk_a::Vector{Float64}
    rk_b::Vector{Float64}
end


# Grid constants
struct Grid
    Lx::Int             # length of box in x direction [meters]
    Ly::Int             # length of box in y direction [meters]
    nx::Int             # total number of grid cells in x direction
    ny::Int             # total number of grid cells in y direction
    NT::Int             # total number of cells on eta-grid 
    Nu::Int             # total number of cells on u-grid
    Nv::Int             # total number of cells on v-grid
    Nq::Int             # total number of cells on vorticity grid 
    dx::Float64 
    dy::Float64 
end 

# The gyre model written by Kloewer and used by Zanna leaves all the states as vectors stacked 
# row-wise. So, computing things like gradients, Laplacians, etc. become matrix operations, and 
# in the following structures we store all of these operators for use in computing the RHS of the system 
# (namely the time derivatives). Boundary conditions are built into the derivative operators. 

# Discrete derivative operators. To avoid mistakes (and because the convention is nice) I mimicked Milan's 
# method for labelling these operators. For example, GTx is the x-derivative for elements living in cell-centers 
# and applying this operator results in a vector whose elements live on the u-grid 
struct Derivatives

    # Discrete gradients 
    GTx::SparseMatrixCSC{Float64, Int64}        
    GTy::SparseMatrixCSC{Float64, Int64}
    Gux::SparseMatrixCSC{Float64, Int64}
    Guy::SparseMatrixCSC{Float64, Int64}
    Gvx::SparseMatrixCSC{Float64, Int64}
    Gvy::SparseMatrixCSC{Float64, Int64}
    Gqy::SparseMatrixCSC{Float64, Int64}
    Gqx::SparseMatrixCSC{Float64, Int64}

    # Discrete Laplacian operators 
    Lu::SparseMatrixCSC{Float64, Int64}
    Lv::SparseMatrixCSC{Float64, Int64}
    LT::SparseMatrixCSC{Float64, Int64}
    Lq::SparseMatrixCSC{Float64, Int64}
    LLu::SparseMatrixCSC{Float64, Int64}
    LLv::SparseMatrixCSC{Float64, Int64}
end

# Interpolation operators, move from one grid to another. For example Ivu moves from the v-grid (horizontal faces)
# to elements on the u-grid
struct Interps 
    Ivu::SparseMatrixCSC{Float64, Int64}
    Iuv::SparseMatrixCSC{Float64, Int64}
    IqT::SparseMatrixCSC{Float64, Int64}
    IuT::SparseMatrixCSC{Float64, Int64}
    IvT::SparseMatrixCSC{Float64, Int64}
    ITu::SparseMatrixCSC{Float64, Int64}
    ITv::SparseMatrixCSC{Float64, Int64}
    Iqu::SparseMatrixCSC{Float64, Int64}
    Iqv::SparseMatrixCSC{Float64, Int64}
    Iuq::SparseMatrixCSC{Float64, Int64}
    Ivq::SparseMatrixCSC{Float64, Int64}
    ITq::SparseMatrixCSC{Float64, Int64}
end

# These are very specific operators that appear only in the computation of the advection terms, when 
# using the Arakawa and Lamb advection scheme. The best place to read about them is in Milan's lovely 
# documentation for his Python code 
struct Advection
    AL1::SparseMatrixCSC{Float64, Int64}
    AL2::SparseMatrixCSC{Float64, Int64}
    index_av::Vector{Int64}
    index_bv::Vector{Int64}
    index_cv::Vector{Int64}
    index_dv::Vector{Int64}
    ALeur::SparseMatrixCSC{Float64, Int64}
    ALeul::SparseMatrixCSC{Float64, Int64}
    Seul::SparseMatrixCSC{Float64, Int64}
    Seur::SparseMatrixCSC{Float64, Int64}
    Sau::SparseMatrixCSC{Float64, Int64}
    Sbu::SparseMatrixCSC{Float64, Int64}
    Scu::SparseMatrixCSC{Float64, Int64}
    Sdu::SparseMatrixCSC{Float64, Int64}
    ALpvu::SparseMatrixCSC{Float64, Int64}
    ALpvd::SparseMatrixCSC{Float64, Int64}
    Spvu::SparseMatrixCSC{Float64, Int64}
    Spvd::SparseMatrixCSC{Float64, Int64}
    Sav::SparseMatrixCSC{Float64, Int64}
    Sbv::SparseMatrixCSC{Float64, Int64}
    Scv::SparseMatrixCSC{Float64, Int64}
    Sdv::SparseMatrixCSC{Float64, Int64}
end


#### Removing, not needed anymore 

# # per a suggestion, I'm creating a new structure that will pre-allocate space to operators that only appear 
# # during the timestepping loop (mainly appear in the computation of the RHS equation)
# # this might become obsolete as I add checkpointing, as all of the advection operators matter to the gradient
# @with_kw struct RHS_terms
    
#     # To initialize struct just need to specify the following, the rest
#     # of the entries will follow
#     Nu::Int
#     Nv::Int
#     NT::Int
#     Nq::Int

#     # Appear in advance
#     umid::Vector{Float64} = zeros(Nu)
#     vmid::Vector{Float64} = zeros(Nv)
#     etamid::Vector{Float64} = zeros(NT)
#     u0::Vector{Float64} = zeros(Nu) 
#     v0::Vector{Float64} = zeros(Nv)
#     eta0::Vector{Float64} = zeros(NT)
#     u1::Vector{Float64} = zeros(Nu)
#     v1::Vector{Float64} = zeros(Nv)
#     eta1::Vector{Float64} = zeros(NT)

#     # Appear in comp_u_v_eta_t
#     h::Vector{Float64} = zeros(NT)          # height of water columns [meters]

#     h_u::Vector{Float64} = zeros(Nu)        # height of water columns interpolated to u-grid 
#     h_v::Vector{Float64} = zeros(Nv)        # height of water columns interpolated to v-grid 
#     h_q::Vector{Float64} = zeros(Nq)

#     U::Vector{Float64} = zeros(Nu)
#     V::Vector{Float64} = zeros(Nv)

#     IuT_u1::Vector{Float64} = zeros(NT)
#     IvT_v1::Vector{Float64} = zeros(NT)
#     kinetic::Vector{Float64} = zeros(NT)

#     kinetic_sq::Vector{Float64} = zeros(NT)

#     Gvx_v1::Vector{Float64} = zeros(Nq)
#     Guy_u1::Vector{Float64} = zeros(Nq)
#     q::Vector{Float64} = zeros(Nq)
#     p::Vector{Float64} = zeros(NT)

#     ITu_ksq::Vector{Float64} = zeros(Nu)
#     ITv_ksq::Vector{Float64} = zeros(Nv)
#     bfric_u::Vector{Float64} = zeros(Nu)
#     bfric_v::Vector{Float64} = zeros(Nv)

#     LLu_u1::Vector{Float64} = zeros(Nu)
#     LLv_v1::Vector{Float64} = zeros(Nv)
#     Mu::Vector{Float64} = zeros(Nu)
#     Mv::Vector{Float64} = zeros(Nv)

#     GTx_p::Vector{Float64} = zeros(Nu)
#     u_t::Vector{Float64} = zeros(Nu)
#     GTy_p::Vector{Float64} = zeros(Nv)
#     v_t::Vector{Float64} = zeros(Nv)
#     Gux_U::Vector{Float64} = zeros(NT)
#     Gvy_V::Vector{Float64} = zeros(NT)
#     eta_t::Vector{Float64} = zeros(NT)
#     adv_u::Vector{Float64} = zeros(Nu)
#     adv_v::Vector{Float64} = zeros(Nv)

#     # Appear in comp_advection 
#     AL1q::Vector{Float64} = zeros(NT)
#     AL2q::Vector{Float64} = zeros(NT)
    
# end

