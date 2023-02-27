# Adapted from https://github.com/CliMA/Oceananigans.jl/blob/main/validation/barotropic_gyre/barotropic_gyre.jl

using Oceananigans
using Oceananigans.Units
using Oceananigans: time_step!
using Printf
using OffsetArrays, Enzyme
using Enzyme

using Enzyme_jll

Enzyme.API.looseTypeAnalysis!(true)

# @show Enzyme_jll.libEnzyme

grid = LatitudeLongitudeGrid(size = (60, 60, 1),
                             longitude = (-30, 30),
                             latitude = (15, 75),
                             z = (-4000, 0))

@inline wind_stress(λ, φ, t, p) = p.τ₀ * cos(2π * (φ - p.φ₀) / p.Lφ)

wind_stress_bc = FluxBoundaryCondition(wind_stress, parameters = (τ₀=1e-4, Lφ=grid.Ly, φ₀=15))

@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, 1]
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, 1]

μ = 1 / 60days
u_bottom_drag_bc = FluxBoundaryCondition(u_bottom_drag, discrete_form=true, parameters=μ)
v_bottom_drag_bc = FluxBoundaryCondition(v_bottom_drag, discrete_form=true, parameters=μ)

u_bcs = FieldBoundaryConditions(top=wind_stress_bc, bottom=u_bottom_drag_bc)
v_bcs = FieldBoundaryConditions(bottom=v_bottom_drag_bc)

νh₀ = 5e3 * (60 / grid.Nx)^2
closure = HorizontalScalarDiffusivity(ν = νh₀)

model = HydrostaticFreeSurfaceModel(; grid, closure,
                                    free_surface = ExplicitFreeSurface(),
                                    momentum_advection = VectorInvariant(),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    boundary_conditions = (u=u_bcs, v=v_bcs),
                                    tracers = nothing,
                                    buoyancy = nothing)

dmodel = deepcopy(model)
    
# HydrostaticFreeSurfaceModel(; grid, closure,
#                 free_surface = ExplicitFreeSurface(),
#                 momentum_advection = VectorInvariant(),
#                 coriolis = HydrostaticSphericalCoriolis(),
#                 boundary_conditions = (u=u_bcs, v=v_bcs),
#                 tracers = nothing,
#                 buoyancy = nothing)

# dmodel = HydrostaticFreeSurfaceModel(; grid, closure)

# Compute stable time-step smaller than grid-scale wave phase propagation
g = model.free_surface.gravitational_acceleration
H = model.grid.Lz
c = sqrt(g * H) # gravity wave speed
Δx = minimum(grid.Δxᶜᶜᵃ[1:grid.Nx])
Δt = 0.1 * Δx / c

function ad_func(model, Δt)
    time_step!(model, Δt)
    return nothing
end

@info "Before taking a time step"
@show maximum(model.velocities.u)

dmodel.velocities.u .= 0. 
dmodel.velocities.v .= 0.

# u, v = model.velocities
# A = zeros(67,66,7)
# B = zeros(66,67,7)
# d_u_in = OffsetArray(A, -2:64, -2:63, -2:4)
# d_v_in = OffsetArray(B, -2:63, -2:64, -2:4)

# u_out = OffsetArray(zeros(67,66,7), -2:64, -2:63, -2:4)
# d_u_out = OffsetArray(ones(67,66,7), -2:64, -2:63, -2:4)

# v_out = OffsetArray(zeros(66,67,7), -2:63, -2:64, -2:4)
# d_v_out = OffsetArray(ones(66,67,7), -2:63, -2:64, -2:4)

autodiff(ad_func, Duplicated(model, dmodel), Const(Δt))
