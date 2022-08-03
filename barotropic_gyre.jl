# Adapted from https://github.com/CliMA/Oceananigans.jl/blob/main/validation/barotropic_gyre/barotropic_gyre.jl

using Oceananigans
using Oceananigans.Units
using Oceananigans: time_step!
using Printf

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

u_bcs = FieldBoundaryConditions(top=surface_wind_stress_bc, bottom=u_bottom_drag_bc)
v_bcs = FieldBoundaryConditions(bottom=v_bottom_drag_bc)

νh₀ = 5e3 * (60 / grid.Nx)^2
closure = HorizontalScalarDiffusivity(ν = νh₀)

model = HydrostaticFreeSurfaceModel(; grid, closure,
                                    momentum_advection = VectorInvariant(),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    boundary_conditions = (u=u_bcs, v=v_bcs),
                                    tracers = nothing,
                                    buoyancy = nothing)

@info "Before taking a time step"
@show maximum(model.velocities.u)

time_step!(model, 6hours)

@info "After taking a time step"
@show maximum(model.velocities.u)
