using Oceananigans
using Oceananigans: time_step!
using Printf

Nx = 60
Ny = 60

# A spherical domain
grid = LatitudeLongitudeGrid(size = (Nx, Ny, 1),
                             longitude = (-30, 30),
                             latitude = (15, 75),
                             z = (-4000, 0))

@show surface_wind_stress_parameters = (τ₀ = 1e-4,
                                        Lφ = grid.Ly,
                                        φ₀ = 15)

@inline surface_wind_stress(λ, φ, t, p) = p.τ₀ * cos(2π * (φ - p.φ₀) / p.Lφ)

surface_wind_stress_bc = FluxBoundaryCondition(surface_wind_stress,
                                               parameters = surface_wind_stress_parameters)

μ = 1 / (60*24*60*60)#60days
@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, 1]
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, 1]

u_bottom_drag_bc = FluxBoundaryCondition(u_bottom_drag, discrete_form = true, parameters = μ)
v_bottom_drag_bc = FluxBoundaryCondition(v_bottom_drag, discrete_form = true, parameters = μ)

u_bcs = FieldBoundaryConditions(top = surface_wind_stress_bc, bottom = u_bottom_drag_bc)
v_bcs = FieldBoundaryConditions(bottom = v_bottom_drag_bc)

νh₀ = 5e3 * (60 / grid.Nx)^2
constant_horizontal_diffusivity = HorizontalScalarDiffusivity(ν = νh₀)

model = HydrostaticFreeSurfaceModel(; grid,
                                    momentum_advection = VectorInvariant(),
                                    free_surface = ImplicitFreeSurface(gravitational_acceleration=0.1),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    boundary_conditions = (u=u_bcs, v=v_bcs),
                                    closure = constant_horizontal_diffusivity,
                                    tracers = nothing,
                                    buoyancy = nothing)

time_step!(model, 6*60*60)#6hours)

@show maximum(model.velocities.u)
@show maximum(model.velocities.v)

u, v = model.velocities

# set!(model, u=u.data, v=v.data)

# time_step!(model, 6*60*60)

# @show maximum(model.velocities.u)
# @show maximum(model.velocities.v)

# u, v = model.velocities

# set!(model, u=u.data, v=v.data)

# time_step!(model, 6*60*60)

# @show maximum(model.velocities.u)
# @show maximum(model.velocities.v)
