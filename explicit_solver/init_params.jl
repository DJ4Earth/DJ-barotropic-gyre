# this function will define all of the desired parameters and return them in a structure for use in other locations
function def_params(grid)

    nx = grid.nx
    ny = grid.ny
    dx = grid.dx 
    dy = grid.dx
    Lx = grid.Lx
    Ly = grid.Ly

    NT = grid.NT

    x = (dx/2):dx:Lx
    y = dy/2:dy:Ly

    xu = x[1:end-1] .+ dx/2 
    yu = copy(y)
    
    xv = copy(x) 
    yv = y[1:end-1] .+ dy/2 
    
    xq = 0:dx:(Lx + dx/2)
    yq = 0:dy:(Ly + dy/2)

    # nu_A = 128*540/(min(nx, ny))     # harmonic mixing coefficient (chosen so that nu_A = 540 meters^2 / sec when dx = 30 km)
    # A_h = nu_A * max(dx, dy)^2       # biharmonic mixing coefficient coefficient [meters^2 / second]

    A_h = compute_viscosity(nx, ny)
    nu = A_h .* ones(NT)             # placing the viscosity coefficient on the tracer grid (cell centers)

    rho_c = 1000.0                   # density
    bottom_drag = 1e-5               # bottom-drag coefficient
    g = 9.81                         # gravity [meters^2 / second]
    H = 500.0                        # depth of the box [meters]
    omega = 2 * pi / (24 * 3600)     # rotation frequency 
    R = 6.371e6                      # Earth's radius [meters]
    theta0 = 35                      # Latitude for beta-plane approx

    f0 = 2 * omega * sin(theta0 * pi / 180)                 # Coriolis parameter
    beta = (2 * omega / R) * cos(theta0 * pi / 180)         # beta of beta-plane approx

    # building the vectors that will contain the coriolis force and wind stress (techincally forcings and not parameters I know)
    # come back to this, want to understand 
    Yq_shift = yq .- Ly/2
    Yq = vec([k for k in Yq_shift, j in 1:nx+1]')
    f(y) = f0 + beta * y
    coriolis = vec(f.(Yq)')

    Yu = vec([k for k in yu, j in 1:length(xu)]')
    ws_cos = cos.(2 * pi .* ((Yu .- Ly/2)./Ly) )
    ws_sin = 2 .* sin.(pi .* ((Yu .- Ly/2)./Ly)  ) 
    wind_stress = 0.12 .* (ws_cos + ws_sin) ./rho_c

    # removing the requirement that dt be an integer, not sure why that's there 
    dt = (0.9 * min(dx, dy)) / (sqrt(g * H))   # CFL condition for dt [seconds]

    # we use RK4 as the timestepper, here I'm storing the coefficients needed for this 
    rk_a = [1/6, 1/3, 1/3, 1/6]
    rk_b = [1/2, 1/2, 1.]

    gyre_params = Params(
    dt,
    g, 
    f0, 
    beta, 
    H, 
    A_h,
    nu, 
    rho_c, 
    bottom_drag,  
    wind_stress, 
    coriolis,
    rk_a,
    rk_b
    )

    return gyre_params 

end


# This function allows me to specify the number of days that we want to run the model, 
# and convert that into the total steps to take 
function days_to_seconds(number_of_days, dt)

    Trun = Int(ceil((number_of_days * 24 * 3600) / dt))

    return Trun

end

# I've decided to add a function that computes the viscosity coefficient based on 
# the grid, so that in the future I can more easily edit how this parameter
# is chosen. Technically possible to leave this in init_params, but nice to have 
# it here
function compute_viscosity(nx, ny)

    return (128 * 540) / min(nx, ny)

end