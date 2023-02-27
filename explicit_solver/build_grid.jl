# This function builds all the parameters relating to the grid. It takes 
#       Lx - the length of the domain from E-W [meters]
#       Ly - the length of the domain from N-S [meters]
#       Nx - the number of cells in the E-W direction 
#       Ny - the number of cells in the N-S direction 
# and returns a structure with a bunch of different parameters all built
# from the above inputs. 
function build_grid(Lx, Ly, nx, ny)

    dx = Lx / nx 
    dy = Ly / ny 

    NT = nx * ny 

    Nu = (nx - 1) * ny 
    Nv = (ny - 1) * nx 
    Nq = (nx + 1) * (ny + 1)

    x = (dx/2):dx:Lx
    y = dy/2:dy:Ly

    xu = x[1:end-1] .+ dx/2 
    yu = copy(y)
    
    xv = copy(x) 
    yv = y[1:end-1] .+ dy/2 
    
    xq = 0:dx:(Lx + dx/2)
    yq = 0:dy:(Ly + dy/2)

    grid_params = Grid(
    Lx, 
    Ly,
    nx, 
    ny,
    NT,
    Nu, 
    Nv,
    Nq,
    dx, 
    dy
    )
    
    return grid_params

end

# This function allows me to specify the number of days that we want to run the model, 
# and convert that into the total steps to take 
function days_to_seconds(Tspinup_days, Trun_days, dt)

    Tspinup_seconds = Int(ceil((Tspinup_days * 24 * 3600) / dt))
    Trun_seconds = Int(ceil((Trun_days * 24 * 3600) / dt))

    return Tspinup_seconds, Trun_seconds

end

# This function will just serve to take the vectors containing state information 
# and transform them to matrices 
function vec_to_mat(u, v, eta, grid)

    state_matrices = gyre_matrix(
        reshape(u, grid.nx-1, grid.ny)',
        reshape(v, grid.nx, grid.ny-1)',
        reshape(eta, grid.nx, grid.ny)'
    )

    return state_matrices

end
