# this will have just one function, which will compute the energy 
# in the model 

function energy(grid, u, v) 

    return sum(u.^2 .+ v.^2) / (grid.nx * grid.ny)
    
end