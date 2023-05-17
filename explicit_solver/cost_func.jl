# this will have just one function, which will compute the cost function given
# a data value and the computed state at that given timestep

function data_misfit(M, d, u, v, eta)

    return dot((M * d - [u; v; eta]), (M * d - [u; v; eta]))

end

function energy(grid, u, v) 

    return sum(u.^2 .+ v.^2) / (grid.nx * grid.ny)
    
end