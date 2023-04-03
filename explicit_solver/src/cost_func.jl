# this will have just one function, which will compute the cost function given
# a data value and the computed state at that given timestep

function data_misfit(d, u, v, eta)

    return dot((d - [u; v; eta]), (d - [u; v; eta]))

end