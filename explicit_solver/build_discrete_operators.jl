# this function will build the discrete operators needed for things such as
# -- moving between grids 
# -- discrete gradients 
# -- discrete laplacians 
# throughout we enforce **free-slip** boundary conditions 

function build_derivs(grid)

    dx = grid.dx 
    dy = grid.dy 
    nx = grid.nx 
    ny = grid.ny

    NT = grid.NT 
    Nv = grid.Nv
    Nq = grid.Nq 

    # indices needed to enforce boundary conditions 
    index1 = 1:NT 
    index1 = filter(x -> x ∉ [nx * j for j in 1:NT], index1)
    index2 = 1:(Nv+(ny-1))
    index2 = filter(x -> x ∉ [(nx + 1)*j for j in 1:(Nv+(ny-1))], index2)

    GTx = (spdiagm(1 => ones(NT-1)) - I)[index1, :] ./ dx 
    GTy = (spdiagm(nx => ones(Nv)) - I)[1:end-nx, :] ./ dy

    # x-derivative from u-grid onto centers
    Gux = -copy(GTx')

    # y-derivative from v-grid onto centers
    Gvy = -copy(GTy')

    # y-derivative from u-grid to q-grid, setup includes boundary conditions
    diag1 = ones(Nq)
    diag1[1:nx+1:Nq] .= 0.0
    diag1[nx+1:nx+1:Nq] .= 0.0

    index3 = filter(x -> diag1[x] != 0, 1:length(diag1))
    diag1[end-nx+1:end] .= 0.0

    Guy1 = spdiagm(diag1)[:, index3][:, nx:end]
    Guy2 = reverse(reverse(Guy1, dims=1), dims=2)
    Guy = (Guy2 - Guy1) ./ dy 

    # y-derivative from q-grid to u-grid, again including boundary conditions
    diag1[end-nx+1:end] .= 1.0
    Gqy1 = spdiagm(diag1)[:,index3][:, nx:end]
    Gqy2 = reverse(reverse(Gqy1, dims=1), dims=2)
    Gqy = (Gqy1 - Gqy2)' ./ dy

    # x-derivative from v-grid to q-grid
    diag2 = ones(Nv + (ny - 1))
    diag2[1:nx+1:end] .= 0.0
    index22 = [Nv + ny - j for j in index2[end:-1:1]]
    Gvx1 = (spdiagm(Nq, Nv + (ny - 1), -(nx+1) => diag2))[:, index2]
    Gvx2 = spdiagm(Nq, Nv + (ny - 1), -(nx+1) => (-diag2[end:-1:1]))[:, index22] 
    Gvx = (Gvx2 + Gvx1) ./ dx

    # x-derivative from q-grid to v-grid
    diag2[1:(nx+1):end] .= 1.0
    Gqx1 = (spdiagm(Nq, Nv + (ny - 1), -(nx+1) => diag2))[:, index2] 
    Gqx2 = spdiagm(Nq, Nv + (ny - 1), -(nx+1) => -diag2[end:-1:1])[:, index22]   
    Gqx = -(Gqx1 + Gqx2)' ./ dx 

    # using above we can now build the discrete Laplacian operators

    Lu = copy(GTx) * copy(Gux) + copy(Gqy) * copy(Guy)
    Lv = copy(Gqx) * copy(Gvx) + copy(GTy) * copy(Gvy)
    LT = copy(Gux) * copy(GTx) + copy(Gvy) * copy(GTy)
    Lq = copy(Gvx) * copy(Gqx) + copy(Guy) * copy(Gqy) 

    LLu = Lu * Lu 
    LLv = Lv * Lv 

    grad_ops = Derivatives(
    GTx, 
    GTy, 
    Gux, 
    Guy, 
    Gvx, 
    Gvy, 
    Gqy, 
    Gqx, 
    Lu, 
    Lv, 
    LT, 
    Lq,
    LLu,
    LLv
    )
    
    return grad_ops

end

function build_interp(grid, grad_ops)

    dx = grid.dx 
    dy = grid.dy 
    nx = grid.nx
    ny = grid.ny
    NT = grid.NT
    Nu = grid.Nu
    Nv = grid.Nv
    Nq = grid.Nq 

    Gux = grad_ops.Gux
    Gvy = grad_ops.Gvy
    GTx = grad_ops.GTx
    GTy = grad_ops.GTy

    index1 = 1:(Nv + nx)
    index1 = filter(x -> x ∉ [nx * j for j in 1:(Nv + nx)], index1)
    index2 = 1:(NT + ny - 1)
    index2 = filter(x -> x ∉ [(nx + 1)*j for j in 1:(NT+(ny-1))], index2)

    # move between v-grid and u-grid, a four point averaging matrix 
    diag1 = ones(Nv) ./ 4
    Ivu1 = spdiagm(Nv+nx, Nv, diag1)
    Ivu2 = spdiagm(Nv+nx, Nv, 1 => diag1[1:end-1])
    Ivu3 = spdiagm(Nv+nx, Nv, (-nx + 1) => diag1)
    Ivu4 = spdiagm(Nv+nx, Nv, -nx => diag1)
    Ivu = (Ivu1 + Ivu2 + Ivu3 + Ivu4)[index1, :]

    # move between u-grid and v-grid
    Iuv = copy(Ivu') 

    # move between u-grid and T-grid (centers), again a four point average
    diag2 = ones(NT + ny - 1) ./ 4 
    IqT1 = spdiagm(NT + ny - 1, Nq, diag2)
    IqT2 = spdiagm(NT + ny - 1, Nq, 1 => diag2)
    IqT3 = spdiagm(NT + ny - 1, Nq, (nx + 1) => diag2)
    IqT4 = spdiagm(NT + ny - 1, Nq, (nx + 2) => diag2)
    IqT = (IqT1 + IqT2 + IqT3 + IqT4)[index2, :]

    # u-grid to T-grid, now a two-point average 
    IuT = abs.(copy(Gux) .* dx ./ 2)

    # v-grid to T-grid
    IvT = abs.(copy(Gvy) .* dy ./ 2)

    # T-grid to u-grid
    ITu = abs.(copy(GTx) .* dx ./ 2)

    # T-grid to v-grid 
    ITv = abs.(copy(GTy) .* dy ./ 2)

    # q-grid to u-grid 
    diag3 = ones(Nq - nx - 1) ./ 2
    index3 = 1:(Nq - nx - 1)
    index3 = filter(x -> x ∉ [1 + (nx + 1) * j for j in 0:(Nq - nx - 1)], index3)
    index3 = filter(x -> x ∉ [j for j in nx+1:nx+1:(Nq - nx - 1)], index3)
    Iqu1 = spdiagm(Nq - nx - 1, Nq, diag3)
    Iqu2 = spdiagm(Nq - nx - 1, Nq, (nx + 1) => diag3)
    Iqu = (Iqu1 + Iqu2)[index3, :]

    # u-grid to q-grid 
    Iuq = copy(Iqu')
    Iuq[2:nx, 1:nx-1] .= spdiagm(ones(nx-1))
    Iuq[end-nx+1:end-1, end-nx+2:end] .= spdiagm(ones(nx-1))

    # q-grid to v-grid 
    diag4 = ones(Nv + ny - 2) ./ 2
    index4 = 1:Nv+ny-2 
    index4 = filter(x -> x ∉ [(nx + 1)*j for j in 1:(Nv+ny-2)], index4)
    Iqv1 = spdiagm(Nv + ny - 2, Nq, (nx+1) => diag4)
    Iqv2 = spdiagm(Nv + ny - 2, Nq, (nx+2) => diag4)
    Iqv = (Iqv1 + Iqv2)[index4, :]

    # v-grid to q-grid 
    Ivq = copy(Iqv')

    # I need a way to find the indices of the elements that correspond to the boundary,
    # and this is my probably inefficient way of doing this. If there's a better way
    # I should come back here 
    a1 = nx+2:nx+1:size(Ivq)[1]
    b1 = 1:nx:size(Ivq)[2]
    a2 = size(Ivq)[1]-nx-1:-nx-1:1 
    b2 = size(Ivq)[2]:-nx:1
    coords1 = diag(collect(Iterators.product(a1, b1)))
    coords2 = diag(collect(Iterators.product(a2, b2)))

    for j in eachindex(coords1)
        Ivq[coords1[j][1], coords1[j][2]] = 1.0
    end
    for j in eachindex(coords2)
        Ivq[coords2[j][1], coords2[j][2]] = 1.0
    end

    # move from T-grid to q-grid 
    diag5 = ones(4 * NT) ./ 4
    diag5[2:2*nx] .= 0.5
    diag5[end-2*nx+2:end] .= 0.5

    diag5[2*nx+1:4*nx:end-2*nx+1] .= 0.5 
    diag5[2*nx+2:4*nx:end-2*nx] .= 0.5

    diag5[end-2*nx:-4*nx:2*nx+1] .= 0.5 

    diag5[end-2*nx-1:-4*nx:2*nx] .= 0.5

    diag5[1] = 1.0 
    diag5[2*nx] = 1.0
    diag5[end-2*nx+1] = 1.0
    diag5[end] = 1.0    

    ITq = copy(IqT)' 
    counter = 1
    for j = 1:size(ITq)[1]
        for k = 1:size(ITq)[2]

        if ITq[j,k] != 0.0
            ITq[j,k] = diag5[counter]
            counter += 1 
        end

        end
    end

    interp_ops = Interps(
    Ivu, 
    Iuv, 
    IqT, 
    IuT, 
    IvT, 
    ITu, 
    ITv, 
    Iqu, 
    Iqv,
    Iuq, 
    Ivq, 
    ITq
    )

    return interp_ops

end

# the following function builds the discrete operators needed to compute the 
# advection term, **come back to this to add comments**
function build_advec(grid)

    dx = grid.dx 
    dy = grid.dy 
    nx = grid.nx
    ny = grid.ny
    NT = grid.NT
    Nu = grid.Nu
    Nv = grid.Nv
    Nq = grid.Nq 

    index1 = 1:(Nq-nx-1)
    index1 = filter(x -> x ∉ [(nx + 1)*j for j in 1:(Nq-nx-1)], index1)

    diag1 = ones(Nq) ./ 24 
    two_diag1 = 2 .* diag1
    AL11 = spdiagm(Nq + nx, Nq, two_diag1)
    AL12 = spdiagm(Nq + nx, Nq, 1 => diag1[1:end-1]) 
    AL13 = spdiagm(Nq + nx, Nq, (nx+1) => diag1[1:end-nx-1])
    AL14 = spdiagm(Nq + nx, Nq, (nx+2) => two_diag1[1:end-nx-2])
    AL1 = (AL11 + AL12 + AL13 + AL14)[index1, :]

    index_av = 1:nx^2
    index_av = filter(x -> x ∉ [j for j in nx:nx:nx^2], index_av)

    index_dv = 2:(nx^2 + 1) 
    index_dv = filter(x -> x ∉ [j for j in nx+1:nx:(nx^2 + 1)], index_dv)

    AL21 = spdiagm(Nq + nx, Nq, diag1)
    AL22 = spdiagm(Nq + nx, Nq, 1 => two_diag1[1:end-1]) 
    AL23 = spdiagm(Nq + nx, Nq, (nx+1) => two_diag1[1:end-nx-1])
    AL24 = spdiagm(Nq + nx, Nq, (nx+2) => diag1[1:end-nx-2])
    AL2 = (AL21 + AL22 + AL23 + AL24)[index1, :]

    index_bv = copy(index_dv)
    index_cv = copy(index_av)

    # U component of advection (see Klower notes for better description)
    diag2 = ones(NT + ny - 1) ./ 24
    index2 = 1:(NT+ny-1)
    index2 = filter(x -> x ∉ [nx + (nx + 1) * j for j in 0:(NT+ny-1)], index2)
    index2 = filter(x -> x ∉ [j for j in nx+1:nx+1:(NT + ny - 1)], index2)
    ALeur1 = spdiagm(NT + ny - 1, Nq, diag2) 
    ALeur2 = spdiagm(NT + ny - 1, Nq, 1 => diag2)
    ALeur3 = spdiagm(NT + ny - 1, Nq, nx+1 => -diag2)
    ALeur4 = spdiagm(NT + ny - 1, Nq, nx+2 => -diag2)
    ALeur = (ALeur1 + ALeur2 + ALeur3 + ALeur4)[index2, :]

    index3 = 1:(NT+ny-1)
    index3 = filter(x -> x ∉ [(nx + 1) * j for j in 1:(NT+ny-1)], index3)
    index3 = filter(x -> x ∉ [1 + (nx+1)*j for j in 0:(NT+ny-1)], index3)
    ALeul1 = spdiagm(NT + ny - 1, Nq, -diag2)
    ALeul2 = spdiagm(NT + ny - 1, Nq, 1 => -diag2)
    ALeul3 = spdiagm(NT + ny - 1, Nq, nx+1 => diag2)
    ALeul4 = spdiagm(NT + ny - 1, Nq, nx+2 => diag2)
    ALeul = (ALeul1 + ALeul2 + ALeul3 + ALeul4)[index3,:]

    ones1 = ones(Nu) 
    ones1[1:nx-1:end] .= 0.0 
    Seur = spdiagm(Nu, Nu, 1 => ones1[2:end])
    Seul = copy(Seur)'

    ones2 = ones(Nu + ny)
    index4 = 1:Nv+nx
    index4 = filter(x -> x ∉ [j for j in nx:nx:Nv+nx], index4)

    Sau = spdiagm(Nu + ny, Nv, 1 => ones2[1:Nv-1])[index4, :]
    Scu = spdiagm(Nu + ny, Nv, ones2[1:Nv])[index4,:]
    Sbu = spdiagm(Nu + ny, Nv, -(nx - 1) => ones2[1:Nv])[index4, :]
    Sdu = spdiagm(Nu + ny, Nv, -nx => ones2[1:Nv])[index4, :]

    # Finally, we create the operators for the V component of advection 
    index5 = 1:Nq-nx-1 
    index5 = filter(x -> x ∉ [j for j in nx+1:nx+1:Nq-nx-1], index5)
    index5 = filter(x -> x ∉ index5[end - nx + 1:end], index5)
    diag3 = ones(Nq) ./ 24
    ALpvu1 = spdiagm(Nq, Nq, -diag3)
    ALpvu2 = spdiagm(Nq, Nq, 1 => diag3[1:end-1])
    ALpvu3 = spdiagm(Nq, Nq, nx+1 => -diag3[1:end-nx-1])
    ALpvu4 = spdiagm(Nq, Nq, nx+2 => diag3[1:end-nx-2])
    ALpvu = (ALpvu1 + ALpvu2 + ALpvu3 + ALpvu4)[index5, :]

    index6 = 1:Nq-nx-1
    index6 = filter(x -> x ∉ [j for j in nx+1:nx+1:Nq-nx-1], index6)
    index6 = filter(x -> x ∉ index6[1:nx], index6)
    ALpvd1 = spdiagm(Nq, Nq, diag3)
    ALpvd2 = spdiagm(Nq, Nq, 1 => -diag3[1:end-1])
    ALpvd3 = spdiagm(Nq, Nq, nx+1 => diag3[1:end-nx-1])
    ALpvd4 = spdiagm(Nq, Nq, nx+2 => -diag3[1:end-nx-2])
    ALpvd = (ALpvd1 + ALpvd2 + ALpvd3 + ALpvd4)[index6, :]

    ones3 = ones(Nv)
    Spvu = spdiagm(Nv, Nv, nx => ones3[1:Nv - nx])
    Spvd = copy(Spvu)'

    ones4 = ones(Nv)
    index7 = 1:Nv+nx
    index7 = filter(x -> x ∉ index7[nx:nx:end], index7)
    Sav = (spdiagm(Nv+nx, Nv, -nx => ones4)[index7, :])'
    Sbv = (spdiagm(Nv+nx, Nv, -nx + 1 => ones4)[index7, :])'
    Scv = (spdiagm(Nv+nx, Nv, ones4)[index7,:])'
    Sdv = (spdiagm(Nv+nx, Nv, 1 => ones4[1:end-1])[index7, :])'

    advec_ops = Advection(
    AL1,
    AL2,
    index_av, 
    index_bv, 
    index_cv, 
    index_dv, 
    ALeur, 
    ALeul, 
    Seul, 
    Seur, 
    Sau, 
    Sbu, 
    Scu, 
    Sdu, 
    ALpvu, 
    ALpvd, 
    Spvu, 
    Spvd, 
    Sav, 
    Sbv, 
    Scv, 
    Sdv
    )

    return advec_ops

end

"""
    @inplacemul c = A*b

Macro to translate c = A*b with `A::SparseMatrixCSC`, `b` and `c` `Vector`s into
`SparseArrays.mul!(c,A,b,true,false)` to perform the sparse matrix - 
dense vector multiplication in-place."""
macro inplacemul(ex)
    @assert ex.head == :(=) "@inplacemul requires expression a = b*c"
    @assert ex.args[2].args[1] == :(*) "@inplacemul requires expression a = b*c"
    
    return quote
        local c = $(esc(ex.args[1]))              # output dense vector
        local A = $(esc(ex.args[2].args[2]))      # input sparse matrix
        local b = $(esc(ex.args[2].args[3]))      # input dense vector

        # c = β*c + α*A*b, with α=1, β=0 so that c = A*b
        SparseArrays.mul!(c,A,b,true,false)
    end
end