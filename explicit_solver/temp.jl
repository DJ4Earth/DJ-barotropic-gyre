using Printf

function find_non_zeros(A)

    for j = 1:size(A)[1]
        for k = 1:size(A)[2]

            if A[j,k] != 0.0
                @printf "(%i, %i)" j k
            end

        end
    end

end

function find_values(A, val)

    indices = []

    for j = 1:size(A)[1]

        if abs(A[j]) > val
            #@printf "A[%i] = %e" j A[j]
            push!(indices, j)
        end
    end

    return indices

end

mutable struct my_struct 
    one::Vector{Float64}
    two::Vector{Float64}
    three::Vector{Float64}
end