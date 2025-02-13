"""
    get_p(x)

convert sample vector into relative probabilites. 
"""
function get_p(x)
    [sum(x .== u) / length(x) for u = 0:maximum(x)]
end

function get_p(x,y)
    hcat([[sum((x .== i) .&& (y .== j)) / length(x) for i = 0:maximum(x)] for j = 0:maximum(y)]...)
end

function empirical_PGF(x::Float64, p)
     v = 0
    for k = eachindex(p)
        v += x[1]^(k) * p[k]
    end
    return(v)
end

function empirical_PGF(x::Vector, p)
    v = 0
    for i = size(p)[1]
        for j = size(p)[2]
            v += x[1]^(i-1) * x[2]^(j-1) * p[i,j]
        end
    end
    return(v)
end


"""

plots the bifurcation diagram in 1d. 
"""
function plot_bifurcation!(ax, xvec, yvec; args...)
    if any(length.(yvec) .== 3)
        b_order = 0
        #if not all in the multistable
        if !all(length(yvec) .== 3)
            b_order = minimum(yvec[end]) < maximum(yvec[1]) ? :hl : :lh 
        else
            b_order = yvec[end][2] < yvec[1][2] ? :hl : :lh
        end
        
        #plot branches
        if b_order == :hl
            inde = findlast(length.(yvec) .== 3)
            lines!(ax, xvec[1:inde], maximum.(yvec[1:inde]); args...)
            
            inds = findfirst(length.(yvec) .== 3)
            lines!(ax, xvec[inds: end], minimum.(yvec[inds:end]); args...)
        
            lines!(ax, xvec[inds:(inde)], [y[2] for y = yvec[inds: (inde) ]]; linestyle = :dash, args...)
        elseif b_order == :lh
             inde = findlast(length.(yvec) .== 3)
            lines!(ax, xvec[1:inde], minimum.(yvec[1:inde]); args...)
            
            inds = findfirst(length.(yvec) .== 3)
            lines!(ax, xvec[inds: end], maximum.(yvec[inds:end]); args...)
        
            lines!(ax, xvec[inds:(inde)], [y[2] for y = yvec[inds: (inde) ]]; linestyle = :dash, args...)
        end
    else
        lines!(ax, xvec, [y[1] for y = yvec]; args...)
    end
end