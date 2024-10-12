mutable struct MiCom
    g::SimpleDiGraph
    c::Vector{Bool}
    s::Vector{Bool}
end

"""
    update_dynamic!(g::SimpleDiGraph{T}, s::Vector{Bool}, c::Vector{Bool},sfix::Vector{Bool} = fill(false, nv(g))) where T <: Int

update the dynamics of a community object. Does not flip persistence for consumers. 
"""
function update_dynamic!(g::SimpleDiGraph{T}, s::Vector{Bool}, c::Vector{Bool},sfix::Vector{Bool} = fill(false, nv(g))) where T <: Int
    ds = similar(s)
    for i = 1:nv(g)
            if sfix[i]
                ds[i] = 1
            else
                if c[i]
                    # println("con") 
                    #only set true if already present
                    @views ds[i] = s[i] && all(s[g.badjlist[i]])
                else
                    # println("res")
                    @views ds[i] = any(s[g.badjlist[i]])
                end
            end
        end
    s .= ds
end

update_dynamic!(com::MiCom, sfix::Vector{Bool} = fill(false, nv(com.g))) = update_dynamic!(com.g,com.s,com.c,sfix) 