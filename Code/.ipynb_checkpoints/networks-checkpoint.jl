"""
    joint_sample(N,λn,λp,λr)

Sample N node degrees from a joint Poisson distribution with covariance λr. 
"""
function joint_sample(N::Int64,λn::Float64,λp::Float64,λr::Float64)
    #define distributions need + prod
    dK = [Poisson(λn + λr), Poisson(λp + λr)]
    #sample initial
    #need
    Kbi = rand(dK[1], N)
    Kco = rand(dK[1], N)
    #prod
    Kbo = rand(dK[2], N)
    Kci = rand(dK[2], N)

    #while loop
    n_eq, p_eq = 1,1
    
    while (n_eq != 0) || (p_eq != 0)
        if n_eq != 0
            #need
            Kbi[rand(1:N)] = rand(dK[1])
            Kco[rand(1:N)] = rand(dK[1])
        end

        if p_eq != 0
            #prod
            Kbo[rand(1:N)] = rand(dK[2])
            Kci[rand(1:N)] = rand(dK[2])
        end
        #eval
        n_eq = sum(Kbi) - sum(Kco)
        p_eq = sum(Kbo) - sum(Kci)
    end
    
    return(Kbi, Kbo, Kci, Kco)
end

"""
    joint_sample_cheat(N,λn,λp,λr)

Sample N node degrees from a joint Poisson distribution with covariance λr. 'cheats' by shuffling the sample. Should be ok with large N but not good for small networks. 
"""
function joint_sample_cheat(N::Int64,λn::Float64,λp::Float64,λr::Float64)
    #define distributions
    dK = [Poisson(λn + λr), 
          Poisson(λp + λr)]
    
    #sample need
    Kbi = rand(dK[1], N)
    Kco = shuffle(Kbi)
    
    #sample prod
    Kbo = rand(dK[2], N)
    Kci = shuffle(Kbo)
    
    return(Kbi, Kbo, Kci, Kco)
end

"""
    generate_network(N,λn,λp,λr)

Generate random network with component sizes N from given rates.
"""
function generate_network(N::Int64,λn::Float64,λp::Float64,λr::Float64)
    Kbi, Kbo, Kci, Kco = joint_sample_cheat(N,λn,λp,λr)
    
    # println(mean(Kbi)," ",mean(Kco))
    # println(mean(Kbo)," ",mean(Kci))
    
    @assert sum(Kbi) == sum(Kco)
    @assert sum(Kbo) == sum(Kci)

    #assign graph
    g = DiGraph(2N)
    
    #get vertex need degree sequences
    begin
        vb_seq = inverse_rle(1:N, Kbi)
        vc_seq = inverse_rle((N+1):(2N), Kco)
        
        #permutate to create random pairs
        l_n = Pair.(shuffle(vc_seq), shuffle(vb_seq))
        add_edge!.(Ref(g), l_n)
    end
    
    #get vertex produce degree sequences
    begin
        vb_seq = inverse_rle(1:N, Kbo)
        vc_seq = inverse_rle((N+1):(2N), Kci)
        
        #permutate to create random pairs
        l_p = Pair.(shuffle(vb_seq), shuffle(vc_seq))
        add_edge!.(Ref(g), l_p)
    end
    return(g)
end

function update_node!(g::SimpleDiGraph{T}, s::Vector{Bool},c::Vector{Bool},i::Int) where T <: Int
    if c[i]
        # println("con")
           s[i] = all(s[g.badjlist[i]])
    else
        # println("res")
            s[i] = any(s[g.badjlist[i]])
    end
end

"""
    update_state!(g,s,c,sfix = falses(nv(g)))

Updates node states using simple rules. Consumers survive when all resources are present. Resources persist when any consumer makes them. sfix determines "fixed" nodes that are not updated. 
"""
function update_state!(g::SimpleDiGraph{T}, s::Vector{Bool}, c::Vector{Bool},sfix::Vector{Bool} = fill(false, nv(g))) where T <: Int
    for i = shuffle(1:nv(g))
            if sfix[i]
                ds[i] = 1
            else
                if c[i]
                    # println("con")
                    @views s[i] = all(s[g.badjlist[i]]) 
                else
                    # println("res")
                    @views s[i] = any(s[g.badjlist[i]])
                end
            end
        end
end

"""
    get_state(g,N,sb = 0.0,sc = 0.0; Nt = 100, ts = false)

Initialise a network and itteratively solve for the steady state. Uses `update_state!` to update dynamics. Returns the state vector and the timeseries if `ts = true`

TODO: add supplied portion
"""
function get_state(g::SimpleDiGraph{T}, c::Vector{Bool}, ps::Float64; Nt = 100, ts = false) where T <: Int
    #inital state
    s = fill(false, nv(g))
    s .= (rand(nv(g)) .< ps)
    # sfix = deepcopy(s)
    if ts
        res = zeros(2,Nt)
    end
    for i = 1:Nt
        if ts
            res[1,i] = mean(s[c])
            res[2,i] = mean(s[.!c])
        end
        update_state!(g, s, c)
        
    end

    if ts
        return(s, res)
    else
        return(s)
    end
end

"""
    function graphviz(g,s,N,graph_dir, img_dir)

generate graph of MiNet from
    g: graph object
    s: node states
    N: component size
    graph_dir: dir to save graph.dot file
    img_dir: dir to save pdf of network
"""
function graphviz(g,s,c,graph_dir, img_dir)
    con = "#ff1010"
    con_ext = "#ffe8e8"
    
    res = "#6495ed"
    res_ext = "#f1f5fd"
    
    #write graph
    open(graph_dir, "w") do file
        write(file, "digraph {\n")
        write(file, "layout=\"sfdp\";\n")
        write(file, "overlap=\"false\";\n")
        write(file, "pack=true;\n")
        write(file, "packmode=\"array_u\";\n")
        write(file, "outputorder=\"edgesfirst\";\n")
        for n in vertices(g)  # add nodes
            if !c[n]
                char = s[n] == 1 ? res : res_ext
            else
                char = s[n] == 1 ? con : con_ext
            end
             write(file, "    $n [fillcolor=\"$char\",
                shape=\"circle\",
                style = \"filled\",
                penwidth = 0,
                label=\"\"];\n")
        end
    
        for e in edges(g)  # add edges
            i=src(e); j=dst(e)
            
            if s[i] == 1
                write(file, "    $i -> $j [penwidth = 1.0, arrowsize=1];\n")
            else
                write(file, "    $i -> $j [penwidth = 0.1, arrowsize=0];\n")
            end
        end
    
        write(file, "}")
    end
    
    run(pipeline(`dot -Tpdf $graph_dir`, stdout=img_dir))
end

