"""
    joint_sample(N,zc,zm,r)

Sample N node degrees from a joint Poisson distribution with covariance r. 
"""
function joint_sample(N::Int64,zc::Float64,zm::Float64,r::Float64)
    #inital sample
    X = rand(Poisson(zc-r),N)
    Y = rand(Poisson(zm-r),N)
    R = rand(Poisson(r),N)
    
    #sample consumer degree
    Kci = X .+ R
    Kco = Y .+ R
    
    #get target degree sums
    Sc = sum(Kci)
    Sm = sum(Kco)
    
    #get degree sum
    Sr = rand(Truncated(Poisson(N*r), 0, min(Sc, Sm)))
    Sx = Sm - Sr
    Sy = Sc - Sr 
    
    R_ = rand(Multinomial(Sr, N))
    X_ = rand(Multinomial(Sx, N))
    Y_ = rand(Multinomial(Sy, N))
    
    Kmi = X_ .+ R_
    Kmo = Y_ .+ R_
    
    return(Kci, Kco, Kmi, Kmo)
end

"""
    generate_network(N,f, p...)

Generate random network with component sizes N drawing the joint degrees from function f with parameters p.
"""
function generate_network(N::Int64, f::Function, p...)
    Kci, Kco, Kmi, Kmo = f(N,p...)
    
    # println(mean(Kbi)," ",mean(Kco))
    # println(mean(Kbo)," ",mean(Kci))
    
    @assert sum(Kci) == sum(Kmo)
    @assert sum(Kco) == sum(Kmi)

    #assign graph
    g = DiGraph(2N)
    
    #get vertex need degree sequences m -> c
    begin
        vc_seq = inverse_rle(1:N, Kci)
        vm_seq = inverse_rle((N+1):(2N), Kmo)
        
        #permutate to create random pairs
        l_n = Pair.(shuffle(vm_seq), shuffle(vc_seq))
        add_edge!.(Ref(g), l_n)
    end
    
    #get vertex produce degree sequences c -> m
    begin
        vc_seq = inverse_rle(1:N, Kco)
        vm_seq = inverse_rle((N+1):(2N), Kmi)
        
        #permutate to create random pairs
        l_p = Pair.(shuffle(vc_seq), shuffle(vm_seq))
        add_edge!.(Ref(g), l_p)
    end
    return(g)
end

"""
    update_node!(g::SimpleDiGraph{T}, s::Vector{Bool},c::Vector{Bool},i::Int) where T <: Int

Update a node i state in cross-feeding network g 
"""
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
    update_state!(g,s,c, sfix = falses(nv(g)))

Updates node states using simple rules. Consumers survive when all resources are present. Resources persist when aSy consumer makes them. sfix determines "fixed" nodes that are not updated. 
"""
function update_state!(g::SimpleDiGraph{T}, s::Vector{Bool}, s0::Vector{Bool}, c::Vector{Bool},sfix::Vector{Bool} = fill(false, nv(g))) where T <: Int
    s0 .= s 
    for i = shuffle(1:nv(g))
            if sfix[i]
                s[i] = 1
            else
                update_node!(g,s,c,i)
            end
        end
end

"""
    get_state(g::SimpleDiGraph{T}, c::Vector{Bool}, ps::Float64) where T <: Int

Initialise a network and itteratively solve for the steady state. Uses `update_state!` to update dynamics. Returns the state vector.

"""
function get_state(g::SimpleDiGraph{T}, c::Vector{Bool}, ps::Float64) where T <: Int
    #inital state
    s = fill(false, nv(g))
    s .= (rand(nv(g)) .< ps)
    s0 = fill(false,nv(g))

    #loop over time
    while !all(s .== s0)
        #update network state
        update_state!(g, s, s0, c)
    end
    
    return(s)
    
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

"""
    graphviz_bipartite(g,s,c,graph_dir, img_dir)

Visualise bipartite network
"""
function graphviz_bipartite(g,s,c,graph_dir, img_dir)
    con = "#ff1010"
    con_ext = "#ffe8e8"
    
    res = "#6495ed"
    res_ext = "#f1f5fd"
    
    #write graph
    open(graph_dir, "w") do file
        write(file, "digraph {\n")
        write(file, "layout=\"neato\";\n")
        # write(file, "overlap=\"false\";\n")
        write(file, "pack=true;\n")
        write(file, "packmode=\"array_u\";\n")
        write(file, "outputorder=\"edgesfirst\";\n")

        ind = 1:length(s)
        
        kb = 0
        kc = 0

        nB = sum(c)
        nC = length(c) - nB
        
        for n in ind  # add nodes
            if !c[n]
                char = s[n] == 1 ? res : res_ext
                posx = 10 * (kc / nC)
                posy = 0
                
                println(posx)
                println(posy)
                
                kc += 1
            else
                char = s[n] == 1 ? con : con_ext
                posx = 10 * (kb / nB)
                posy = 1
                kb += 1
            end
             write(file, "    $n [fillcolor=\"$char\",
                shape=\"circle\",
                style = \"filled\",
                width = 0.1,
                penwidth = 0,
                label=\"\",
                pos=\"$(posx),$(posy)!\"];\n")
        end
    
        for e in edges(g)  # add edges
            i=src(e); j=dst(e)

            t = c[i] == 1 ? "solid" : "dashed"
            if (s[i] == 1) && (s[j] == 1)
                write(file, "    $i -> $j [penwidth = 0.2, arrowsize=0.1, style = $(t)];\n")
            else
                write(file, "    $i -> $j [penwidth = 0.0, arrowsize=0, style = $(t)];\n")
            end
        end
    
        write(file, "}")
    end
    
    run(pipeline(`dot -Tpdf $graph_dir`, stdout=img_dir))
end

