using Pkg

Pkg.activate(".")

using LinearAlgebra
using Distributions
using Random
using StatsBase


using ForwardDiff
using LambertW
using Roots 

using Graphs
# using CairoMakie
# using GraphMakie

using DelimitedFiles
using JLD2

include("../Code/MiNet/MiNet.jl")

##define PGFs
A(X, s) = (1-s) + s * X # keep proportion s
F(X,Y,λ) = exp(λ[1]*(X − 1) + λ[2]*(Y − 1) + λ[3]*(X*Y − 1))

#Parameters
λc = 10.0
λm = 10.0

#generate p_list
n_s = 100
s_vec = range(0.0, 0.99, length = n_s)

#network simulations
#network size
n_s_network = 25
s_vec_network = range(0.0, 0.99, length = n_s_network)

N = 10000 #system size
n_r = 100 #number of system replicates
n_rep = 1 #numer of simulation replicates

#analytical
c0_s_sols = Matrix{Vector}(undef, n_s, n_s)
m0_s_sols = similar(c0_s_sols)

k = [0]
Threads.@threads for i = 1:n_s
    for j = 1:n_s
        
        k[1] += 1
        print("\r", k)

        #keep s species
        s = s_vec[i]
        #supply r resources - keep proportion 1 -r
        r = s_vec[j]

        C(x) = F(A(x[1], 1-r), 1.0,  [λc,λm,0.0]) 
        M(x) = F(A(x[1], s), 1.0,  [λm,λc,0.0]) 

        c1(x) = C(1-M(1-x)) - x
        c0 = Roots.find_zeros(c1, 0.0, 1.0)
        m0 = M.(c0)
        
        
        c0_s_sols[i,j] = s .* c0 
        m0_s_sols[i,j] = r .+ (1 .- r) .* m0
    
    end
end


# #networks
c0_sim_sols = Array{Float64,4}(undef, n_r, n_s_network, n_s_network, n_rep)
m0_sim_sols = similar(c0_sim_sols)


#average degree: need produce covariance
for r = 1:n_r
    #generate network
    #create network and set initial state
    g = MiNet.generate_network(N,MiNet.joint_sample,λc,λm,0.0);
    c = fill(false, 2N)
    c[1:N] .= true
    s = fill(true, 2N);
    k = [0]
    #loop over consumer removals
     Threads.@threads for i = 1:n_s_network
        k[1] += 1
        print("\r ",r," ", k)
        for j = 1:n_s_network
            #take copies
            g_copy = deepcopy(g)
            c_copy = deepcopy(c)
            
            #indicies to remove
            c_rm = sample(1:N, Int(floor(s_vec_network[i]*N)), replace = false)
            m_rm = sample(N+1:2N, Int(floor(s_vec_network[j]*N)), replace = false)
            
            to_rm = vcat(c_rm, m_rm)
            to_keep = filter(x -> x ∉ to_rm, 1:2N)
    
            rem_vertices!(g_copy, to_rm, keep_order = true)
            c_copy = c_copy[to_keep]
            
            #simulate
            for k = 1:n_rep
                s_copy = MiNet.get_state(g_copy, c_copy, rand())
                
                c0_sim_sols[r,i,j,k] = sum(s_copy[c_copy]) / N
                m0_sim_sols[r,i,j,k] = sum(s_copy[.!c_copy]) / N
            end
        end
    end
end


save("./Results/JLD2/fig_2.jld2", Dict("pred" => (c0_s_sols,m0_s_sols), "sim" => (c0_sim_sols,m0_sim_sols)))