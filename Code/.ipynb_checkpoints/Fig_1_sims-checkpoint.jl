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
using CairoMakie
using GraphMakie

using DelimitedFiles
using JLD2

include("../Code/MiNet/MiNet.jl")

#params
n_l = 100
l_vec = range(exp(1), 10.0, length = n_l)

n_l_sim = 100
l_vec_sim = range(0.1, 10.0, length = n_l_sim)

ρ = 0.0
N = 10000
p_vec = [4.0]


####
#phaseplot
####
println("phase plot")
#bivarate poisson PGF
F(X,Y,λ) = exp(λ[1]*(X − 1) + λ[2]*(Y − 1) + λ[3]*(X*Y − 1))

#generate params list
p_mat = [([λn,λp, 0.0], [λp,λn, 0.0]) for λn = l_vec, λp = l_vec]

b0_sols = Matrix{Vector}(undef,n_l, n_l)
c0_sols = similar(b0_sols)

k = [0]
Threads.@threads for i = 1:n_l
    for j = 1:n_l
        k[1] += 1
        if k[1] % 100 == 0
            print("\r", k)
        end
        
        B(x) = F(x[1],x[2], p_mat[i,j][1])
        C(x) = F(x[1],x[2], p_mat[i,j][2])

        b0,c0 = MiNet.solve_arrival_probs(B,C)

        b0_sols[i,j] = b0 
        c0_sols[i,j] = c0 
    end
end

###network simulations
println("")
println("networks plot")
Random.seed!(1)

#allocate result matricies
sim_mat  = Array{Float64,3}(undef, n_l_sim, length(p_vec), 2)

#across need degree variation
k = [0]
Threads.@threads for l = 1:n_l_sim
    k[1] += 1
    print("\r", k)
    λn = l_vec_sim[l]
    c = vcat(fill(true, N), fill(false,N))
    
    #get simulated proportions persisting
    for i = 1:length(p_vec)
        λp = p_vec[i]
        #make network
        g = MiNet.generate_network(N,MiNet.joint_sample,λn, λp, 0.0);
        is = rand()
        s = MiNet.get_state(g, c, is)

        sim_mat[l,i,1] = sum(s[1:N]) / N
        sim_mat[l,i,2] = sum(s[(N+1):end]) / N
    end
end

##predictions
println("")
println("predictions plot")

pred_mat = Array{Vector{Float64},3}(undef, n_l_sim, length(p_vec), 2)

F(X,Y,λ) = exp(λ[1]*(X − 1) + λ[2]*(Y − 1) + λ[3]*(X*Y − 1))

# #predictions
Threads.@threads for l = 1:n_l_sim
    for i = 1:length(p_vec)
        λn = l_vec_sim[l]
        λp = p_vec[i]
        
        B(x) = F(x[1],x[2], [λn,λp,0.0])
        C(x) = F(x[1],x[2], [λp,λn,0.0])
    
        pred_vals = 0
        pred_vals = MiNet.solve_arrival_probs(B,C)
      
        pred_mat[l,i,1] = pred_vals[1]
        pred_mat[l,i,2] = pred_vals[2]
    end
end

save("./Results/JLD2/fig_1.jld2", Dict("phase" => (b0_sols,c0_sols), "sim" => sim_mat, "pred" => pred_mat))