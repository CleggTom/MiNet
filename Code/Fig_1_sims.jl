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

#params
n_z = 100
z_vec = range(exp(1), 10.0, length = n_z)

n_z_sim = 100
z_vec_sim = range(0.1, 10.0, length = n_z_sim)

ρ = 0.0
N = 10000
p_vec = [2.0,4.0]


####
#phaseplot
####
println("phase plot")
#bivarate poisson PGF
F(X,Y,z) = exp(z[1]*(X − 1) + z[2]*(Y − 1) + z[3]*(X*Y − 1))

#generate params list
p_mat = [([zc,zm, 0.0], [zm,zc, 0.0]) for zc = z_vec, zm = z_vec]

c0_sols = Matrix{Vector}(undef,n_z, n_z)
m0_sols = similar(c0_sols)

k = [0]
Threads.@threads for i = 1:n_z
    for j = 1:n_z
        k[1] += 1
        if k[1] % 100 == 0
            print("\r", k)
        end
        
        C(x) = F(x[1],x[2], p_mat[i,j][1])
        M(x) = F(x[1],x[2], p_mat[i,j][2])

        c0,m0 = MiNet.solve_arrival_probs(C, M)

        c0_sols[i,j] = c0 
        m0_sols[i,j] = m0 
    end
end

###network simulations
println("")
println("networks plot")
Random.seed!(1)

#allocate result matricies
sim_mat  = Array{Float64,3}(undef, n_z_sim, length(p_vec), 2)

#across need degree variation
k = [0]
Threads.@threads for l = 1:n_z_sim
    k[1] += 1
    print("\r", k)
    zc = z_vec_sim[l]
    c = vcat(fill(true, N), fill(false,N))
    
    #get simulated proportions persisting
    for i = 1:length(p_vec)
        zm = p_vec[i]
        #make network
        g = MiNet.generate_network(N,MiNet.joint_sample, zc, zm, 0.0);
        is = rand()
        s = MiNet.get_state(g, c, is)

        sim_mat[l,i,1] = sum(s[1:N]) / N
        sim_mat[l,i,2] = sum(s[(N+1):end]) / N
    end
end

##predictions
println("")
println("predictions plot")

pred_mat = Array{Vector{Float64},3}(undef, n_z_sim, length(p_vec), 2)

F(X,Y,λ) = exp(λ[1]*(X − 1) + λ[2]*(Y − 1) + λ[3]*(X*Y − 1))

# #predictions
Threads.@threads for l = 1:n_z_sim
    for i = 1:length(p_vec)
        zc = z_vec_sim[l]
        zm = p_vec[i]
        
        C(x) = exp(-zc * (1 - x))
        M(x) = exp(-zm * (1 - x))
    
        f(x) = C(1 - M(1-x))-x
        sol = find_zeros(f, 0, 1)
      
        pred_mat[l,i,1] = sol
        pred_mat[l,i,2] = M.(1 .- sol)
    end
end

save("./Results/JLD2/fig_1.jld2", Dict("phase" => (c0_sols,m0_sols), "sim" => sim_mat, "pred" => pred_mat))