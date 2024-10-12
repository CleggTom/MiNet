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

function calc_params(zx,zy,ρ)
    r = ρ*sqrt(zx*zy)
    zx_ = zx 
    zy_ = zy
    return(zx_,zy_,r)
end

#params
n_l = 50
l_vec = range(0.01, 10.0, length = n_l)

# n_l_sim = 100
# l_vec_sim = range(0.1, 10.0, length = n_l_sim)

n_cor = 6
cor_vec = range(0,1.0, length = n_cor) 

ρ = 0.0
N = 5000
n_rep = 20
p_vec = [4.0]


####
#phaseplot
####
println("phase plot")
#bivarate poisson PGF
F(X,Y,λ) = exp((λ[1]-λ[3])*(X − 1) + (λ[2]-λ[3])*(Y − 1) + λ[3]*(X*Y − 1))

#generate params list
p_mat = [(calc_params(zn,zp,ρ), calc_params(zp,zn,ρ)) for zn = l_vec, zp = l_vec, ρ = cor_vec]
feas = [p[3] <= min(p[1],p[2]) for p = map(x -> x[1], p_mat)]


b0_sols = Array{Vector,3}(undef,n_l, n_l, n_cor)
c0_sols = similar(b0_sols)

k = [0]
Threads.@threads for i = 1:n_l
    for j = 1:n_l
        for p = 1:n_cor
            k[1] += 1
            if k[1] % 100 == 0
                print("\r", k)
            end

            if feas[i,j,p]
                B(x) = F(x[1],x[2], p_mat[i,j,p][1])
                C(x) = F(x[1],x[2], p_mat[i,j,p][2])
        
                b0,c0 = MiNet.solve_arrival_probs(B,C)
        
                b0_sols[i,j,p] = b0 
                c0_sols[i,j,p] = c0 
            else
                b0_sols[i,j,p] = [] 
                c0_sols[i,j,p] = []    
            end
        end
    end
end

# ###network simulations
println("")
println("networks plot")
Random.seed!(1)

#allocate result matricies
b0_sim_sols = Array{Vector,3}(undef,n_l, n_l, n_cor)
c0_sim_sols = similar(b0_sols)

#across need degree variation
k = [0]
Threads.@threads for i = 1:n_l
    for j = 1:n_l
        for p = 1:n_cor
            k[1] += 1
            if k[1] % 100 == 0
                print("\r", k)
            end
            if feas[i,j,p]
                reps_b = zeros(n_rep)
                reps_c = zeros(n_rep)
                for r = 1:n_rep
                    c = vcat(fill(true, N), fill(false,N))
                    g = MiNet.generate_network(N,MiNet.joint_sample, p_mat[i,j,p][1]...);
                    is = rand()
                    s = MiNet.get_state(g, c, is)
                    reps_b[r] = sum(s[1:N]) / N
                    reps_c[r] = sum(s[(N+1):end]) / N
                end
                b0_sim_sols[i,j,p] = reps_b
                c0_sim_sols[i,j,p] = reps_c
            else
                b0_sim_sols[i,j,p] = []
                c0_sim_sols[i,j,p] = []    
            end        
        end
    end
end


save("./Results/JLD2/fig_cor.jld2", 
        Dict("phase" => (b0_sols,c0_sols), 
             "sim" => (b0_sim_sols,c0_sim_sols)) 
    )