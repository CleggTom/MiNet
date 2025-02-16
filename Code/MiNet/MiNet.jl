module MiNet
    using ForwardDiff
    using Roots
    using LambertW
    using Distributions
    using Random
    using Graphs
    using StatsBase

    include("./analytics.jl") #analytical results
    include("./networks.jl") #generation and simulation of networks
    include("./utils.jl") #utility functions

end