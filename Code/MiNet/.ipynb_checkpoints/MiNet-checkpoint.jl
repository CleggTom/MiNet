module MiNet
    using ForwardDiff
    using Roots
    using LambertW
    using Distributions
    using Random
    using Graphs
    using StatsBase

    include("./analytics.jl")
    include("./networks.jl")
    # include("./dynamics.jl")
    include("./utils.jl")
    include("./powerlaw.jl")

end