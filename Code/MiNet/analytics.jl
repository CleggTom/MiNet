"""
    arrival_prop_funcs(B::Function,C::Function)

takes functions B and C describing the joint in/out degree distributions for consumers and resources and returns the functions to calculate the probability of arriving at a surviving node.
"""
function arrival_prop_funcs(B::Function,C::Function)
    #get required functions
    #derivatives evaluated at 1
    dBY(X) = ForwardDiff.gradient(B, [X, 1.0])[2] #Ko * P(Ki,Ko) * X^Ki
    # dBX(Y) = ForwardDiff.gradient(B, [1.0, Y])[1] #Ki * P(Ki,Ko) * Y^Ko

    dCY(X) = ForwardDiff.gradient(C, [X, 1.0])[2]
    # dCX(Y) = ForwardDiff.gradient(C, [1.0, Y])[1]

    #arrival degree - normalised by mean
    B1_I(X) = dBY(X) / dBY(1.0) #Ko * P(Ki,Ko) * X^Ki / #Ko * P(Ki,Ko)
    C1_I(X) = dCY(X) / dCY(1.0)

    #actual probability functions
    fb(c1) = B1_I(c1)
    fc(b1) = 1 - C1_I(1-b1)

    return Dict(:b => fb, :c => fc)
end

"""
    solve_arrival_probs(B,C,sb,sc)

Solve the eqations B and C to get the solutions for survivng propotions. Also accepts arguments sb and sc for the proportion of supplied resources or consumers.
"""
function solve_arrival_probs(B,C)
    f_sol = arrival_prop_funcs(B,C)
        
    b1(x) = f_sol[:b](f_sol[:c](x)) - x
    b1_sols = Roots.find_zeros(b1, 0.0, 1.0)
    c1_sols = f_sol[:c].(b1_sols)
    b0_sols = [ B([c, 1]) for c = c1_sols]
    c0_sols = [1 - C([1 - b, 1]) for b = b1_sols]

    return b0_sols, c0_sols
end

"""
    bifurcation_manifold(λc)

    get the manifold separating the cusp bifurcation as a function of λc (resource in degree).
"""
function bifurcation_manifold(λc)
    W0 = lambertw(-1 / (λc), 0)
    W1 = lambertw(-1 / (λc), -1)
    
    λb0 = -W0 / exp(1 / W0)
    λb1 = -W1 / exp(1 / W1)
    return([λb0,λb1])
end