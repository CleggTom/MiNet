"""
    arrival_prop_funcs(C::Function,M::Function)

Takes generating functions C and M describing the joint in/out degree distributions for consumers and resources and returns the functions to calculate the probability of arriving at a surviving node.
"""
function arrival_prop_funcs(C::Function,M::Function)
    #get required functions
    #derivatives evaluated at 1
    dCY(X) = ForwardDiff.gradient(C, [X, 1.0])[2] #Ko * P(Ki,Ko) * X^Ki
    dMY(X) = ForwardDiff.gradient(M, [X, 1.0])[2]

    #arrival degree - normalised by mean
    C1_I(X) = dCY(X) / dCY(1.0) #Ko * P(Ki,Ko) * X^Ki / #Ko * P(Ki,Ko)
    M1_I(X) = dMY(X) / dMY(1.0)

    #actual probability functions
    fc(m1) = M1_I(m1)
    fm(c1) = 1 - C1_I(1-c1)

    return Dict(:c => fc, :m => fm)
end

"""
    solve_arrival_probs(C,M,sc,sm)

Solve the eqations C and M to get the solutions for survivng propotions. Also accepts arguments sb and sc for the proportion of supplied resources or consumers.
"""
function solve_arrival_probs(C,M)
    f_sol = arrival_prop_funcs(C,M)
        
    c1(x) = f_sol[:c](f_sol[:m](x)) - x
    c1_sols = Roots.find_zeros(c1, 0.0, 1.0)
    m1_sols = f_sol[:m].(c1_sols)
    
    c0_sols = [ C([m, 1]) for m = m1_sols]
    m0_sols = [1 - M([1 - c, 1]) for c = c1_sols]

    return c0_sols, m0_sols
end

"""
    bifurcation_manifold(zc)

    get the manifold separating the cusp bifurcation as a function of zm (resource in degree).
"""
function bifurcation_manifold(zm)
    W0 = lambertw(-1 / (zm), 0)
    W1 = lambertw(-1 / (zm), -1)
    
    λc0 = -W0 / exp(1 / W0)
    λc1 = -W1 / exp(1 / W1)
    return([λc0,λc1])
end