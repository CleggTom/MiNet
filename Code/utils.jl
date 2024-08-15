"""
    get_p(x)

convert sample vector into relative probabilites. 
"""
function get_p(x)
    [sum(x .== u) / length(x) for u = 0:maximum(x)]
end