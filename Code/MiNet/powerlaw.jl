function H(N,s)
    sum([1 / k^s for k = 1:N])
end

function zipf_p(N,s)
    [1 / (k^s * H(N,s)) for k = 1:N]
end