function H(N,s)
    sum([1 / k^(s + 1) for k = 1:N])
end

function zipf_p(N,s)
    [1 / (k^(s+1) * H(N,s)) for k = 1:N]
end