using Oceananigans
using NCDatasets
using Printf
using CairoMakie
using Oceananigans.Fields
using Oceananigans.AbstractOperations: volume
using NaNStatistics
using Makie.Colors

function get_η(ε, ν)
    η = (ν^3 ./ ε) .^ (1/4)
    return η
end

function is_kolmogorov_resolved(Δx, ε, ν)
    η = get_η(ε, ν)
    η_min = minimum(x for x in η if x > 0)
    if Δx < η_min
        return true, η_min
    else
        return false, η_min
    end
end

function get_ratio(Δx, ε, ν)
    η = get_η(ε, ν)
    η_data = filter(x -> isfinite(x) && x > 0, vec(η))
    r = Δx ./ η_data
    return r
end

function compute_statistics(Δx, ε, ν)
    r = get_ratio(Δx, ε, ν)

    r_min = minimum(r)
    r_max = maximum(r)
    r_median = median(r)
    r_mean = mean(r)
    r_95 = quantile(r, 0.95)

    frac_r_greater_than_1 = count(>(1.0), r) / length(r)

    return Dict(
        "r_min" => r_min,
        "r_max" => r_max,
        "r_median" => r_median,
        "r_mean" => r_mean,
        "r_95" => r_95,
        "frac_r_greater_than_1" => frac_r_greater_than_1
    )
end