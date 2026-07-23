# gmix.jl
#
# Mixing-rate-in-buoyancy-coordinates diagnostics, G_mix(b).
#
# Two independent estimators are provided — they compute the same quantity by
# different routes and can disagree, so both are kept for comparison:
#   - G_mix_calc    : sorted-binning of cumulative χ·dV, then 0.5·d²/db².
#   - G_mix_calc_v2 : product-rule decomposition of d²/db²[V(b)·χ̄(b)].
#
# Physics only — no plotting.

export gaussian_smooth, G_mix_calc, G_mix_calc_v2, G_mix_calc_v3
export diffusive_flux_convergence, blended_b_edges

# 1-D Gaussian smoothing with a ±3σ truncated kernel.
function gaussian_smooth(x::Vector, σ::Real)
    n           = length(x) # number of points we are smoothing over
    kernel_half = ceil(Int, 3σ) # how far out to reach when averaging around each point using the fact that 99.7% of area is within 3σ
    out         = similar(x, Float64) #preallocating output array

    for i in 1:n # i is the point we are currently computing over (center of current window)
        wsum = 0.0; vsum = 0.0 #wsum = sum of weights, vsum = sum of weight x value 
        for j in max(1, i - kernel_half):min(n, i + kernel_half) #j is the neighbor we're looping over so (i-j) is how far that neighbor is from center
            w     = exp(-0.5 * ((i - j) / σ)^2) #gaussian weight for neighbor j 

            wsum += w #accumulate the weight
            vsum += w * x[j] #vsum is the weighted value of our array of all neighbors
        end
        out[i] = vsum / wsum # the smoothed value is the weighted average vsum/wsum, normalizing = dividing weighted sum by total weight
    end
    return out
end

# G_mix via sorted binning of the cumulative χ·dV integral.
#
# Sort cells by buoyancy, accumulate χ·dV, evaluate the cumulative integral at
# each buoyancy bin edge, smooth, then take 0.5·d²/db².
function G_mix_calc(b_region::Vector, χdV_region::Vector, b_range; n_b_bins=500)
    b_min, b_max = b_range
    b_bins = range(b_min, b_max, length=n_b_bins)

    perm     = sortperm(b_region)
    b_sorted = b_region[perm]
    cum_χdV  = cumsum(χdV_region[perm])

    integral_vals = zeros(n_b_bins)
    for (i, b_0) in enumerate(b_bins)
        idx = searchsortedlast(b_sorted, b_0)
        integral_vals[i] = idx > 0 ? cum_χdV[idx] : 0.0
    end
    
    db    = step(b_bins)
    G_mix = 0.5 .* diff(diff(integral_vals)) ./ db^2
    b_out = collect(b_bins)[2:end-1]
    return b_out, G_mix
end

# G_mix via product rule: d²/db²[V(b)·χ̄(b)] = V·χ̄'' + 2·V'·χ̄' + V''·χ̄.
function G_mix_calc_v2(b_region::Vector, χdV_region::Vector, dV_region::Vector, b_range;
                       n_b_bins=500, σ=10)
    b_min, b_max = b_range
    b_bins = range(b_min, b_max, length=n_b_bins)
    db = step(b_bins)

    perm     = sortperm(b_region)
    b_sorted = b_region[perm]
    cum_χdV  = cumsum(χdV_region[perm])
    cum_dV   = cumsum(dV_region[perm])

    χ̄ = cum_χdV ./ cum_dV

    χ̄_binned = zeros(n_b_bins)
    V_binned  = zeros(n_b_bins)
    for (i, b_0) in enumerate(b_bins)
        idx = searchsortedlast(b_sorted, b_0)
        χ̄_binned[i] = idx > 0 ? χ̄[idx] : 0.0
        V_binned[i]  = idx > 0 ? cum_dV[idx] : 0.0
    end

    term1 = 0.5 .* V_binned[2:end-1] .* diff(diff(χ̄_binned)) ./ db^2
    term2 = 0.5 .* χ̄_binned[2:end-1] .* diff(diff(V_binned)) ./ db^2
    term3 = (diff(V_binned) .* diff(χ̄_binned)) ./ db^2

    G_mix = term1 .+ term2 .+ term3[1:end-1]
    return gaussian_smooth(G_mix, σ)
end


# ---------------------------------------------------------------------------
# Non-uniform buoyancy axis
# ---------------------------------------------------------------------------

# Bin edges placed uniformly in a *blended* coordinate,
#
#     φ(b) = (1-λ)·(b - b_min)/(b_max - b_min)  +  λ·F(b)
#             \______ buoyancy ruler ______/       \_ volume ruler _/
#
# where F(b) is the fraction of volume with buoyancy below b (the exact volume CDF,
# evaluated at every cell — not binned).  λ=0 reproduces a uniform axis; λ=1 gives
# pure equal-volume (quantile) bins.
#
# Why blend rather than take either extreme.  A uniform axis puts the same bin width
# in the well-mixed layer (where cells sample b almost continuously) as in a quiescent
# basin's stratified tail, where b ≈ b(z) and adjacent z-levels can sit further apart
# in b than one bin is wide — so bins land between levels, the cumulative sums used by
# G_mix go flat, and G_mix reads exactly 0 there ("holes").  Pure quantile bins cure
# that but collapse the sparse tail into one enormous bin.  Blending widens bins where
# volume is thin (killing the holes) while still spending resolution where the data
# supports it.
#
# λ is run-specific: it depends on how concentrated the volume distribution is, so
# re-tune it (see scripts/analysis_scripts/gmix_bin_tuning.jl) for a new Ra or grid.
#
# `b_vals` and `vol_vals` are flat vectors over the cells defining the axis (normally
# every wet cell of one quasi-steady snapshot).  Returns `n_edges` sorted edges; ties
# are dropped, so the result can be shorter where the distribution is degenerate.
function blended_b_edges(b_vals::AbstractVector, vol_vals::AbstractVector, n_edges::Integer;
                         λ=0.7)
    0 <= λ <= 1 || throw(ArgumentError("λ must be in [0, 1], got $λ"))
    n_edges >= 2 || throw(ArgumentError("n_edges must be ≥ 2, got $n_edges"))
    length(b_vals) == length(vol_vals) ||
        throw(DimensionMismatch("b_vals and vol_vals must have equal length"))

    perm     = sortperm(b_vals)
    b_sorted = b_vals[perm]
    b_min, b_max = b_sorted[1], b_sorted[end]
    b_max > b_min || throw(ArgumentError("b_vals are constant; cannot build an axis"))

    F = cumsum(vol_vals[perm])
    F ./= F[end]                                    # volume ruler, 0 → 1
    n = (b_sorted .- b_min) ./ (b_max - b_min)      # buoyancy ruler, 0 → 1
    φ = (1 - λ) .* n .+ λ .* F                      # blended ruler, monotone 0 → 1

    edges = [b_sorted[clamp(searchsortedfirst(φ, q), 1, length(b_sorted))]
             for q in range(0, 1, length=n_edges)]
    return unique(edges)
end


