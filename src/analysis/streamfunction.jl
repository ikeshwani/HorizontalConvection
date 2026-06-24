# streamfunction.jl
#
# Overturning streamfunction diagnostics.
#   - get_ψ      : ψ(x, z, t) in physical space, ψ = -∫∫ u dy dz (cumulative in z)
#   - get_ψb_sort: ψ(x, b, t) in buoyancy space via a sort + cumsum sweep
#
# Physics only — no plotting.

using NCDatasets
using NaNStatistics
using Printf

export get_ψ, get_ψb_sort

# ---- ψ(x, z, t) overturning streamfunction in physical space ----
#
# ψ(x, z, t) = -∫_{-H}^{z} (∫ u dy) dz'.  Non-mutating: u_all is copied before
# masking immersed (u == 0) cells so the caller's array is untouched.
function get_ψ(u_all, Δy_vec, Δz_vec, Nx, Nz, Nt)
    u_work = copy(u_all)
    u_work[u_work .== 0] .= NaN
    ∫udy   = dropdims(nansum(u_work .* reshape(Δy_vec, 1, :, 1, 1), dims=2), dims=2)
    ∫udy_w = ∫udy .* reshape(Δz_vec, 1, Nz, 1)
    ∫udy_w[isnan.(∫udy_w)] .= 0
    return Float32.(-cumsum(∫udy_w, dims=2))  # [Nx, Nz, Nt]
end

# Convenience method: read u and the grid metadata straight from a NetCDF dataset
# and delegate to the array method above.
function get_ψ(ds)
    Nx, Ny, Nz = ds.attrib["Nx"], ds.attrib["Ny"], ds.attrib["Nz"]
    Δy_vec = ds["Δy_aca"][:]
    Δz_vec = ds["Δz_aac"][:]
    Nt     = length(ds["time"][:])
    u_all  = Array(ds["u"][1:Nx, :, :, :])
    return get_ψ(u_all, Δy_vec, Δz_vec, Nx, Nz, Nt)
end

# ---- ψ(x, b, t) via sort + cumsum ----
#
# For each x-column at time t, ψ(x, b₀, t) = -∫∫_{b < b₀} u dy dz.
# Instead of rebuilding a boolean mask for each of the n_b_bins buoyancy
# levels (the old approach), we:
#   1. sort the Ny*Nz cells in the column by their buoyancy value,
#   2. compute a cumulative sum of u·Δy·Δz in that sorted order,
#   3. sweep a two-pointer through the sorted b array once to evaluate all
#      bins in O(Ny·Nz + n_b_bins) rather than O(n_b_bins · Ny · Nz).
#
function get_ψb_sort(b_all, u_all, Δy_vec, Δz_vec, Nx, Ny, Nz, Nt;
                     b_range=(-1.0, 1.0), n_b_bins=501)

    b_bins = collect(range(b_range[1], b_range[2], length=n_b_bins))
    W      = reshape(Δy_vec, Ny, 1) .* reshape(Δz_vec, 1, Nz)  # [Ny, Nz]

    ψ_b = zeros(Float32, Nx, n_b_bins, Nt)

    for t in 1:Nt
        for i in 1:Nx
            b_col  = vec(b_all[i, :, :, t])
            uw_col = vec(u_all[i, :, :, t] .* W)

            perm     = sortperm(b_col)
            b_sorted = b_col[perm]
            cum_uw   = cumsum(uw_col[perm])
            n_col    = length(b_sorted)

            # two-pointer sweep: b_bins is already sorted so j only advances
            j = 0
            for k in 1:n_b_bins
                while j < n_col && b_sorted[j + 1] < b_bins[k]
                    j += 1
                end
                ψ_b[i, k, t] = j > 0 ? Float32(-cum_uw[j]) : 0.0f0
            end
        end
        t % 50 == 0 && @printf("  t = %d / %d\n", t, Nt)
    end

    return ψ_b, b_bins
end
