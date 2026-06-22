using NCDatasets
using Printf
using NaNStatistics

# ---- paths ----
data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/Control/RA1e8/4x_stretch/512_128/"
outfile  = joinpath(data_dir, "psi_b_Control_RA1e8_seg1to12.nc")

segments = 1:12

# ---- load grid info from seg1 ----
println("loading grid info from seg1...")
ds1    = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
x      = ds1["x_caa"][:]
y      = ds1["y_aca"][:]
z      = ds1["z_aac"][:]
Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
Δy_vec = ds1["Δy_aca"][:]
Δz_vec = ds1["Δz_aac"][:]
close(ds1)

# ---- load segments, deduplicating overlapping time steps ----
println("loading b and u from segments $(segments)...")
b_segs    = Vector{Array{Float32,4}}()
u_segs    = Vector{Array{Float32,4}}()
time_segs = Vector{Vector{Float64}}()

let t_last = -Inf
    for s in segments
        bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
        vfile = NCDataset(joinpath(data_dir, "velocities_seg$(s).nc"))

        t_seg = bfile["time"][:]
        valid = findall(t_seg .> t_last)

        if isempty(valid)
            @printf("  seg %d: all %d steps are duplicates — skipping\n", s, length(t_seg))
            close(bfile); close(vfile)
            continue
        end

        n_skip = valid[1] - 1
        n_skip > 0 && @printf("  seg %d: skipping first %d overlapping step(s)\n", s, n_skip)

        n_v     = size(vfile["u"], 4)
        t_range = valid[1]:min(valid[end], n_v)
        push!(b_segs,    Array(bfile["b"][:, :, :, t_range]))
        push!(u_segs,    Array(vfile["u"][1:Nx, :, :, t_range]))
        push!(time_segs, t_seg[t_range])

        t_last = t_seg[valid[end]]
        close(bfile); close(vfile)
        @printf("  seg %d: loaded %d steps (t = %.2f → %.2f)\n", s, length(t_range), t_seg[valid[1]], t_last)
    end
end

b_all = cat(b_segs...; dims=4)
u_all = cat(u_segs...; dims=4)
time  = vcat(time_segs...)
Nt    = length(time)
println("total time steps: $Nt  (t = $(time[1]) → $(time[end]))")

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

# ---- ψ(x, z, t) overturning streamfunction in physical space ----
function get_ψ(u_all, Δy_vec, Δz_vec, Nx, Nz, Nt)
    u_work = copy(u_all)
    u_work[u_work .== 0] .= NaN
    ∫udy   = dropdims(nansum(u_work .* reshape(Δy_vec, 1, :, 1, 1), dims=2), dims=2)
    ∫udy_w = ∫udy .* reshape(Δz_vec, 1, Nz, 1)
    ∫udy_w[isnan.(∫udy_w)] .= 0
    return Float32.(-cumsum(∫udy_w, dims=2))
end

# ---- compute ----
b_range  = (-1.0, 1.0)
n_b_bins = 501

println("computing ψ(x, b, t) with sort method...")
ψ_b, b_bins = get_ψb_sort(b_all, u_all, Δy_vec, Δz_vec, Nx, Ny, Nz, Nt;
                           b_range=b_range, n_b_bins=n_b_bins)

println("computing ψ(x, z, t)...")
ψ = get_ψ(u_all, Δy_vec, Δz_vec, Nx, Nz, Nt)

# ---- save ----
println("saving to $outfile ...")
NCDataset(outfile, "c") do ds_out
    defDim(ds_out, "x",    Nx)
    defDim(ds_out, "b",    n_b_bins)
    defDim(ds_out, "z",    Nz)
    defDim(ds_out, "time", Nt)

    defVar(ds_out, "x",    x,      ("x",))
    defVar(ds_out, "b",    b_bins, ("b",))
    defVar(ds_out, "z",    z,      ("z",))
    defVar(ds_out, "time", time,   ("time",))

    v_ψb = defVar(ds_out, "psi_b", Float32, ("x", "b", "time"))
    v_ψb[:, :, :] = ψ_b
    v_ψb.attrib["long_name"] = "overturning streamfunction in buoyancy space ψ(x,b,t)"
    v_ψb.attrib["units"]     = "m²/s"

    v_ψ = defVar(ds_out, "psi", Float32, ("x", "z", "time"))
    v_ψ[:, :, :] = ψ
    v_ψ.attrib["long_name"] = "overturning streamfunction ψ(x,z,t)"
    v_ψ.attrib["units"]     = "m²/s"

    ds_out.attrib["Ra"]      = "1e8"
    ds_out.attrib["source"]  = "Control/RA1e8/4x_stretch/512_128"
    ds_out.attrib["segments"] = "1:12"
end
println("saved → $outfile")
