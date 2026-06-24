# psi_of_b_sort.jl
#
# Overturning streamfunction in buoyancy space ψ(x,b,t) and physical space
# ψ(x,z,t), via the sort + cumsum method.
#
# Handles BOTH experiments — set `experiment` below:
#   "control" : flat bottom
#   "hill"    : 3-hill GRC topography
# The physics (get_ψb_sort / get_ψ) is geometry-agnostic: immersed cells under
# the hills have u=0/b=0 and are masked out exactly like the (absent) dry cells
# in the flat case, so the same functions serve both.
#
# Thin script: get_ψb_sort / get_ψ physics live in
# TopographicHorizontalConvection; this file loads segment data, calls them,
# and writes the NetCDF.
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/psi_of_b_sort.jl

using TopographicHorizontalConvection   # physics: get_ψb_sort, get_ψ
using NCDatasets
using Printf

# ---- config ----
experiment = "control"          # "control" (flat bottom) or "hill" (3-hill GRC)

Ra_str   = "1e8"
b_range  = (-1.0, 1.0)
n_b_bins = 501

if experiment == "control"
    data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/Control/RA1e8/4x_stretch/512_128/"
    segments = 1:12
    tag      = "Control"
    source   = "Control/RA1e8/4x_stretch/512_128"
elseif experiment == "hill"
    data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/"
    segments = 1:20
    tag      = "3hill"
    source   = "GRC/RA1e8/4x_stretch/512_128"
else
    error("unknown experiment: $experiment (use \"control\" or \"hill\")")
end

outfile = joinpath(data_dir, "psi_b_$(tag)_RA1e8_seg$(first(segments))to$(last(segments)).nc")

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

# ---- compute (physics from src/) ----
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

    ds_out.attrib["Ra"]       = Ra_str
    ds_out.attrib["source"]   = source
    ds_out.attrib["segments"] = "$(first(segments)):$(last(segments))"
end
println("saved → $outfile")
