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
experiment = "hill"          # "control" (flat bottom) or "hill" (3-hill GRC)

Ra_str   = "1e8"
b_range  = (-1.0, 1.0)
n_b_bins = 501

if experiment == "control"
    data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_flat_baseforcing_zerostart/"
    segments = 1:16
    tag      = "Control"
    source   = "ra1e8_4xstretch_flat_baseforcing_zerostart"
elseif experiment == "hill"
    data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/"
    segments = 1:23
    tag      = "3hill"
    source   = "ra1e8_4xstretch_threehill_baseforcing_zerostart"
else
    error("unknown experiment: $experiment (use \"control\" or \"hill\")")
end

outfile = joinpath(data_dir, "psi_b_$(tag)_RA1e8_seg$(first(segments))to$(last(segments)).nc")

# ---- load grid info from seg1 ----
println("loading grid info from seg1...")
ds1    = NCDataset(joinpath(data_dir, "velocities_seg1.nc"))
x      = ds1["x_caa"][:]
y      = ds1["y_aca"][:]
z      = ds1["z_aac"][:]
Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
Δy_vec = ds1["Δy_aca"][:]
Δz_vec = ds1["Δz_aac"][:]
close(ds1)

# ---- stream segments: load one at a time, compute ψ, keep only small outputs ----
#
# ψ(x,b,t) and ψ(x,z,t) are computed independently per time step, so there is no
# need to hold every segment's b/u in RAM at once (that OOM-killed the job).
# Instead we load a single segment, reduce it to the (much smaller) ψ arrays,
# free the big inputs, and only concatenate the compact per-segment results at
# the end.  b_bins is deterministic, so we build it directly here.
println("streaming segments $(segments): load → compute ψ → free...")
b_bins    = collect(range(b_range[1], b_range[2], length=n_b_bins))
ψb_segs   = Vector{Array{Float32,3}}()
ψ_segs    = Vector{Array{Float32,3}}()
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

        b_seg = Array(bfile["b"][:, :, :, t_range])
        u_seg = Array(vfile["u"][1:Nx, :, :, t_range])
        close(bfile); close(vfile)

        Nt_seg = length(t_range)
        ψb_seg, _ = get_ψb_sort(b_seg, u_seg, Δy_vec, Δz_vec, Nx, Ny, Nz, Nt_seg;
                                b_range=b_range, n_b_bins=n_b_bins)
        ψ_seg     = get_ψ(u_seg, Δy_vec, Δz_vec, Nx, Nz, Nt_seg)

        push!(ψb_segs,   ψb_seg)
        push!(ψ_segs,    ψ_seg)
        push!(time_segs, t_seg[t_range])

        b_seg = nothing; u_seg = nothing
        GC.gc()

        t_last = t_seg[valid[end]]
        @printf("  seg %d: %d steps (t = %.2f → %.2f) → ψ computed\n",
                s, Nt_seg, t_seg[valid[1]], t_last)
    end
end

ψ_b  = cat(ψb_segs...; dims=3)
ψ    = cat(ψ_segs...;  dims=3)
time = vcat(time_segs...)
Nt   = length(time)
println("total time steps: $Nt  (t = $(time[1]) → $(time[end]))")

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
