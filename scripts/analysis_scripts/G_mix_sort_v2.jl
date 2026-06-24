# G_mix_sort_v2.jl
#
# Regional G_mix(b,t) + overturning streamfunction via the PRODUCT-RULE
# estimator (G_mix_calc_v2: decompose d²/db²[V(b)·χ̄(b)] = V·χ̄'' + 2·V'·χ̄' +
# V''·χ̄).  This is a genuinely different estimator from the sorted-binning
# method in G_mix_sort.jl and the two can disagree — both are kept on purpose.
#
# Handles BOTH experiments — set `experiment` below:
#   "control" : flat bottom (all cells wet)
#   "hill"    : 3-hill GRC topography (wet mask from immersed boundary)
#
# Memory-efficient two-pass loader: pass 1 scans only the time vectors, pass 2
# loads one segment of field data at a time so the full b/χ/u arrays never sit
# in memory at once.
#
# Thin script: physics (gaussian_smooth, G_mix_calc_v2, region builders,
# save_gmix_regions) lives in TopographicHorizontalConvection.
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/G_mix_sort_v2.jl

using TopographicHorizontalConvection   # physics
using NCDatasets
using CairoMakie
using Printf
using Statistics
using NaNStatistics

# ---- config ----
experiment = "control"          # "control" (flat bottom) or "hill" (3-hill GRC)

Ra       = 1e8
b_range  = (-1, 1)
n_b_bins = 501

if experiment == "control"
    data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/Control/RA1e8/4x_stretch/512_128/"
    plot_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU/GRC/Control/RA1e8/4x_stretch/figures/"
    segments = 1:12
    tag      = "Control"
elseif experiment == "hill"
    data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/"
    plot_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU/GRC/RA1e8/4x_stretch/figures/"
    segments = 1:14
    tag      = "3hill"
else
    error("unknown experiment: $experiment (use \"control\" or \"hill\")")
end
mkpath(plot_dir)

outfile = joinpath(data_dir, "Gmix_regions_v2_$(tag)_RA1e8_seg$(first(segments))to$(last(segments)).nc")

# ---- load grid info from seg1 ----
ds1    = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
x      = ds1["x_caa"][:]
y      = ds1["y_aca"][:]
z      = ds1["z_aac"][:]
Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
Lx     = ds1.attrib["Lx"]
Δx_vec = ds1["Δx_caa"][:]
Δy_vec = ds1["Δy_aca"][:]
Δz_vec = ds1["Δz_aac"][:]

# wet mask: flat-bottom control is all wet; hills are b==0, so read the wet mask
# from the second time step (avoids init zeros).
wet = experiment == "hill" ? (Array(ds1["b"][:, :, :, 2]) .!= 0) : trues(Nx, Ny, Nz)
close(ds1)

Δx = reshape(Δx_vec, Nx, 1, 1)
Δy = reshape(Δy_vec, 1, Ny, 1)
Δz = reshape(Δz_vec, 1, 1, Nz)
ΔV = Δx .* Δy .* Δz
ΔA_2d = dropdims(Δx .* Δy, dims=3)

# ---- region precompute (physics from src/) ----
region_masks   = gmix_region_masks(x, z, Lx, Ra)
region_precomp = precompute_regions(region_masks, ΔA_2d, wet)

# ---- precompute time-invariant dV flat vector ----
dV_flat = vec(ΔV)

# ---- output b axis ----
b_out = collect(range(b_range[1], b_range[2], length=n_b_bins))[2:end-1]
n_b   = length(b_out)

# ---- Pass 1: collect time vector (cheap — no field data loaded) ----
println("Pass 1: scanning time vectors from segments $(segments[1])–$(segments[end])...")
time_all = Float64[]
let t_last = -Inf
    for s in segments
        bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
        t_seg = Float64.(bfile["time"][:])
        close(bfile)
        valid = findall(t_seg .> t_last)
        isempty(valid) && continue
        append!(time_all, t_seg[valid])
        t_last = t_seg[valid[end]]
    end
end
Nt   = length(time_all)
time = time_all
println("total time steps: $Nt  (t = $(time[1]) → $(time[end]))")

# ---- pre-allocate output arrays ----
Gmix_regions = Dict(r.name => zeros(Float32, n_b, Nt) for r in region_precomp)
ψ_all        = zeros(Float32, Nx, Nz, Nt)

# ---- Pass 2: process one segment at a time ----
println("Pass 2: computing G_mix + ψ segment by segment...")

let t_last = -Inf, t_offset = 0
for s in segments
    bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
    vfile = NCDataset(joinpath(data_dir, "velocities_seg$(s).nc"))

    t_seg = Float64.(bfile["time"][:])
    valid = findall(t_seg .> t_last)

    if isempty(valid)
        @printf("  seg %d: all steps are duplicates — skipping\n", s)
        close(bfile); close(vfile)
        continue
    end

    n_skip = valid[1] - 1
    n_skip > 0 && @printf("  seg %d: skipping first %d overlapping step(s)\n", s, n_skip)

    n_v     = size(vfile["u"], 4)
    t_range = valid[1]:min(valid[end], n_v)
    nt      = length(t_range)
    gi      = t_offset+1 : t_offset+nt   # global index range

    @printf("  seg %d: loading %d steps (t = %.2f → %.2f)...\n",
            s, nt, t_seg[t_range[1]], t_seg[t_range[end]])

    b_seg = Array(bfile["b"][:, :, :, t_range])
    χ_seg = Array(bfile["chi"][:, :, :, t_range])
    u_seg = Array(vfile["u"][1:Nx, :, :, t_range])
    close(bfile); close(vfile)

    # ---- G_mix for each time step in this segment (physics from src/) ----
    for (ti, g) in enumerate(gi)
        b_flat   = vec(b_seg[:, :, :, ti])
        χdV_flat = vec(χ_seg[:, :, :, ti] .* ΔV)
        for r in region_precomp
            G = G_mix_calc_v2(b_flat[r.idxs], χdV_flat[r.idxs], dV_flat[r.idxs], b_range; n_b_bins)
            Gmix_regions[r.name][:, g] = G
        end
        g % 100 == 0 && @printf("    G_mix: step %d / %d\n", g, Nt)
    end

    # ---- ψ for this segment: ψ[x,z,t] = -cumsum_z(∫u dy · Δz) (physics from src/) ----
    ψ_all[:, :, gi] = get_ψ(u_seg, Δy_vec, Δz_vec, Nx, Nz, nt)

    t_last   = t_seg[t_range[end]]
    t_offset += nt

    # free segment arrays before loading the next segment
    b_seg = nothing; χ_seg = nothing; u_seg = nothing
    GC.gc()

    @printf("  seg %d: done\n", s)
end  # for s
end  # let t_last

# ---- save (physics/io from src/) ----
save_gmix_regions(outfile, b_out, time, Gmix_regions, region_precomp, ψ_all, x, z)

# ---- plot ----
function plot_gmix_regions(b_out, time, Gmix_regions, region_precomp, plot_dir, figname)
    names     = [r.name for r in region_precomp]
    n_regions = length(names)
    fig  = Figure(size=(300 * n_regions, 400))
    clim = 0.008
    for (i, name) in enumerate(names)
        ax = Axis(fig[1, 2i-1], xlabel="time", ylabel="buoyancy", title="Gmix density: $name")
        hm = heatmap!(ax, time, b_out, Gmix_regions[name]', colormap=:balance, colorrange=(-clim, clim))
        Colorbar(fig[1, 2i], hm)
    end
    save(joinpath(plot_dir, figname), fig)
    println("saved figure → $(joinpath(plot_dir, figname))")
    return fig
end

figname = "Gmix_density_regions_v2_$(tag)_RA1e8_seg$(first(segments))to$(last(segments)).png"
plot_gmix_regions(b_out, time, Gmix_regions, region_precomp, plot_dir, figname)
