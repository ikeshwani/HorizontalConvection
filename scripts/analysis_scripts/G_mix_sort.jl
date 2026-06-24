# G_mix_sort.jl
#
# Regional G_mix(b,t) + overturning streamfunction via the SORTED-BINNING
# estimator (G_mix_calc: sort cells by buoyancy, cumulative ∫χ·dV, then
# 0.5·d²/db²).
#
# Handles BOTH experiments — set `experiment` below:
#   "control" : flat bottom (all cells wet)
#   "hill"    : 3-hill GRC topography (wet mask from immersed boundary)
# The region masks and G_mix physics are geometry-agnostic; only the data path,
# segment range, and wet mask differ between the two.
#
# Thin script: physics (gaussian_smooth, G_mix_calc, get_ψ, region builders,
# save_gmix_regions) lives in TopographicHorizontalConvection; this file just
# loads segment data, calls that physics, saves, and plots.
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/G_mix_sort.jl

using TopographicHorizontalConvection   # physics
using NCDatasets
using CairoMakie
using Printf
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
    segments = 1:19
    tag      = "3hill"
else
    error("unknown experiment: $experiment (use \"control\" or \"hill\")")
end
mkpath(plot_dir)

outfile = joinpath(data_dir, "Gmix_regions_$(tag)_RA1e8_seg$(first(segments))to$(last(segments)).nc")

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

# wet mask: flat-bottom control has no immersed boundary (all wet); hills are
# b==0, so read the wet mask from the second time step (avoids init zeros).
wet = experiment == "hill" ? (Array(ds1["b"][:, :, :, 2]) .!= 0) : trues(Nx, Ny, Nz)
close(ds1)

Δx = reshape(Δx_vec, Nx, 1, 1)
Δy = reshape(Δy_vec, 1, Ny, 1)
Δz = reshape(Δz_vec, 1, 1, Nz)
ΔV = Δx .* Δy .* Δz
ΔA_2d = dropdims(Δx .* Δy, dims=3)

# ---- load segments, skipping overlapping time steps ----
println("loading b, χ, u from segments $(segments)...")
b_segs    = Vector{Array{Float32,4}}()
χ_segs    = Vector{Array{Float32,4}}()
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

        # Clip to the velocity file's time dimension — it may be shorter than b.
        n_v     = size(vfile["u"], 4)
        t_range = valid[1]:min(valid[end], n_v)
        push!(b_segs,    Array(bfile["b"][:, :, :, t_range]))
        push!(χ_segs,    Array(bfile["chi"][:, :, :, t_range]))
        push!(u_segs,    Array(vfile["u"][1:Nx, :, :, t_range]))
        push!(time_segs, t_seg[t_range])

        t_last = t_seg[valid[end]]
        close(bfile); close(vfile)
        @printf("  seg %d: loaded %d steps (t = %.2f → %.2f)\n", s, length(t_range), t_seg[valid[1]], t_last)
    end
end

b_all = cat(b_segs...; dims=4)
χ_all = cat(χ_segs...; dims=4)
u_all = cat(u_segs...; dims=4)
time  = vcat(time_segs...)
Nt    = length(time)
println("total time steps: $Nt  (t = $(time[1]) → $(time[end]))")

# ---- region precompute (physics from src/) ----
region_masks   = gmix_region_masks(x, z, Lx, Ra)
region_precomp = precompute_regions(region_masks, ΔA_2d, wet)

# ---- output b axis ----
b_out = collect(range(b_range[1], b_range[2], length=n_b_bins))[2:end-1]
n_b   = length(b_out)

# ---- main time loop: G_mix per region (physics from src/) ----
Gmix_regions = Dict(r.name => zeros(Float32, n_b, Nt) for r in region_precomp)

println("computing G_mix: $Nt time steps × $(length(region_precomp)) regions...")
for t in 1:Nt
    b_flat   = vec(b_all[:, :, :, t])
    χdV_flat = vec(χ_all[:, :, :, t] .* ΔV)

    for r in region_precomp
        _, G = G_mix_calc(b_flat[r.idxs], χdV_flat[r.idxs], b_range; n_b_bins)
        Gmix_regions[r.name][:, t] = G
    end

    t % 50 == 0 && @printf("  t = %d / %d\n", t, Nt)
end

# ---- streamfunction (physics from src/) ----
println("computing streamfunction ψ...")
ψ = get_ψ(u_all, Δy_vec, Δz_vec, Nx, Nz, Nt)

# ---- save (physics/io from src/) ----
save_gmix_regions(outfile, b_out, time, Gmix_regions, region_precomp, ψ, x, z)

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

figname = "Gmix_density_regions_$(tag)_RA1e8_seg$(first(segments))to$(last(segments)).png"
fig = plot_gmix_regions(b_out, time, Gmix_regions, region_precomp, plot_dir, figname)
