# G_mix_flux_convergence.jl
#
# Regional G_mix(b,t) + overturning streamfunction via the THIRD estimator:
# convergence of the diffusive buoyancy flux (G_mix_calc_v3):
#
#     G_mix(b) = d/db ∫_{V(b'<b)} -∇·(-κ∇b) dV
#
# For each snapshot the per-cell flux convergence ∫_cell(-∇·F)dV is built once on
# the full domain (diffusive_flux_convergence), then accumulated over V(b'<b)
# within each region and differentiated once in b (G_mix_calc_v3).  This is a
# genuinely different route from the sorted-binning (v1) and product-rule (v2)
# estimators and the three can disagree — all are kept on purpose.
#
# Handles BOTH experiments — set `experiment` below:
#   "control" : flat bottom (all cells wet)
#   "hill"    : 3-hill GRC topography (wet mask from immersed boundary)
#
# Output format is IDENTICAL to G_mix_sort_v2.jl (save_gmix_regions): per-region
# Gmix_<name>(b,t) + ψ(x,z,t), on the same b axis.  Only the filename tag (v3)
# and the estimator differ.
#
# Memory-efficient two-pass loader: pass 1 scans only the time vectors, pass 2
# loads one segment of field data at a time.
#
# Thin script: physics (diffusive_flux_convergence, G_mix_calc_v3, region
# builders, get_ψ, save_gmix_regions) lives in TopographicHorizontalConvection.
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/G_mix_flux_convergence.jl

using TopographicHorizontalConvection   # physics
using NCDatasets
using CairoMakie
using Printf
using Statistics

# ---- config ----
# experiment / Ra can be overridden from the environment (GMIX_EXPERIMENT, GMIX_RA)
# so one submit script can launch both runs; defaults reproduce the hill Ra1e8 case.
experiment = get(ENV, "GMIX_EXPERIMENT", "hill")   # "control" (flat) or "hill" (3-hill GRC)

Ra       = parse(Float64, get(ENV, "GMIX_RA", "1e8"))   # 1e8 or 1e6
b_range  = (-1, 1)
n_b_bins = 501

# Ra → path tag + filename tag
if Ra == 1e8
    Ra_tag, Ra_str = "ra1e8", "RA1e8"
elseif Ra == 1e6
    Ra_tag, Ra_str = "ra1e6", "RA1e6"
else
    error("unsupported Ra: $Ra (use 1e8 or 1e6)")
end

if experiment == "control"
    topo     = "flat"
    tag      = "Control"
    segments = Ra == 1e8 ? (1:16) : (1:7)
elseif experiment == "hill"
    topo     = "threehill"
    tag      = "3hill"
    segments = Ra == 1e8 ? (1:23) : (1:10)
else
    error("unknown experiment: $experiment (use \"control\" or \"hill\")")
end

data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/$(Ra_tag)_4xstretch_$(topo)_baseforcing_zerostart/"
plot_dir = joinpath(data_dir, "figures")   # figures live inside the run folder
mkpath(plot_dir)

outfile = joinpath(data_dir, "Gmix_regions_v3_$(tag)_$(Ra_str)_seg$(first(segments))to$(last(segments)).nc")

# ---- load grid info from seg1 ----
ds1    = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
x      = ds1["x_caa"][:]
y      = ds1["y_aca"][:]
z      = ds1["z_aac"][:]
Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
Lx     = ds1.attrib["Lx"]
Ra_a   = ds1.attrib["Ra"]
Pr     = ds1.attrib["Pr"]
b★     = ds1.attrib["b★"]
H      = ds1.attrib["H"]

Δx_vec = ds1["Δx_caa"][:]           # cell widths (centers)
Δy_vec = ds1["Δy_aca"][:]
Δz_vec = ds1["Δz_aac"][:]
Δx_faa = ds1["Δx_faa"][:]           # center-to-center distances (faces)
Δy_afa = ds1["Δy_afa"][:]
Δz_aaf = ds1["Δz_aaf"][:]

# wet mask: flat-bottom control is all wet; hills are b==0, so read the wet mask
# from the second time step (avoids init zeros).
wet = experiment == "hill" ? (Array(ds1["b"][:, :, :, 2]) .!= 0) : trues(Nx, Ny, Nz)
close(ds1)

# nondimensional diffusivity: ν = sqrt(Pr·b★·H³/Ra), κ = ν/Pr
ν = sqrt(Pr * b★ * H^3 / Ra_a)
κ = ν / Pr
@printf("κ = %.3e   (Ra=%.0e, Pr=%g)\n", κ, Ra_a, Pr)

Δx = reshape(Δx_vec, Nx, 1, 1)
Δy = reshape(Δy_vec, 1, Ny, 1)
Δz = reshape(Δz_vec, 1, 1, Nz)
ΔA_2d = dropdims(Δx .* Δy, dims=3)

# ---- region precompute (physics from src/) ----
region_masks   = gmix_region_masks(x, z, Lx, Ra)
region_precomp = precompute_regions(region_masks, ΔA_2d, wet)

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
println("Pass 2: computing G_mix (flux-convergence) + ψ segment by segment...")

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
    u_seg = Array(vfile["u"][1:Nx, :, :, t_range])
    close(bfile); close(vfile)

    # ---- G_mix for each time step in this segment (physics from src/) ----
    for (ti, g) in enumerate(gi)
        b_snap   = b_seg[:, :, :, ti]
        CV       = diffusive_flux_convergence(b_snap, wet, κ,
                                              Δx_vec, Δy_vec, Δz_vec,
                                              Δx_faa, Δy_afa, Δz_aaf)
        b_flat   = vec(b_snap)
        CV_flat  = vec(CV)
        for r in region_precomp
            G = G_mix_calc_v3(b_flat[r.idxs], CV_flat[r.idxs], b_range; n_b_bins)
            Gmix_regions[r.name][:, g] = G
        end
        g % 100 == 0 && @printf("    G_mix: step %d / %d\n", g, Nt)
    end

    # ---- ψ for this segment: ψ[x,z,t] = -cumsum_z(∫u dy · Δz) (physics from src/) ----
    ψ_all[:, :, gi] = get_ψ(u_seg, Δy_vec, Δz_vec, Nx, Nz, nt)

    t_last   = t_seg[t_range[end]]
    t_offset += nt

    # free segment arrays before loading the next segment
    b_seg = nothing; u_seg = nothing
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

figname = "Gmix_density_regions_v3_$(tag)_$(Ra_str)_seg$(first(segments))to$(last(segments)).png"
plot_gmix_regions(b_out, time, Gmix_regions, region_precomp, plot_dir, figname)
