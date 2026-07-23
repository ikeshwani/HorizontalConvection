# G_mix_domain.jl
#
# Whole-domain G_mix(b,t) + overturning streamfunction via the SORTED-BINNING
# estimator (G_mix_calc).  Unlike G_mix_sort.jl — which splits the domain into
# nine regions — this computes a SINGLE G_mix over every wet cell in the domain.
#
# Handles BOTH experiments — set `experiment` below:
#   "control" : flat bottom (all cells wet)
#   "hill"    : 3-hill GRC topography (wet mask from immersed boundary)
#
# Thin script: physics (G_mix_calc, get_ψ) lives in
# TopographicHorizontalConvection; this file loads segment data, calls that
# physics over the full wet domain, saves, and plots.
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/G_mix_domain.jl

using TopographicHorizontalConvection   # physics: G_mix_calc, get_ψ
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
    data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_flat_baseforcing_zerostart/"
    segments = 14:15
    tag      = "Control"
elseif experiment == "hill"
    data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/"
    segments = 1:22
    tag      = "3hill"
else
    error("unknown experiment: $experiment (use \"control\" or \"hill\")")
end
plot_dir = joinpath(data_dir, "figures")   # figures live inside the run folder
mkpath(plot_dir)

outfile = joinpath(data_dir, "Gmix_domain_$(tag)_RA1e8_seg$(first(segments))to$(last(segments)).nc")

# ---- load grid info from seg1 ----
ds1    = NCDataset(joinpath(data_dir, "buoyancy_seg14.nc"))
x      = ds1["x_caa"][:]
z      = ds1["z_aac"][:]
Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
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

wet_idxs = findall(vec(wet))   # linear indices of all wet cells (whole domain)

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

# ---- output b axis ----
b_out = collect(range(b_range[1], b_range[2], length=n_b_bins))[2:end-1]
n_b   = length(b_out)

# ---- main time loop: G_mix over the whole wet domain (physics from src/) ----
Gmix = zeros(Float32, n_b, Nt)



println("computing whole-domain G_mix: $Nt time steps...")
for t in 1:Nt
    b_flat   = vec(b_all[:, :, :, t])
    χdV_flat = vec(χ_all[:, :, :, t] .* ΔV)
    _, G = G_mix_calc(b_flat[wet_idxs], χdV_flat[wet_idxs], b_range; n_b_bins)
    Gmix[:, t] = G
    t % 50 == 0 && @printf("  t = %d / %d\n", t, Nt)
end

#time averaged gmix
Gmix_mean = nanmean(Gmix, dims=2)

# ---- streamfunction (physics from src/) ----
println("computing streamfunction ψ...")
ψ = get_ψ(u_all, Δy_vec, Δz_vec, Nx, Nz, Nt)

# # ---- save ----
# println("saving to $outfile ...")
# NCDataset(outfile, "c") do ds_out
#     defDim(ds_out, "b",    n_b)
#     defDim(ds_out, "time", Nt)
#     defDim(ds_out, "x",    Nx)
#     defDim(ds_out, "z",    Nz)

#     defVar(ds_out, "b",    b_out, ("b",))
#     defVar(ds_out, "time", time,  ("time",))
#     defVar(ds_out, "x",    x,     ("x",))
#     defVar(ds_out, "z",    z,     ("z",))

#     v_g = defVar(ds_out, "Gmix", Gmix, ("b", "time"))
#     v_g.attrib["long_name"] = "whole-domain G_mix density G(b,t)"

#     defVar(ds_out, "psi", ψ, ("x", "z", "time"))

#     ds_out.attrib["experiment"] = tag
#     ds_out.attrib["Ra"]         = Ra
#     ds_out.attrib["segments"]   = "$(first(segments)):$(last(segments))"
# end
# println("saved → $outfile")

# ---- plot ----
figname = "Gmix_domain_$(tag)_RA1e8_seg$(first(segments))to$(last(segments)).png"
fig  = Figure(size=(900, 450))
clim = 0.008
ax   = Axis(fig[1, 1], xlabel="time", ylabel="buoyancy",
            title="Whole-domain G_mix density — $(tag) Ra=$(Ra)")
hm = heatmap!(ax, time, b_out, Gmix', colormap=:balance, colorrange=(-clim, clim))
Colorbar(fig[1, 2], hm)
save(joinpath(plot_dir, figname), fig)
println("saved figure → $(joinpath(plot_dir, figname))")

figname = "Gmix_domain_time_avg$(tag)_RA1e8_seg$(first(segments))to$(last(segments)).png"
fig  = Figure(size=(900, 450))
clim = 0.008
ax   = Axis(fig[1, 1], xlabel="time", ylabel="buoyancy",
            title="Whole-domain G_mix density — $(tag) Ra=$(Ra)")
hm = heatmap!(ax, time, b_out, Gmix', colormap=:balance, colorrange=(-clim, clim))
Colorbar(fig[1, 2], hm)
save(joinpath(plot_dir, figname), fig)
println("saved figure → $(joinpath(plot_dir, figname))")