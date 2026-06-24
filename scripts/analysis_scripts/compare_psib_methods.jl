# compare_psib_methods.jl
#
# Validation: confirm the two ψ(x,b,t) estimators agree —
#   get_ψb_sort (fast sort + cumsum)  vs  get_ψb_mask (slow per-bin masking).
# Both live in src/analysis/streamfunction.jl and take identical arguments, so
# this runs each on the same loaded data and compares them numerically + visually
#   - numeric: max |ψ_sort - ψ_mask|  (absolute and relative)
#   - visual : ψ_b averaged over x and time, vs buoyancy, both methods overlaid
#
# RESULT (hill GRC, RA1e8, seg 14:15, 52 time steps, stride 12) — 2026-06-24:
#   max |ψ_sort - ψ_mask|   = 3.376e-09
#   max |ψ_sort| (scale)    = 3.943e-03
#   max relative difference = 8.561e-07     -> agreement to Float32 round-off
#   timing: sort 8.8 s   vs   mask 321 s    (~36x slower; mask alloc'd 229 GiB)
# The overlaid curves are visually indistinguishable. The masking method is thus
# validated as a reference; get_ψb_sort is the one to use in production.
#
# NOTE: the mask method is memory/time heavy (rebuilds a full [Nx,Ny,Nz,Nt]
# array per buoyancy bin). For interactive checks subsample time (time_stride)
# and/or use few segments; for a full run submit it as a SLURM job.
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/compare_psib_methods.jl

using TopographicHorizontalConvection   # get_ψb_sort, get_ψb_mask
using NCDatasets
using NaNStatistics
using CairoMakie
using Printf

# ---- config ----
data_dir    = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/"
fig_dir     = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU/GRC/RA1e8/4x_stretch/"
segments    = 14:15
time_stride = 12          # subsample time to keep the mask method quick here
b_range     = (-1.0, 1.0)
n_b_bins    = 501
mkpath(fig_dir)

# ---- grid info from the first segment ----
ds1    = NCDataset(joinpath(data_dir, "buoyancy_seg$(first(segments)).nc"))
Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
Δy_vec = ds1["Δy_aca"][:]
Δz_vec = ds1["Δz_aac"][:]
close(ds1)

# ---- load segments (dedup overlap, subsample time) ----
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
            close(bfile); close(vfile); continue
        end
        n_v       = size(vfile["u"], 4)
        full_rng  = valid[1]:min(valid[end], n_v)
        t_range   = full_rng[1:time_stride:end]      # strided read keeps memory low
        push!(b_segs,    Array(bfile["b"][:, :, :, t_range]))
        push!(u_segs,    Array(vfile["u"][1:Nx, :, :, t_range]))
        push!(time_segs, t_seg[t_range])
        t_last = t_seg[valid[end]]
        close(bfile); close(vfile)
        @printf("  seg %d: loaded %d steps (stride %d)\n", s, length(t_range), time_stride)
    end
end

b_all = cat(b_segs...; dims=4)
u_all = cat(u_segs...; dims=4)
time  = vcat(time_segs...)
Nt    = length(time)
@printf("total steps used: %d  (t = %.1f → %.1f)\n", Nt, time[1], time[end])

# ---- compute both methods on identical input ----
println("sort method..."); @time ψ_sort, b_bins = get_ψb_sort(b_all, u_all, Δy_vec, Δz_vec, Nx, Ny, Nz, Nt; b_range=b_range, n_b_bins=n_b_bins)
println("mask method..."); @time ψ_mask, _      = get_ψb_mask(b_all, u_all, Δy_vec, Δz_vec, Nx, Ny, Nz, Nt; b_range=b_range, n_b_bins=n_b_bins)

# ---- numeric comparison ----
Δ        = abs.(ψ_sort .- ψ_mask)
max_abs  = maximum(Δ)
scale    = maximum(abs.(ψ_sort))
max_rel  = max_abs / scale
@printf("\nmax |ψ_sort - ψ_mask|        = %.3e\n", max_abs)
@printf("max |ψ_sort| (scale)         = %.3e\n", scale)
@printf("max relative difference      = %.3e\n", max_rel)

# ---- visual comparison: ψ_b averaged over x and time, vs buoyancy ----
ψ_sort_mean = vec(nanmean(nanmean(ψ_sort, dims=1), dims=3))   # [n_b_bins]
ψ_mask_mean = vec(nanmean(nanmean(ψ_mask, dims=1), dims=3))

fig = Figure(size=(800, 600))
ax  = Axis(fig[1,1],
           xlabel = "buoyancy b",
           ylabel = "⟨ψ_b⟩ (x- and time-averaged) [m²/s]",
           title  = @sprintf("ψ_b mask vs sort comparison — hill seg %d:%d (max rel diff = %.1e)",
                             first(segments), last(segments), max_rel))
lines!(ax, b_bins, ψ_sort_mean; linewidth=3, color=:dodgerblue, label="sort method")
lines!(ax, b_bins, ψ_mask_mean; linewidth=2, linestyle=:dash, color=:red, label="mask method")
axislegend(ax; position=:rt)

outpng = joinpath(fig_dir, "psi_mask_vs_sort_comparison_hill_seg14to15.png")
save(outpng, fig)
@printf("saved → %s\n", outpng)
