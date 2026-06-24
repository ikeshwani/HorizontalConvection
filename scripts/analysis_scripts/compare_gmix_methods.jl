# compare_gmix_methods.jl
#
# Validation: do the two G_mix estimators agree?
#   v1 = G_mix_calc     (sorted-binning of cumulative ∫χ·dV, then 0.5·d²/db²)
#   v2 = G_mix_calc_v2  (product-rule decomposition of d²/db²[V·χ̄])
#
# Run on the HILL simulation, segments 14:15.  For each of the 9 regions we
# time-average G_mix(b,t) over the loaded window and overlay v1 vs v2 on its own
# panel (3×3 grid).  Time is subsampled (stride) to stay light on a login node.
#
# ----------------------------------------------------------------------------
# RESULT (hill seg 14:15, time_stride=12, 52 timesteps):
#
#   region           max|v1-v2|     max|v1|
#   plume           1.4772e-02   2.4215e-03
#   boundary_layer  2.0886e-02   4.3864e-03
#   basin0          5.7158e-05   7.8868e-05
#   hill1           1.0573e-04   1.2961e-04
#   basin1          4.9976e-05   6.6585e-05
#   hill2           1.3810e-04   1.8556e-04
#   basin2          6.8840e-05   8.9526e-05
#   hill3           1.7596e-04   2.3121e-04
#   basin3          1.6777e-04   2.2522e-04
#
# The two estimators do NOT agree — they are genuinely different numerical
# routes to G_mix and disagree, especially in high-gradient regions:
#   - Interior basins/hills: same shape and sign; v2 (product-rule) is spikier
#     near the core buoyancy values (difference ~ signal magnitude).
#   - plume & boundary_layer: strong divergence — v2 oscillates where v1 is near
#     flat, and max|v1-v2| EXCEEDS max|v1|.  Steep V(b)/χ̄(b) gradients there make
#     the product-rule form V·χ̄'' + 2V'·χ̄' + V''·χ̄ amplify derivative noise,
#     whereas v1 differentiates the already-smoothed cumulative integral once.
#
# This is why G_mix_sort.jl (v1) and G_mix_sort_v2.jl (v2) are kept as separate
# scripts and must NOT be merged.
# Figure: figures/.../Gmix_v1_vs_v2_comparison_3hill_seg14to15.png
# ----------------------------------------------------------------------------
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/compare_gmix_methods.jl

using TopographicHorizontalConvection   # physics
using NCDatasets
using CairoMakie
using Printf
using Statistics
using NaNStatistics

# ---- config ----
data_dir    = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/"
fig_dir     = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU/GRC/RA1e8/4x_stretch/figures/"
segments    = 14:15
Ra          = 1e8
b_range     = (-1, 1)
n_b_bins    = 501
time_stride = 12        # subsample timesteps (keeps the login-node run quick)
mkpath(fig_dir)

# ---- grid info + wet mask from seg1 ----
ds1    = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
x      = ds1["x_caa"][:]
z      = ds1["z_aac"][:]
Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
Lx     = ds1.attrib["Lx"]
Δx_vec = ds1["Δx_caa"][:]
Δy_vec = ds1["Δy_aca"][:]
Δz_vec = ds1["Δz_aac"][:]
wet    = Array(ds1["b"][:, :, :, 2]) .!= 0   # hills are b==0
close(ds1)

Δx = reshape(Δx_vec, Nx, 1, 1)
Δy = reshape(Δy_vec, 1, Ny, 1)
Δz = reshape(Δz_vec, 1, 1, Nz)
ΔV    = Δx .* Δy .* Δz
ΔA_2d = dropdims(Δx .* Δy, dims=3)

# ---- load segments 14:15 (dedup + subsample) ----
println("loading b, χ from segments $(segments) (stride $(time_stride))...")
b_segs = Vector{Array{Float32,4}}()
χ_segs = Vector{Array{Float32,4}}()
let t_last = -Inf
    for s in segments
        bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
        t_seg = bfile["time"][:]
        valid = findall(t_seg .> t_last)
        if isempty(valid); close(bfile); continue; end
        t_range = valid[1]:time_stride:valid[end]
        push!(b_segs, Array(bfile["b"][:, :, :, t_range]))
        push!(χ_segs, Array(bfile["chi"][:, :, :, t_range]))
        t_last = t_seg[valid[end]]
        close(bfile)
        @printf("  seg %d: %d steps\n", s, length(t_range))
    end
end
b_all = cat(b_segs...; dims=4)
χ_all = cat(χ_segs...; dims=4)
Nt    = size(b_all, 4)
println("total timesteps: $Nt")

# ---- regions (physics from src/) ----
region_masks   = gmix_region_masks(x, z, Lx, Ra)
region_precomp = precompute_regions(region_masks, ΔA_2d, wet)
dV_flat        = vec(ΔV)

b_out = collect(range(b_range[1], b_range[2], length=n_b_bins))[2:end-1]
n_b   = length(b_out)

# ---- compute both estimators, accumulate time mean per region ----
G1 = Dict(r.name => zeros(Float64, n_b) for r in region_precomp)  # v1 sort
G2 = Dict(r.name => zeros(Float64, n_b) for r in region_precomp)  # v2 product-rule

println("computing G_mix v1 + v2 over $Nt steps × $(length(region_precomp)) regions...")
for t in 1:Nt
    b_flat   = vec(b_all[:, :, :, t])
    χdV_flat = vec(χ_all[:, :, :, t] .* ΔV)
    for r in region_precomp
        _, g1 = G_mix_calc(b_flat[r.idxs], χdV_flat[r.idxs], b_range; n_b_bins)
        g2    = G_mix_calc_v2(b_flat[r.idxs], χdV_flat[r.idxs], dV_flat[r.idxs], b_range; n_b_bins)
        G1[r.name] .+= g1
        G2[r.name] .+= g2
    end
end
for r in region_precomp
    G1[r.name] ./= Nt
    G2[r.name] ./= Nt
end

# ---- numeric agreement summary ----
println("\nregion           max|v1-v2|     max|v1|")
for r in region_precomp
    d = maximum(abs.(G1[r.name] .- G2[r.name]))
    s = maximum(abs.(G1[r.name]))
    @printf("  %-15s %.4e   %.4e\n", r.name, d, s)
end

# ---- plot: 3×3, one panel per region, both versions overlaid ----
names = [r.name for r in region_precomp]
fig   = Figure(size=(1300, 1100))
for (i, name) in enumerate(names)
    row = (i - 1) ÷ 3 + 1
    col = (i - 1) % 3 + 1
    ax  = Axis(fig[row, col], title=name, xlabel="buoyancy", ylabel="⟨G_mix⟩ₜ")
    l1  = lines!(ax, b_out, G1[name], color=:dodgerblue, linewidth=3)
    l2  = lines!(ax, b_out, G2[name], color=:red, linewidth=2, linestyle=:dash)
    if i == 1
        axislegend(ax, [l1, l2], ["v1 sort", "v2 product-rule"], position=:rt)
    end
end
Label(fig[0, :], "G_mix v1 (sort) vs v2 (product-rule) — hill seg $(first(segments))–$(last(segments)), time-averaged",
      fontsize=20, font=:bold)

figname = "Gmix_v1_vs_v2_comparison_3hill_seg$(first(segments))to$(last(segments)).png"
save(joinpath(fig_dir, figname), fig)
println("\nsaved figure → $(joinpath(fig_dir, figname))")
