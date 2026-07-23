# compare_gmix_methods.jl
#
# Compare two G_mix estimators, region by region:
#   v2   = G_mix_calc_v2  (product-rule decomposition of d²/db²[V·χ̄]) — the
#          sorting method used in G_mix_sort_v2.jl, computed from χ.
#   CODF = convergence of the diffusive flux (the method in gmix_CODF.jl):
#          G_mix(b) = ∂/∂b ∫_{V(b'<b)} -∇·(-κ∇b) dV, computed from b directly.
#          Inlined here verbatim from gmix_CODF.jl (NOT the src helpers).
#
# Run on the HILL simulation, segments 14:15.  For each of the 9 regions we
# time-average G_mix(b,t) over the loaded window and overlay the two methods on
# its own panel (3×3 grid).  Time is subsampled (stride) to stay light.
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/compare_gmix_methods.jl

using TopographicHorizontalConvection   # physics
using NCDatasets
using CairoMakie
using Printf
using Statistics
using NaNStatistics

# ---- config ----
data_dir    = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/"
fig_dir     = joinpath(data_dir, "figures")
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
Pr, b★, H = ds1.attrib["Pr"], ds1.attrib["b★"], ds1.attrib["H"]
Δx_vec = ds1["Δx_caa"][:]
Δy_vec = ds1["Δy_aca"][:]
Δz_vec = ds1["Δz_aac"][:]
Δx_faa = ds1["Δx_faa"][:]   # center-to-center spacings (faces), for CODF gradients
Δy_afa = ds1["Δy_afa"][:]
Δz_aaf = ds1["Δz_aaf"][:]
wet    = Array(ds1["b"][:, :, :, 2]) .!= 0   # hills are b==0
close(ds1)

# diffusivity for the flux-convergence estimator (CODF)
ν = sqrt(Pr * b★ * H^3 / Ra)
κ = ν / Pr

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

b_out = collect(range(b_range[1], b_range[2], length=n_b_bins))[2:end-1]   # v2 axis (499)
n_b   = length(b_out)

# ============================================================================
# CODF method — inlined verbatim from gmix_CODF.jl (convergence of the
# diffusive flux). b-axis is the 500 edge-midpoints (b_codf), distinct from the
# 499-point b_out that v2 uses — each line is plotted on its own axis.
# ============================================================================
b_edges_codf = collect(range(b_range[1], b_range[2], length=n_b_bins))
b_codf       = 0.5 .* (b_edges_codf[1:end-1] .+ b_edges_codf[2:end])   # 500 midpoints

# per-cell ∫_cell(-∇·F)dV for one snapshot b [Nx,Ny,Nz]
function conv_dV_snapshot(b)
    b = Array{Float64}(b)
    b[.!wet] .= NaN

    # X (solid walls at both ends: no-flux)
    flux_x = -κ .* diff(b, dims=1) ./ reshape(Δx_faa[2:Nx], Nx-1, 1, 1)
    flux_x_full = zeros(Nx+1, Ny, Nz)
    flux_x_full[2:Nx, :, :] .= flux_x
    flux_x_full[isnan.(flux_x_full)] .= 0.0
    convX = -1 .* diff(flux_x_full, dims=1) ./ Δx

    # Y (periodic: face 1 == face Ny+1 == wrap face)
    flux_y = -κ .* diff(b, dims=2) ./ reshape(Δy_afa[2:Ny], 1, Ny-1, 1)
    flux_y_wrap = -κ .* (b[:, 1:1, :] .- b[:, Ny:Ny, :]) ./ Δy_afa[1]
    flux_y_full = zeros(Nx, Ny+1, Nz)
    flux_y_full[:, 2:Ny, :] .= flux_y
    flux_y_full[:, 1,    :] .= flux_y_wrap[:, 1, :]
    flux_y_full[:, Ny+1, :] .= flux_y_wrap[:, 1, :]
    flux_y_full[isnan.(flux_y_full)] .= 0.0
    convY = -1 .* diff(flux_y_full, dims=2) ./ Δy

    # Z (bottom wall + top surface: no-flux)
    flux_z = -κ .* diff(b, dims=3) ./ reshape(Δz_aaf[2:Nz], 1, 1, Nz-1)
    flux_z_full = zeros(Nx, Ny, Nz+1)
    flux_z_full[:, :, 2:Nz] .= flux_z
    flux_z_full[isnan.(flux_z_full)] .= 0.0
    convZ = -1 .* diff(flux_z_full, dims=3) ./ Δz

    CONV_dV = (convX .+ convY .+ convZ) .* ΔV
    CONV_dV[.!wet] .= 0.0
    return CONV_dV
end

# G_mix(b) over one region: M(b)=∫_{b'<b}(-∇·F)dV over its cells, then one d/db
function gmix_region_codf(b_flat, CONV_flat, idxs)
    bg = b_flat[idxs]
    cg = CONV_flat[idxs]
    M  = zeros(n_b_bins)
    for n in 1:n_b_bins
        M[n] = sum(@view cg[bg .< b_edges_codf[n]])
    end
    return diff(M) ./ diff(b_edges_codf)
end

# ---- compute both estimators, accumulate time mean per region ----
G2 = Dict(r.name => zeros(Float64, n_b)         for r in region_precomp)  # v2 product-rule (sort)
Gc = Dict(r.name => zeros(Float64, n_b_bins-1)  for r in region_precomp)  # CODF flux-convergence

println("computing G_mix v2 + CODF over $Nt steps × $(length(region_precomp)) regions...")
for t in 1:Nt
    b_snap    = b_all[:, :, :, t]
    b_flat    = vec(b_snap)
    χdV_flat  = vec(χ_all[:, :, :, t] .* ΔV)
    CONV_flat = vec(conv_dV_snapshot(b_snap))
    for r in region_precomp
        g2 = G_mix_calc_v2(b_flat[r.idxs], χdV_flat[r.idxs], dV_flat[r.idxs], b_range; n_b_bins)
        gc = gmix_region_codf(b_flat, CONV_flat, r.idxs)
        G2[r.name] .+= g2
        Gc[r.name] .+= gc
    end
end
for r in region_precomp
    G2[r.name] ./= Nt
    Gc[r.name] ./= Nt
end

# ---- magnitude summary (axes differ, so report peaks separately) ----
println("\nregion           max|v2|       max|CODF|")
for r in region_precomp
    @printf("  %-15s %.4e   %.4e\n", r.name,
            maximum(abs.(G2[r.name])), maximum(abs.(Gc[r.name])))
end

# ---- plot: 3×3, one panel per region, v2 vs CODF overlaid ----
names = [r.name for r in region_precomp]
fig   = Figure(size=(1300, 1100))
for (i, name) in enumerate(names)
    row = (i - 1) ÷ 3 + 1
    col = (i - 1) % 3 + 1
    ax  = Axis(fig[row, col], title=name, xlabel="buoyancy", ylabel="⟨G_mix⟩ₜ")
    l2  = lines!(ax, b_out,  G2[name], color=:red,      linewidth=2, linestyle=:dash)
    lc  = lines!(ax, b_codf, Gc[name], color=:seagreen, linewidth=3)
    if i == 1
        axislegend(ax, [l2, lc],
                   ["v2 sort (product-rule)", "CODF (flux-conv)"], position=:rt)
    end
end
Label(fig[0, :], "G_mix: v2 sort (product-rule) vs CODF (flux-conv) — hill seg $(first(segments))–$(last(segments)), time-averaged",
      fontsize=20, font=:bold)

figname = "Gmix_sortv2_vs_CODF_comparison_3hill_seg$(first(segments))to$(last(segments)).png"
save(joinpath(fig_dir, figname), fig)
println("\nsaved figure → $(joinpath(fig_dir, figname))")
