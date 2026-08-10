using TopographicHorizontalConvection   # boundary_layer_depth
using NCDatasets
using CairoMakie
using Statistics
using Printf

data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/"
plot_dir = joinpath(@__DIR__)   # scripts/scratch/ -- keep scratch outputs next to the script

# ============================================================================
# EDIT THESE and rerun to play around
# ============================================================================
b0 = -0.3553     # b_in: min_t(b_in(t)) over segment 32 -- the conservative (safe-for-every-t) value
Δb = 0.0066      # b_out(conservative) - b_in(conservative) = -0.3487 - (-0.3553)
it_pick = nothing   # snapshot index into buoyancy_seg32.nc; `nothing` -> last one
# ============================================================================

ds = NCDataset(joinpath(data_dir, "buoyancy_seg32.nc"))
x = ds["x_caa"][:]; y = ds["y_aca"][:]; z = ds["z_aac"][:]
Nx, Ny, Nz = ds.attrib["Nx"], ds.attrib["Ny"], ds.attrib["Nz"]
Ra, Pr, b★, H, Lx = ds.attrib["Ra"], ds.attrib["Pr"], ds.attrib["b★"], ds.attrib["H"], ds.attrib["Lx"]

Δx_face = ds["Δx_faa"][:]; Δy_face = ds["Δy_afa"][:]; Δz_face = ds["Δz_aaf"][:]
Δx_center = reshape(ds["Δx_caa"][:], Nx, 1, 1)
Δy_center = reshape(ds["Δy_aca"][:], 1, Ny, 1)
Δz_center = reshape(ds["Δz_aac"][:], 1, 1, Nz)
vol = Δx_center .* Δy_center .* Δz_center
ν = sqrt(Pr * b★ * H^3 / Ra); κ = ν / Pr

wet = ds["b"][:, :, :, 2] .!= 0
it = it_pick === nothing ? length(ds["time"][:]) : it_pick
t_val = ds["time"][it]
b = Array{Float64}(ds["b"][:, :, :, it])
close(ds)
b[.!wet] .= NaN

# ---- hill1 INTERIOR region only (excludes bl_hill1, the near-surface strip) --
zBL = boundary_layer_depth(Lx, Ra)
mask_hill1 = (reshape(x, Nx,1,1) .>= -1.35) .& (reshape(x, Nx,1,1) .< -0.65) .&
             (reshape(z, 1,1,Nz) .< zBL)
idxs = findall(vec(mask_hill1 .& wet))

b_h1 = vec(b)[idxs]
println("t = $t_val   hill1-interior cell count = $(length(idxs))")
println("hill1-interior buoyancy range: ", extrema(filter(isfinite, b_h1)))
println("zBL = $zBL  (hill1-interior is everything below this)")

# ---- conv_dV_snapshot: per-cell convergence of the diffusive flux -----------
# (same formula as gmix_CODF.jl / FBN_PSI.jl -- copied here so this script is
# fully self-contained and easy to poke at)
function conv_dV_snapshot(b)
    flux_x = -κ .* diff(b, dims=1) ./ reshape(Δx_face[2:Nx], Nx-1, 1, 1)
    flux_x_full = zeros(Nx+1, Ny, Nz)
    flux_x_full[2:Nx, :, :] .= flux_x
    flux_x_full[isnan.(flux_x_full)] .= 0.0
    convX = -1 .* diff(flux_x_full, dims=1) ./ Δx_center

    flux_y = -κ .* diff(b, dims=2) ./ reshape(Δy_face[2:Ny], 1, Ny-1, 1)
    flux_y_wrap = -κ .* (b[:, 1:1, :] .- b[:, Ny:Ny, :]) ./ Δy_face[1]
    flux_y_full = zeros(Nx, Ny+1, Nz)
    flux_y_full[:, 2:Ny, :] .= flux_y
    flux_y_full[:, 1,    :] .= flux_y_wrap[:, 1, :]
    flux_y_full[:, Ny+1, :] .= flux_y_wrap[:, 1, :]
    flux_y_full[isnan.(flux_y_full)] .= 0.0
    convY = -1 .* diff(flux_y_full, dims=2) ./ Δy_center

    flux_z = -κ .* diff(b, dims=3) ./ reshape(Δz_face[2:Nz], 1, 1, Nz-1)
    flux_z_full = zeros(Nx, Ny, Nz+1)
    flux_z_full[:, :, 2:Nz] .= flux_z
    flux_z_full[isnan.(flux_z_full)] .= 0.0
    convZ = -1 .* diff(flux_z_full, dims=3) ./ Δz_center

    CONV_dV = (convX .+ convY .+ convZ) .* vol
    CONV_dV[.!wet] .= 0.0
    return CONV_dV
end

# ---- grad_b_mag: cell-centered |∇b| (see FBN_PSI.jl discussion) -----------
function grad_b_mag(b)
    gx_face = diff(b, dims=1) ./ reshape(Δx_face[2:Nx], Nx-1, 1, 1)
    gx_ok = .!isnan.(gx_face); gx_face[.!gx_ok] .= 0.0
    gx_sum = zeros(Nx,Ny,Nz); gx_cnt = zeros(Nx,Ny,Nz)
    gx_sum[1:Nx-1,:,:] .+= gx_face; gx_cnt[1:Nx-1,:,:] .+= gx_ok
    gx_sum[2:Nx,  :,:] .+= gx_face; gx_cnt[2:Nx,  :,:] .+= gx_ok
    gx = gx_sum ./ max.(gx_cnt, 1)

    gy_face = diff(b, dims=2) ./ reshape(Δy_face[2:Ny], 1, Ny-1, 1)
    gy_wrap = (b[:,1:1,:] .- b[:,Ny:Ny,:]) ./ Δy_face[1]
    gy_full = cat(gy_wrap, gy_face, gy_wrap, dims=2)
    gy_ok = .!isnan.(gy_full); gy_full[.!gy_ok] .= 0.0
    gy = (gy_full[:,1:Ny,:] .+ gy_full[:,2:Ny+1,:]) ./ max.(gy_ok[:,1:Ny,:] .+ gy_ok[:,2:Ny+1,:], 1)

    gz_face = diff(b, dims=3) ./ reshape(Δz_face[2:Nz], 1, 1, Nz-1)
    gz_ok = .!isnan.(gz_face); gz_face[.!gz_ok] .= 0.0
    gz_sum = zeros(Nx,Ny,Nz); gz_cnt = zeros(Nx,Ny,Nz)
    gz_sum[:,:,1:Nz-1] .+= gz_face; gz_cnt[:,:,1:Nz-1] .+= gz_ok
    gz_sum[:,:,2:Nz]   .+= gz_face; gz_cnt[:,:,2:Nz]   .+= gz_ok
    gz = gz_sum ./ max.(gz_cnt, 1)

    g = sqrt.(gx.^2 .+ gy.^2 .+ gz.^2)
    g[isnan.(g)] .= 0.0
    g[.!wet] .= 0.0
    return g
end

CONV_dV = conv_dV_snapshot(b)
gradmag = grad_b_mag(b)

# ---- F_BN at a single buoyancy value (divergence-theorem cumulative sum,
# restricted to hill1-interior cells) ----------------------------------------
# shell is [b_in, b_out]: the interval where water enters hill1 from the left
# (basin0) but provably can't yet be reaching the right boundary (basin1) --
# confirmed exclusive to hill1 (0% overlap with basin1) in the single-
# snapshot check below.
blo, bhi = b0, b0 + Δb   # b0 = b_in (left/entering), b0+Δb = b_out (right/reaching)

F_BN_edge(bval) = -sum(@view CONV_dV[idxs][b_h1 .< bval])

F_BN_lo  = F_BN_edge(blo)
F_BN_hi  = F_BN_edge(bhi)
F_BN_mid = 0.5 * (F_BN_lo + F_BN_hi)     # edge->bin-center convention

# ---- bounding isopycnal area for the single bin [blo, bhi) -----------------
# the TRUE, tilt-aware surface area of the b=const surface (co-area formula),
# not the flat seafloor-outcrop footprint (Σ Δx·Δy where the bottom cell is
# denser than a threshold) -- those two only agree in the small-aspect-ratio
# limit where isopycnals stay nearly flat, which doesn't hold here given how
# tall these hills are relative to the domain depth.
sel = (b_h1 .>= blo) .& (b_h1 .< bhi)
gv  = vec(gradmag)[idxs]
vv  = vec(vol)[idxs]
A_bound = sum((gv .* vv)[sel]) / Δb

efficiency = F_BN_mid / max(A_bound, eps())

V_region = sum(vv)
V_shell  = sum(vv[sel])

# ---- outcrop area: the classical flat-isopycnal-limit proxy -- Σ Δx·Δy over
# seafloor (x,y) columns whose BOTTOM-MOST wet cell's own buoyancy falls in
# the same bin [blo,bhi). This is directly comparable to A_bound as-is (both
# are true areas -- no extra /Δb needed here, since we're summing literal
# horizontal patches of seafloor, not a co-area density).
ΔA_2d = dropdims(Δx_center .* Δy_center, dims=3)   # [Nx,Ny] cell footprint
ix_hill1 = findall(xi -> -1.35 <= xi < -0.65, x)   # hill1's own x-span

bottom_b = fill(NaN, Nx, Ny)
for iy in 1:Ny, ix in ix_hill1
    k = findfirst(view(wet, ix, iy, :))            # bottom-most WET cell (z ordered bottom->top)
    k === nothing || (bottom_b[ix, iy] = b[ix, iy, k])
end

outcrop_sel = (bottom_b .>= blo) .& (bottom_b .< bhi)   # NaN entries outside ix_hill1 compare false, safely excluded
A_outcrop = sum(ΔA_2d[outcrop_sel])

println()
println("shell = [$blo, $bhi]   (b_in = $b0, b_out = $bhi)")
@printf("F_BN(blo)       = %.4e\n", F_BN_lo)
@printf("F_BN(bhi)       = %.4e\n", F_BN_hi)
@printf("F_BN (binned)   = %.4e   <- averaged onto the bin, like G_mix's edge->center convention\n", F_BN_mid)
@printf("A_bound         = %.4e   <- bounding isopycnal area (true, tilt-aware surface area) in this bin\n", A_bound)
@printf("A_outcrop       = %.4e   <- flat seafloor-outcrop footprint (Σ Δx·Δy), %d columns\n", A_outcrop, count(outcrop_sel))
@printf("A_bound/A_outcrop = %.3f   <- how much the true surface area exceeds the flat-footprint proxy\n",
        A_bound / max(A_outcrop, eps()))
@printf("efficiency      = %.4e   <- F_BN_mid / A_bound\n", efficiency)
println()
@printf("V_region (hill1-interior total) = %.4e\n", V_region)
@printf("V_shell  (cells in this bin)     = %.4e   (%.2f%% of V_region)\n", V_shell, 100*V_shell/V_region)

# ---- does this same [blo, bhi] band show up in the NEIGHBORING basins too?
# same interior-only (z<zBL) cut, same bin -- if a comparable fraction of
# basin0/basin1's own volume sits in this band, the contour isn't unique to
# hill1; it's just water that's also present next door.
mask_basin0 = (reshape(x, Nx,1,1) .>= -1.8)  .& (reshape(x, Nx,1,1) .< -1.35) .& (reshape(z,1,1,Nz) .< zBL)
mask_basin1 = (reshape(x, Nx,1,1) .>= -0.65) .& (reshape(x, Nx,1,1) .< -0.35) .& (reshape(z,1,1,Nz) .< zBL)

for (nm, mask) in (("basin0", mask_basin0), ("hill1", mask_hill1), ("basin1", mask_basin1))
    idxs_nm = findall(vec(mask .& wet))
    bnm = vec(b)[idxs_nm]
    vnm = vec(vol)[idxs_nm]
    sel_nm = (bnm .>= blo) .& (bnm .< bhi)
    Vreg_nm = sum(vnm)
    Vshell_nm = sum(vnm[sel_nm])
    @printf("  %-8s  b-range = (%+.3f, %+.3f)   V_shell/V_region = %.2f%%\n",
            nm, extrema(bnm)..., 100*Vshell_nm/Vreg_nm)
end

# ---- plot: WIDE view spanning basin0 - hill1 - basin1, with the b0/b0+Δb
# shell shaded and region boundaries marked, so you can see with your own eyes
# whether the shell is confined to hill1 or bleeds into its neighbors --------
iy = argmin(abs.(y .- y[end]/4))

b2d = b[:, iy, :]

fig = Figure(size=(900, 550))
ax = Axis(fig[1,1], xlabel="x", ylabel="z",
          title=@sprintf("basin0 | hill1 | basin1  --  shell [%.3f, %.3f]", blo, bhi))
hm = heatmap!(ax, x, z, b2d, colormap=:balance, nan_color=:black)
contourf!(ax, x, z, b2d, levels=[blo, bhi], colormap=[:yellow],
          extendlow=:transparent, extendhigh=:transparent)
contour!(ax, x, z, b2d, levels=[blo, bhi], color=:black, linewidth=2)
hlines!(ax, zBL, color=:cyan, linewidth=2, linestyle=:dash)
vlines!(ax, [-1.35, -0.65, -0.35], color=:magenta, linewidth=1.5, linestyle=:dot)
text!(ax, -1.75, zBL + 0.02, text="zBL", color=:cyan, fontsize=11)
text!(ax, -1.55, -0.05, text="basin0", color=:magenta, fontsize=11)
text!(ax, -1.05, -0.05, text="hill1", color=:magenta, fontsize=11)
text!(ax, -0.55, -0.05, text="basin1", color=:magenta, fontsize=11)
Colorbar(fig[1,2], hm, label="b")
xlims!(ax, -1.8, -0.2); ylims!(ax, -H, 0.0)

outpng = joinpath(plot_dir, "single_contour_efficiency.png")
save(outpng, fig)
println("\nsaved -> $outpng")

# ============================================================================
# DIRECT calculation: the maximum buoyancy in hill1-interior such that
# Ψ_in_right ≈ 0 -- read straight from gmix_CODF.jl's own output, which
# already computed psi_b(x,b,time) at full time resolution. No need to
# recompute the sort-based streamfunction ourselves.
# ============================================================================
gmix_file = joinpath(data_dir, "Gmix_quantile_regions_CODF_3hill_RA1e8_seg1to32.nc")
ds_g = NCDataset(gmix_file)
b_centers_g = Float64.(ds_g["b"][:])
x_g         = Float64.(ds_g["x"][:])
time_g      = Float64.(ds_g["time"][:])

it_g = argmin(abs.(time_g .- t_val))          # match the same physical time as our snapshot
iL   = nearest_xi(x_g, -1.35)                  # hill1's LEFT (upstream, basin0-facing) boundary
iR   = nearest_xi(x_g, -0.65)                  # hill1's RIGHT (downstream, basin1-facing) boundary

# psi_b sign convention (gmix_CODF.jl:395-398):
#   Ψ_in_left  = -ψ_b[iL, :, :]   (positive = water entering the column from the left)
#   Ψ_in_right =  ψ_b[iR, :, :]   (positive = water entering the column from the right)
ψ_in_left  = -Float64.(ds_g["psi_b"][iL, :, it_g])
ψ_in_right =  Float64.(ds_g["psi_b"][iR, :, it_g])
close(ds_g)

# ---- b_in / b_out: EXACT, straight from the raw buoyancy field -----------
# Ψ(b) is exactly zero for any b below a column's TRUE minimum buoyancy --
# trivially, since the integration set {b' < b} is empty. No approximation,
# no need for the CODF's ψ_b curve at all: the CODF b-axis is quantile-spaced
# (equal GLOBAL volume per bin), so it has huge gaps wherever a buoyancy range
# holds little of the domain's TOTAL volume -- exactly the range around
# hill1's boundaries. Scanning that sparse axis for "the last zero sample"
# overshoots badly (verified: it reported -0.6525 here, but the raw column
# never gets colder than -0.355 -- a coarse-axis artifact, not a real
# feature). The column's own raw minimum is the precise answer instead.
ix_L = argmin(abs.(x .- x_g[iL]))
ix_R = argmin(abs.(x .- x_g[iR]))

b_in  = minimum(filter(isfinite, b[ix_L, :, :]))   # coldest water entering hill1 from basin0
b_out = minimum(filter(isfinite, b[ix_R, :, :]))   # coldest water reaching hill1's right boundary

println()
println("hill1 boundary minima (exact, from the raw field), t = $t_val:")
@printf("  b_in  (left boundary,  x=%.3f) = %.4f   <- coldest water entering from basin0\n", x[ix_L], b_in)
@printf("  b_out (right boundary, x=%.3f) = %.4f   <- coldest water reaching basin1\n", x[ix_R], b_out)
@printf("  Δb = b_out - b_in               = %.4f   <- physically-motivated G_mix finite-difference stencil\n", b_out - b_in)
@printf("  (compare to hill1-interior's own top_support ≈ %.4f from the volume argument)\n",
        maximum(filter(isfinite, b_h1)))

# ---- honest line-drawing: don't let `lines!` bridge bins that are wider
# than a real, resolved gap -- that bridge is what visually implies a smooth
# transition/gradual water-mass presence across a span where nothing actually
# exists. Insert a NaN break wherever consecutive bin centers are farther
# apart than gap_thresh; scatter (the real, correctly-computed points) is
# left untouched so no information is lost, only the false interpolation.
function break_at_gaps(b_axis, y; gap_thresh=0.02)
    b_new = Float64[]; y_new = Float64[]
    for i in eachindex(b_axis)
        push!(b_new, b_axis[i]); push!(y_new, y[i])
        if i < length(b_axis) && abs(b_axis[i+1] - b_axis[i]) > gap_thresh
            push!(b_new, NaN); push!(y_new, NaN)
        end
    end
    return b_new, y_new
end

b_left_line,  ψ_left_line  = break_at_gaps(b_centers_g, ψ_in_left)
b_right_line, ψ_right_line = break_at_gaps(b_centers_g, ψ_in_right)

fig_psi = Figure(size=(700, 450))
ax_psi = Axis(fig_psi[1,1], ylabel="b", xlabel="Ψ_in(b)",
              title="hill1: Ψ_in_left (x=$(round(x_g[iL],digits=3))) vs Ψ_in_right (x=$(round(x_g[iR], digits=3)))")
lines!(ax_psi,  ψ_left_line,  b_left_line,  color=:crimson,   linewidth=2, label="Ψ_in_left")
scatter!(ax_psi, ψ_in_left,  b_centers_g, color=:crimson,   markersize=5)
lines!(ax_psi,  ψ_right_line, b_right_line, color=:royalblue, linewidth=2, label="Ψ_in_right")
scatter!(ax_psi, ψ_in_right, b_centers_g, color=:royalblue, markersize=5)
vlines!(ax_psi, 0.0, color=:gray, linewidth=0.8)
hlines!(ax_psi, b_in,  color=:crimson,   linewidth=2, linestyle=:dash, label="b_in (raw min) = $(round(b_in, digits=3))")
hlines!(ax_psi, b_out, color=:darkorange, linewidth=2, linestyle=:dash, label="b_out (raw min) = $(round(b_out, digits=3))")
axislegend(ax_psi, position=:lb, labelsize=9)
outpng_psi = joinpath(plot_dir, "psi_in_right_zero_crossing.png")
save(outpng_psi, fig_psi)
println("saved -> $outpng_psi")
println("(the scatter dots are the CODF file's actual b-axis sample points -- look for the gap")
println(" around b_in/b_out: that gap is why the connected LINE looked like it stayed at zero")
println(" much further down than the water actually does.)")

# ============================================================================
# Robustness check: do b_in / b_out hold across the WHOLE final segment, not
# just this one snapshot? At each t, b_in(t)/b_out(t) IS the column's true
# minimum -- "no water colder than B exists at time t" is only true for
# B <= (that timestep's own minimum). For a FIXED B to make that claim hold
# at EVERY t in the window, B must be <= the SMALLEST (coldest) of the
# per-timestep minima -- i.e. the conservative choice is MINIMUM over time,
# not maximum. (Picking the max, as an earlier version of this comment
# wrongly claimed, actually picks a B that's too WARM: some timestep's real
# minimum dips below it, so real water ends up colder than "the boundary" --
# exactly the basin1/hill2 contamination this was caught by.)
# Scoped to segment 32 only (not the full 32-segment run): the earlier
# segments include spin-up transients that aren't representative of the
# equilibrated state this whole exercise cares about.
#
# Same exact-minimum approach as above, NOT the axis-scan -- pull just the
# two boundary COLUMNS across all of segment 32 directly from disk (a cheap
# NetCDF slice, not the full 4D cube) and take each timestep's own minimum.
# ============================================================================
ds_seg = NCDataset(joinpath(data_dir, "buoyancy_seg32.nc"))
t_seg32 = ds_seg["time"][:]
b_col_L_seg32 = Array{Float64}(ds_seg["b"][ix_L, :, :, :])   # [Ny, Nz, Nt]
b_col_R_seg32 = Array{Float64}(ds_seg["b"][ix_R, :, :, :])   # [Ny, Nz, Nt]
close(ds_seg)

wet_col_L = wet[ix_L, :, :]
wet_col_R = wet[ix_R, :, :]
nt_seg = length(t_seg32)

b_in_t  = [minimum(@view(b_col_L_seg32[:, :, k])[wet_col_L]) for k in 1:nt_seg]
b_out_t = [minimum(@view(b_col_R_seg32[:, :, k])[wet_col_R]) for k in 1:nt_seg]

println()
println("b_in(t) / b_out(t) across all $nt_seg snapshots of segment 32 (t = $(t_seg32[1]) to $(t_seg32[end])):")
@printf("  b_in   min/median/max = %.4f / %.4f / %.4f\n", minimum(b_in_t), median(b_in_t), maximum(b_in_t))
@printf("  b_out  min/median/max = %.4f / %.4f / %.4f\n", minimum(b_out_t), median(b_out_t), maximum(b_out_t))
@printf("  conservative (fixed-for-all-t) choices: b_in = %.4f, b_out = %.4f, Δb = %.4f\n",
        minimum(b_in_t), minimum(b_out_t), minimum(b_out_t) - minimum(b_in_t))

fig_bt = Figure(size=(700, 350))
ax_bt = Axis(fig_bt[1,1], xlabel="time", ylabel="b",
             title="b_in(t) / b_out(t) across segment 32 (t=$(round(t_seg32[1],digits=1)) to $(round(t_seg32[end],digits=1)))")
lines!(ax_bt, t_seg32, b_in_t,  color=:crimson,    linewidth=2, label="b_in(t)")
lines!(ax_bt, t_seg32, b_out_t, color=:darkorange, linewidth=2, label="b_out(t)")
hlines!(ax_bt, minimum(b_in_t),  color=:crimson,    linestyle=:dash, linewidth=1.2,
        label="b_in conservative (min) = $(round(minimum(b_in_t), digits=4))")
hlines!(ax_bt, minimum(b_out_t), color=:darkorange, linestyle=:dash, linewidth=1.2,
        label="b_out conservative (min) = $(round(minimum(b_out_t), digits=4))")
axislegend(ax_bt, position=:rb)
outpng_bt = joinpath(plot_dir, "b_in_cold_over_time.png")
save(outpng_bt, fig_bt)
println("saved -> $outpng_bt")

# ============================================================================
# G_mix / dM/dt / Ψ backsolve: dM/dt = G_mix + Ψ + Gsurf, checked at b_out.
#
# b_in falls INSIDE the CODF axis's giant first bin (confirmed earlier), so
# anything read off the saved Gmix_col_hill1/psi_col_hill1 arrays there would
# just be that bin's smeared aggregate over [-0.95,-0.354], not a value
# specific to hill1's real transition. b_out sits in a normal, well-resolved
# bin, so it's the point we trust for reading the saved CODF arrays.
# ============================================================================

# linear interpolation on a sorted (possibly non-uniform) axis, matching
# fbn_psi_comparison.jl:550-557
function interp_at(xs, ys, x0)
    x0 <= xs[1] && return ys[1]
    x0 >= xs[end] && return ys[end]
    i = searchsortedlast(xs, x0)
    i = clamp(i, 1, length(xs) - 1)
    frac = (x0 - xs[i]) / (xs[i+1] - xs[i])
    return ys[i] + frac * (ys[i+1] - ys[i])
end

# ---- determine the CODF interval bounds up front, so F_BN averaging (A)
# and the measured terms (B) use the exact same window ---------------------
ds_gK = NCDataset(gmix_file)
t_start_int = ds_gK["t_start"][:]
t_end_int   = ds_gK["t_end"][:]
close(ds_gK)
kk = length(t_start_int)   # last interval = equilibrium window
println()
println("using CODF interval $kk: t = $(t_start_int[kk]) to $(t_end_int[kk]) (matches segment 32's own equilibrium window)")

# ---- (A) independent G_mix estimate: F_BN finite-difference over [b_in,
# b_out], AVERAGED over every snapshot in the SAME interval gmix_CODF.jl
# used for Gmix_int/psi_int/dMdt -- matches how Gmix_int itself was built
# (snapshot-by-snapshot, then nanmean, NOT from a time-averaged buoyancy
# field), so this is now a fair, apples-to-apples comparison instead of one
# instant vs a 9.7-time-unit mean.
interval_idx = findall(t -> t_start_int[kk] - 1e-9 <= t <= t_end_int[kk] + 1e-9, t_seg32)
println("averaging F_BN over $(length(interval_idx)) snapshots (t=$(t_seg32[interval_idx[1]]) to $(t_seg32[interval_idx[end]]))")

ds_segA = NCDataset(joinpath(data_dir, "buoyancy_seg32.nc"))
F_BN_lo_sum = 0.0
F_BN_hi_sum = 0.0
for k in interval_idx
    global F_BN_lo_sum, F_BN_hi_sum   # top-level `for` needs this to mutate outer scalars
    bk = Array{Float64}(ds_segA["b"][:, :, :, k])
    bk[.!wet] .= NaN
    CONV_dV_k = conv_dV_snapshot(bk)
    bk_h1 = vec(bk)[idxs]
    F_BN_lo_sum += -sum(@view CONV_dV_k[idxs][bk_h1 .< blo])
    F_BN_hi_sum += -sum(@view CONV_dV_k[idxs][bk_h1 .< bhi])
end
close(ds_segA)

n_int_snaps  = length(interval_idx)
F_BN_lo_mean = F_BN_lo_sum / n_int_snaps
F_BN_hi_mean = F_BN_hi_sum / n_int_snaps
Gmix_from_FBN = (F_BN_hi_mean - F_BN_lo_mean) / (bhi - blo)

println()
println("G_mix estimate via F_BN finite-difference, interval-averaged (exact b_in/b_out, no axis dependency):")
@printf("  F_BN_lo (mean) = %.4e,  F_BN_hi (mean) = %.4e\n", F_BN_lo_mean, F_BN_hi_mean)
@printf("  Gmix_from_FBN = ΔF_BN/Δb = %.4e\n", Gmix_from_FBN)

# A secant over [blo,bhi] approximates the derivative at the interval's
# MIDPOINT, not at either endpoint -- same edge->center logic as F_BN_mid
# above, and the same convention gmix_CODF.jl itself uses (b_centers =
# midpoint of consecutive edges). Comparing Gmix_from_FBN against a
# pointwise value AT bhi (as the very first version of this comparison did)
# was comparing two different things whenever G_mix isn't flat across the
# interval -- which we now know it isn't.
bmid = 0.5 * (blo + bhi)
println("  (a secant over [blo,bhi] approximates the derivative at the MIDPOINT, b_mid=$bmid -- not at bhi)")

# ---- (B) directly-measured G_mix / Ψ / Gsurf / dM/dt / R, ALL already
# saved as interval-mean quantities in the CODF file -- gmix_CODF.jl already
# computed dMdt_hill1 as the exact endpoint finite-difference over each
# interval (Step 5), so there's no need to recompute it from M by hand. Read
# at b_out via linear interpolation on b_centers_g -- b_in still falls
# INSIDE the coarse axis's giant first bin, so it's excluded here (see the
# earlier naive-interp check for why).
ds_gB = NCDataset(gmix_file)
b_bin = ds_gB["b"][:]
Gmix_int_hill1_k  = Float64.(ds_gB["Gmix_int_hill1"][:, kk])
psi_int_hill1_k   = Float64.(ds_gB["psi_int_hill1"][:, kk])
Gsurf_int_hill1_k = Float64.(ds_gB["Gsurf_int_hill1"][:, kk])
dMdt_hill1_k      = Float64.(ds_gB["dMdt_hill1"][:, kk])
R_hill1_k         = Float64.(ds_gB["R_hill1"][:, kk])
close(ds_gB)

println()
println("using CODF interval $kk: t = $(t_start_int[kk]) to $(t_end_int[kk]) (matches segment 32's own equilibrium window)")

Gmix_meas  = interp_at(b_centers_g, Gmix_int_hill1_k,  bmid)
psi_meas   = interp_at(b_centers_g, psi_int_hill1_k,   bmid)
Gsurf_meas = interp_at(b_centers_g, Gsurf_int_hill1_k, bmid)
dMdt_bout  = interp_at(b_centers_g, dMdt_hill1_k,      bmid)
R_meas     = interp_at(b_centers_g, R_hill1_k,         bmid)

println()
println("directly-measured, interval-mean budget terms at b_out=$bhi (well-resolved on the CODF axis):")
@printf("  Gmix_int_hill1(b_mid)  = %.4e\n", Gmix_meas)
@printf("  psi_int_hill1(b_mid)   = %.4e\n", psi_meas)
@printf("  Gsurf_int_hill1(b_mid) = %.4e   <- should be ~0 (sub-boundary-layer, no surface forcing)\n", Gsurf_meas)
@printf("  dMdt_hill1(b_mid)      = %.4e   <- already computed by gmix_CODF.jl, not recomputed here\n", dMdt_bout)

# for honesty: show what the SAME read gives at b_in, to make the coarse-bin
# contamination visible rather than just asserted
Gmix_meas_bin_naive = interp_at(b_centers_g, Gmix_int_hill1_k, blo)
println()
println("(for comparison only -- b_in=$blo falls INSIDE the coarse axis's giant first bin,")
println(" so this is smeared over the whole [-0.95,-0.354] range, not specific to hill1's")
println(" real transition. Included to make the contamination visible, not to be trusted.)")
@printf("  Gmix_int_hill1(b_in, naive interp) = %.4e\n", Gmix_meas_bin_naive)

# ---- (C) the actual budget closure check, cross-checked against gmix_CODF.jl's
# OWN already-saved residual R_hill1 -- since interp_at is LINEAR, and R is a
# linear combination of the other four arrays on the SAME b_centers_g axis,
# interpolating R directly must equal our own hand-computed residual to
# floating-point precision. This is a pure arithmetic identity check (confirms
# we're reading/interpolating correctly) -- it does NOT independently confirm
# the physics closes; that's what the residual's actual SIZE tells us.
budget_rhs = Gmix_meas + psi_meas + Gsurf_meas
R_handcomputed = dMdt_bout - budget_rhs
println()
println("budget closure check at b_out=$bhi, using gmix_CODF.jl's own dMdt_hill1:")
@printf("  dM/dt (measured)              = %.4e\n", dMdt_bout)
@printf("  Gmix + Ψ + Gsurf (measured)   = %.4e\n", budget_rhs)
@printf("  R (hand-computed)             = %.4e\n", R_handcomputed)
@printf("  R_hill1 (saved, interpolated) = %.4e   <- should match R (hand-computed) to ~machine precision\n", R_meas)
@printf("  %%diff (dMdt vs Gmix+Ψ+Gsurf)  = %+.1f%%\n", 100 * (dMdt_bout - budget_rhs) / abs(budget_rhs))
@printf("  Gsurf contamination            = %.2f%% of |Gmix+Ψ+Gsurf|\n", 100 * abs(Gsurf_meas) / abs(budget_rhs))

dMdt_bout
budget_rhs .+ R_meas
# ---- (D) both directions of the closure, per the original request ---------
println()
println("both directions of the closure, per the original request:")
Gmix_backsolve = dMdt_bout - psi_meas - Gsurf_meas
psi_backsolve  = dMdt_bout - Gmix_meas - Gsurf_meas
@printf("  G_mix_backsolve = dMdt - Ψ - Gsurf    = %.4e   (measured Gmix_int = %.4e, %%diff = %+.1f%%)\n",
        Gmix_backsolve, Gmix_meas, 100*(Gmix_backsolve-Gmix_meas)/abs(Gmix_meas))
@printf("  Ψ_backsolve     = dMdt - Gmix - Gsurf = %.4e   (measured Ψ_int   = %.4e, %%diff = %+.1f%%)\n",
        psi_backsolve, psi_meas, 100*(psi_backsolve-psi_meas)/abs(psi_meas))
@printf("  Gmix_from_FBN (independent, part A, %d-snapshot interval mean) = %.4e   (%%diff vs Gmix_int = %+.1f%%)\n",
        n_int_snaps, Gmix_from_FBN, 100*(Gmix_from_FBN-Gmix_meas)/abs(Gmix_meas))
println("  (now apples-to-apples: both Gmix_from_FBN and Gmix_int are means over the same")
println("   $(round(t_end_int[kk]-t_start_int[kk], digits=1))-time-unit interval, t=$(t_start_int[kk]) to $(t_end_int[kk]).)")


#so the problem is that gmix from gmix_CODF and gmix calculated with FBN are not the same
# even after the averaging over the interval

fig=Figure()
ax = Axis(fig[1,1], title = "hill 1 gmix versus buoyancy", xlabel = "gmix", ylabel = "b")
lines!(ax, Gmix_int_hill1_k, b_bin)
hlines!(ax, blo, linestyle =:dash, color=:orange)
hlines!(ax, bhi, linestyle =:dash, color=:red)
ylims!(ax, -0.4, -0.3)
fig

outpng = joinpath(plot_dir, "zoom_in_gmix.png")
save(outpng, fig)
println("\nsaved -> $outpng")

# ============================================================================
# x-y slice (fixed z), instead of the x-z slices used everywhere else: shows
# how the hill's 3D shape -- specifically the channel notch that shortens
# each hill near y=0 -- shapes where the [blo,bhi] shell actually sits.
# Picked near the bottom (z close to -0.99), right around where the earlier
# diagnostic found hill1's own coldest cell (z=-0.9928, y=-0.0195 -- almost
# exactly the channel).
# ============================================================================
iz = argmin(abs.(z .- (-0.6)))
b_xy = b[:, :, iz]
dry_xy = .!wet[:, :, iz]

println()
println("x-y slice at z=$(z[iz]) (closest to -0.99)")

fig_xy = Figure(size=(900, 500))
ax_xy = Axis(fig_xy[1,1], xlabel="x", ylabel="y",
             title="hill1 region, x-y slice at z=$(round(z[iz], digits=3))  --  shell [$blo, $bhi]")
hm_xy = heatmap!(ax_xy, x, y, b_xy, colormap=:balance, nan_color=:transparent)
heatmap!(ax_xy, x, y, dry_xy, colormap=cgrad([RGBAf(0.15,0.15,0.15,0.0), RGBAf(0.15,0.15,0.15,1.0)]), colorrange=(0,1))
contourf!(ax_xy, x, y, b_xy, levels=[blo, bhi], colormap=[:yellow],
          extendlow=:transparent, extendhigh=:transparent)
contour!(ax_xy, x, y, b_xy, levels=[blo, bhi], color=:black, linewidth=1.5)
vlines!(ax_xy, [-1.35, -0.65, -0.35], color=:magenta, linewidth=1.5, linestyle=:dot)
Colorbar(fig_xy[1,2], hm_xy, label="b")
xlims!(ax_xy, -1.8, -0.2)

outpng_xy = joinpath(plot_dir, "single_contour_xy_slice.png")
save(outpng_xy, fig_xy)
println("saved -> $outpng_xy")