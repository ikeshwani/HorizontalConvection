# fbn_psi_bresolved.jl
#
# Buoyancy-resolved Bryden & Nurser (2003) bulk-formula test, replacing the
# earlier single-number bar chart (fbn_psi_comparison.jl's fbn_psi_comparison.png)
# with the full curve comparison the exact identity actually licenses:
#
#   F_BN(b) ≡ ∫_{A(b)} (-κ∇b)·n̂ dS = -M(b)      (gmix_region's cumulative M)
#   G_mix(b) = ∂F_BN/∂b                          exact, divergence theorem + FTC
#   quasi-steady, away from the surface:  G_mix(b) ≈ -Ψ(b)
#   ⇒ -G_mix(b) is the transport INFERENCE at every b, not just one bulk number
#     from a secant -- so we plot the full -G_mix(b) curve against the directly
#     measured Ψ(b), plus the secant-derived Ψ_est as a horizontal reference
#     line spanning [b_in, b_out].
#
# Scope: Ra=1e8, 3-hill (Seafloor) only (Ra=1e6 has no gmix file yet -- out of
# scope). Data source: the NON-rebinned CODF file (Gmix_quantile_regions_
# CODF_3hill_RA1e8_seg1to34.nc, from gmix_CODF.jl), NOT the actively-rebinned
# one gmix_budget_combined_seafloor.jl uses. Switched deliberately, after
# checking both files against Recipe B's b_in/b_out for each hill:
#   - gmix_CODF.jl freezes ONE buoyancy axis (built once from a single late
#     snapshot) for the whole run and bins every snapshot directly against
#     it -- no per-chunk local-axis-then-interpolate step at all, so there's
#     no source for the fresh-vs-reconstructed mismatch that the actively-
#     rebinned file showed (~23% for hill1, traced to exactly that
#     interpolation step). Confirmed empirically: hill1's b_in/b_out span 3
#     resolved b_centers on this file (0 on the rebinned one), hill2 spans
#     11, hill3 spans 12 -- all land in real, separate bins rather than one
#     hill's whole range being swallowed into a single sparse-tail bin.
#   - The one caveat that does NOT go away: hill1's b_in itself still falls
#     inside a genuinely huge bin ([-0.9512,-0.3597], width 0.59) at the very
#     bottom of the axis -- there just isn't much total domain volume that
#     dense, on ANY quantile axis. So F_BN(b_in) is still an interpolated
#     estimate there, but the curve BETWEEN b_in and b_out is no longer
#     entirely blank the way it was on the rebinned file.
#   - No per-chunk local axis also means there's no "which chunk does this
#     window straddle" concern, so the window is a plain [990.0, 1000.0]
#     (matching the budget figures) rather than a file-specific chunk
#     boundary.
#
# b_in / b_out ("Recipe B", from region_contours.jl / single_contour_efficiency.jl
# -- NOT fbn_psi_comparison.jl's (b0_hill, b1_down=downstream-quantile) recipe,
# which mixed one rigorous endpoint with one tuned/arbitrary one):
#   Ψ(b) is EXACTLY zero for any b colder than a boundary column's own true
#   minimum buoyancy -- trivially, since the integration set {b' < b} is empty
#   there. So at each of the 4 basin/hill interfaces (hill1-left, hill1-right
#   = hill2-left, hill2-right = hill3-left, hill3-right), the conservative
#   (minimum-over-every-snapshot-in-the-window) column minimum is the exact
#   buoyancy below which transport through THAT face is provably zero. No
#   quantile, no free parameter. hill_n's b_in/b_out are its own two interfaces
#   (chained: hill_n's right edge is hill_(n+1)'s left edge).
#
# Because b_in is a boundary COLUMN's own minimum (not the hill's overall
# minimum the way fbn_psi_comparison.jl's b0_hill was BY CONSTRUCTION),
# F_BN(b_in) is not guaranteed to be exactly 0 here -- it's checked and
# reported honestly below, and the secant is drawn as the literal two-point
# line (F_BN(b_in), b_in) -> (F_BN(b_out), b_out), not anchored at zero.
#
# -G_mix(b)/Ψ(b) curves themselves are still the FULL hill column (interior +
# bl_hill_n combined, i.e. Gmix_col_hill$n/psi_col_hill$n -- "the same way
# G_cols is assembled" in gmix_CODF.jl), unaffected by the b_in/b_out change.

using TopographicHorizontalConvection   # boundary_layer_depth, precompute_regions, nearest_xi
using NCDatasets
using CairoMakie
using NaNStatistics
using Statistics
using Printf

# ---- config ----
data_dir  = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/"
plot_dir  = joinpath(data_dir, "figures")
mkpath(plot_dir)
gmix_file = joinpath(data_dir, "Gmix_quantile_regions_CODF_3hill_RA1e8_seg1to34.nc")
segments  = 1:34

t_lo, t_hi = 990.0, 1000.0   # no per-chunk local axis on this file, so no straddling concern (see header comment)
hills      = 1:3

col_bounds = [
    ("basin0", -1.8,  -1.35), ("hill1", -1.35, -0.65), ("basin1", -0.65, -0.35),
    ("hill2",  -0.35,  0.35), ("basin2",  0.35,  0.65), ("hill3",   0.65,  1.35),
    ("basin3",  1.35,  Inf),
]
hill_names  = ["hill$n" for n in hills]
hill_bounds = [(nm, xlo, xhi) for (nm, xlo, xhi) in col_bounds if nm in hill_names]

# ---- grid + metadata (mirrors gmix_CODF.jl / fbn_psi_comparison.jl Pass 0) ----
ds1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
Ra, Pr, b★, H = ds1.attrib["Ra"], ds1.attrib["Pr"], ds1.attrib["b★"], ds1.attrib["H"]
Lx = ds1.attrib["Lx"]

x = ds1["x_caa"][:]
z = ds1["z_aac"][:]

Δx_face = ds1["Δx_faa"][:]
Δy_face = ds1["Δy_afa"][:]
Δz_face = ds1["Δz_aaf"][:]

Δx_center = reshape(ds1["Δx_caa"][:], Nx, 1, 1)
Δy_center = reshape(ds1["Δy_aca"][:], 1, Ny, 1)
Δz_center = reshape(ds1["Δz_aac"][:], 1, 1, Nz)
ΔA_2d     = dropdims(Δx_center .* Δy_center, dims=3)

vol = Δx_center .* Δy_center .* Δz_center
ν = sqrt(Pr * b★ * H^3 / Ra)
κ = ν / Pr
Ly = sum(Float64.(ds1["Δy_aca"][:]))
κLy = κ * Ly   # nondimensionalization divisor for every volume-flux quantity below (b -> b/b★ separately)

wet = ds1["b"][:, :, :, 2] .!= 0
close(ds1)

# ---- region masks for the fresh F_BN pass: hill{1,2,3} interior + bl_hill{1,2,3} ----
zBL = boundary_layer_depth(Lx, Ra)
X = reshape(x, :, 1, 1); Z = reshape(z, 1, 1, :)
region_masks = vcat(
    [("hill$n",    (X .>= xlo) .& (X .< xhi) .& (Z .< zBL)) for (n, (nm, xlo, xhi)) in enumerate(hill_bounds)],
    [("bl_hill$n", (X .>= xlo) .& (X .< xhi) .& (Z .> zBL)) for (n, (nm, xlo, xhi)) in enumerate(hill_bounds)],
)
region_precomp = precompute_regions(region_masks, ΔA_2d, wet)
region_names   = [r.name for r in region_precomp]

# ---- Recipe B interfaces: hill1-left, hill1-right(=hill2-left), hill2-right(=hill3-left), hill3-right ----
interface_xs = [hill_bounds[1][2]]                              # hill1's own left edge
append!(interface_xs, [xhi for (nm, xlo, xhi) in hill_bounds])  # every hill's own right edge
@printf("Recipe B interfaces (x): %s\n", interface_xs)
interface_ix = [nearest_xi(x, xt) for xt in interface_xs]

# ---- pull the buoyancy axis from the CODF file; explicit window on OUR OWN terms ----
ds_g = NCDataset(gmix_file)
b_edges   = Float64.(ds_g["b_edges"][:])
b_centers = Float64.(ds_g["b"][:])
time_g    = Float64.(ds_g["time"][:])
i_lo = nearest_xi(time_g, t_lo)
i_hi = nearest_xi(time_g, t_hi)
rng  = i_lo:i_hi
@printf("equilibrium window: t = %.4f -> %.4f  (global time indices %d:%d, %d snapshots)\n",
        time_g[i_lo], time_g[i_hi], i_lo, i_hi, length(rng))

# ---- fresh F_BN(b) + Recipe B boundary-column conservative minima, in ONE
# pass over the raw buoyancy field (same snapshots serve both purposes) ------
function M_region(b, CONV_dV, idxs, b_edges)
    bg = vec(b)[idxs]
    cg = vec(CONV_dV)[idxs]
    M  = zeros(length(b_edges))
    for n in eachindex(b_edges)
        M[n] = sum(@view cg[bg .< b_edges[n]])
    end
    return M
end

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

function segments_covering(data_dir, segments, t0, t1)
    hits = Tuple{Int, UnitRange{Int}}[]
    for s in segments
        bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
        t_seg = Float64.(bfile["time"][:])
        close(bfile)
        idx = findall((t_seg .>= t0 - 1e-9) .& (t_seg .<= t1 + 1e-9))
        isempty(idx) || push!(hits, (s, idx[1]:idx[end]))
    end
    return hits
end

seg_hits = segments_covering(data_dir, segments, t_lo, t_hi)
@printf("equilibrium window covered by segment(s): %s\n", [s for (s, _) in seg_hits])

M_fresh_sum   = Dict(name => zeros(Float64, length(b_edges)) for name in region_names)
interface_bmin = fill(Inf, length(interface_xs))   # conservative (min-over-window) column minimum, per interface
n_snap_fresh  = 0

for (s, local_range) in seg_hits
    bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
    b_seg = Array{Float64}(bfile["b"][:, :, :, local_range])
    close(bfile)
    nt = size(b_seg, 4)
    @printf("  seg %d: loading %d snapshot(s) for fresh F_BN + boundary minima...\n", s, nt)
    for ti in 1:nt
        b = b_seg[:, :, :, ti]
        b[.!wet] .= NaN
        CONV_dV = conv_dV_snapshot(b)
        for r in region_precomp
            M_fresh_sum[r.name] .+= M_region(b, CONV_dV, r.idxs, b_edges)
        end
        for (k, ix) in enumerate(interface_ix)
            col = @view b[ix, :, :]
            wet_col = @view wet[ix, :, :]
            m = minimum(@view col[wet_col])
            m < interface_bmin[k] && (interface_bmin[k] = m)
        end
        global n_snap_fresh += 1
    end
    b_seg = nothing; GC.gc()
end

M_fresh    = Dict(name => M_fresh_sum[name] ./ n_snap_fresh for name in region_names)
F_BN_fresh = Dict(name => -M_fresh[name] for name in region_names)
@printf("fresh F_BN averaged over %d snapshot(s)\n", n_snap_fresh)

println("\nRecipe B boundary-column conservative minima (min over window, all wet z at that x):")
for (k, xt) in enumerate(interface_xs)
    @printf("  interface %d (x=%.4f, nearest x=%.4f): b_conservative = %.4f\n",
            k, xt, x[interface_ix[k]], interface_bmin[k])
end

hill_contours = Dict(n => (b_in=interface_bmin[n], b_out=interface_bmin[n+1]) for n in hills)
println("\nchained b_in/b_out per hill (Recipe B):")
for n in hills
    c = hill_contours[n]
    @printf("  hill%d  b_in = %+.4f   b_out = %+.4f   Δb = %.4f\n", n, c.b_in, c.b_out, c.b_out - c.b_in)
end

# ---- reconstruct F_BN from the already-saved Gmix_$(name)(b,t), same window ----
function reconstruct_M(Gmix_mean, b_edges)
    n = length(b_edges)
    M = zeros(Float64, n)
    for k in 1:n-1
        M[k+1] = M[k] - Gmix_mean[k] * (b_edges[k+1] - b_edges[k])
    end
    return M
end

M_recon = Dict{String, Vector{Float64}}()
for name in region_names
    Gmix_mean = nanmean(Float64.(ds_g["Gmix_$(name)"][:, rng]), dims=2)[:]
    M_recon[name] = reconstruct_M(Gmix_mean, b_edges)
end
F_BN_recon = Dict(name => -M_recon[name] for name in region_names)

println("\nfresh vs reconstructed F_BN agreement (per region, max|Δ| over b_edges) -- required sanity check:")
for name in region_names
    err   = maximum(abs.(F_BN_fresh[name] .- F_BN_recon[name]))
    scale = max(maximum(abs.(F_BN_fresh[name])), 1e-30)
    @printf("  %-10s  max|Δ| = %.3e   (%.2f%% of max|F_BN|)\n", name, err, 100*err/scale)
end

# ---- nondimensionalize: b -> b/b★, every volume-flux term -> /(κ·Ly) --------
# Done here, AFTER the fresh-vs-reconstructed sanity check above (which
# depends on matching raw b_edges against raw Gmix_mean -- an internal
# reconstruct_M integration step that would silently break if either side got
# rescaled first). From this point on, b_edges/b_centers/hill_contours/
# F_BN_fresh are reassigned in place to their nondim values, so every
# downstream use (plots, tables) is automatically nondim without having to
# track down each call site -- same "convert once, use everywhere after"
# pattern gmix_scratch_plots.jl already uses for its own d̂.
b_edges       = b_edges ./ b★
b_centers     = b_centers ./ b★
hill_contours = Dict(n => (b_in=interface_bmin[n]/b★, b_out=interface_bmin[n+1]/b★) for n in hills)
F_BN_fresh    = Dict(name => F_BN_fresh[name] ./ κLy for name in region_names)

# ---- combine interior + bl_ into full-depth hill columns ----
F_BN_col_fresh = Dict(n => F_BN_fresh["hill$n"] .+ F_BN_fresh["bl_hill$n"] for n in hills)

function interp_at(xs, ys, x0)
    x0 <= xs[1] && return ys[1]
    x0 >= xs[end] && return ys[end]
    i = searchsortedlast(xs, x0)
    i = clamp(i, 1, length(xs)-1)
    frac = (x0 - xs[i]) / (xs[i+1] - xs[i])
    return ys[i] + frac * (ys[i+1] - ys[i])
end

println("\nb_in sanity check: F_BN(b_in) -- NOT guaranteed exactly 0 under Recipe B (b_in is the")
println("boundary COLUMN's own minimum, not the hill's overall minimum), reported honestly:")
for n in hills
    b_in = hill_contours[n].b_in
    fbn0  = interp_at(b_edges, F_BN_col_fresh[n], b_in)
    scale = maximum(abs.(F_BN_col_fresh[n]))
    @printf("  hill%d  b_in = %+.4f   F_BN(b_in) = %+.3e   (%.2f%% of max|F_BN| = %.3e)\n",
            n, b_in, fbn0, 100*abs(fbn0)/max(scale, 1e-30), scale)
end

# ---- G_mix(b), Ψ(b), dMdt(b): buoyancy-resolved, window-averaged, from the
# already-saved per-snapshot FULL-COLUMN (interior+bl_hill_n combined) arrays
# -- "the same way G_cols is assembled" in gmix_CODF.jl ----------------------
neg_Gmix   = Dict{Int, Vector{Float64}}()   # -G_mix(b), on b_centers
Ψ_measured = Dict{Int, Vector{Float64}}()   # measured Ψ(b), on b_centers
dMdt_hill  = Dict{Int, Vector{Float64}}()   # dM/dt(b), on b_centers
for n in hills
    Gmix_col = Float64.(ds_g["Gmix_col_hill$n"][:, rng])
    psi_col  = Float64.(ds_g["psi_col_hill$n"][:, rng])
    M_col    = Float64.(ds_g["M_hill$n"][:, rng]) .+ Float64.(ds_g["M_bl_hill$n"][:, rng])

    neg_Gmix[n]   = -vec(mean(Gmix_col, dims=2)) ./ κLy
    Ψ_measured[n] =  vec(mean(psi_col,  dims=2)) ./ κLy
    dMdt_hill[n]  = ((M_col[:, end] .- M_col[:, 1]) ./ (time_g[i_hi] - time_g[i_lo])) ./ κLy
end

# ---- Ψ_est: B&N secant estimate (literal two-point line, NOT anchored at
# zero -- see header comment), and sign check ---------------------------------
# Ψ_est is not just "an approximation" of a point value: since G_mix = dF_BN/db
# exactly, the secant slope (F_BN(b_out)-F_BN(b_in))/(b_out-b_in) is EXACTLY
# the b-average of G_mix(b) over [b_in,b_out] (fundamental theorem of
# calculus), so Ψ_est = ⟨-G_mix(b)⟩ over that same range -- a true average,
# not a single-point value. For the comparison against Ψ_measured to be
# apples-to-apples with what Ψ_est actually represents, Ψ_measured should
# ALSO be the range-average ⟨Ψ_measured(b)⟩ over [b_in,b_out], not a value at
# one point. We fall back to interpolating at bmid only if the range turns
# out to contain no resolved b_centers at all (can happen on a coarser axis;
# doesn't happen on this file for any of the 3 hills).
println("\nΨ_est (bulk, F_BN secant, Recipe B b_in/b_out) vs Ψ_measured (direct, buoyancy-resolved MEAN over [b_in,b_out]):")
Ψ_est     = Dict{Int, Float64}()
Ψ_meas    = Dict{Int, Float64}()
dMdt_mean = Dict{Int, Float64}()
for n in hills
    b_in, b_out = hill_contours[n].b_in, hill_contours[n].b_out
    F0 = interp_at(b_edges, F_BN_col_fresh[n], b_in)
    F1 = interp_at(b_edges, F_BN_col_fresh[n], b_out)
    Ψ_est[n] = -(F1 - F0) / (b_out - b_in)

    lo, hi = minmax(b_in, b_out)
    sel = findall(lo .<= b_centers .<= hi)
    if isempty(sel)
        bmid = 0.5 * (b_in + b_out)
        Ψ_meas[n]    = interp_at(b_centers, Ψ_measured[n], bmid)
        dMdt_mean[n] = interp_at(b_centers, dMdt_hill[n],  bmid)
        @printf("  hill%d  (no resolved b_centers in range -- fell back to bmid=%.4f interpolation)\n", n, bmid)
    else
        Ψ_meas[n]    = nanmean(Ψ_measured[n][sel])
        dMdt_mean[n] = nanmean(dMdt_hill[n][sel])
    end
    sign_match = sign(Ψ_est[n]) == sign(Ψ_meas[n])

    @printf("  hill%d  Ψ_est = %+.4e   ⟨Ψ_measured⟩ = %+.4e   sign match: %s   (%d b-bins in [%.3f,%.3f])\n",
            n, Ψ_est[n], Ψ_meas[n], sign_match, length(sel), lo, hi)

    if !sign_match
        println("\n  ⚠ SIGN MISMATCH for hill$n -- stopping rather than absorbing silently.")
        @printf("    b_in = %+.4f  b_out = %+.4f  F0 = %+.4e  F1 = %+.4e\n", b_in, b_out, F0, F1)
        @printf("    Ψ_est = %+.4e   ⟨Ψ_measured⟩ = %+.4e\n", Ψ_est[n], Ψ_meas[n])
        error("Ψ_est / Ψ_measured sign mismatch for hill$n -- check b_in/b_out ordering or a face-sign convention before trusting these numbers.")
    end
end

# ---- quantify agreement over [b_in, b_out]: pointwise curve RMS, AND the
# closure check ΔΨ = Ψ_est - ⟨Ψ_measured⟩ vs -⟨∂M/∂t⟩ (see comment above --
# this follows directly from ∂M/∂t = G_mix + Ψ + G_surface + R, G_surface≈0,
# averaged over [b_in,b_out]) --------------------------------------------------
println("\nagreement over [b_in, b_out] (per hill):")
ΔΨ = Dict{Int, Float64}()
for n in hills
    b_in, b_out = hill_contours[n].b_in, hill_contours[n].b_out
    lo, hi = minmax(b_in, b_out)
    sel = findall(lo .<= b_centers .<= hi)

    ΔΨ[n] = Ψ_est[n] - Ψ_meas[n]
    @printf("  hill%d  ΔΨ = Ψ_est - ⟨Ψ_measured⟩ = %+.4e   -⟨∂M/∂t⟩ = %+.4e   ratio ΔΨ/(-⟨∂M/∂t⟩) = %.2f\n",
            n, ΔΨ[n], -dMdt_mean[n], ΔΨ[n] / -dMdt_mean[n])

    if isempty(sel)
        @printf("  hill%d  no resolved b_centers in [%.3f, %.3f] -- skipping RMS/20%% diagnostic for this hill\n", n, lo, hi)
        continue
    end

    diff_sel  = neg_Gmix[n][sel] .- Ψ_measured[n][sel]
    max_psi   = maximum(abs.(Ψ_measured[n][sel]))
    rms_norm  = sqrt(mean(diff_sel.^2)) / max_psi

    bad = sel[abs.(diff_sel) .> 0.2 * max_psi]
    bad_range = isempty(bad) ? "none" : @sprintf("[%.3f, %.3f] (%d of %d bins)",
                                                  minimum(b_centers[bad]), maximum(b_centers[bad]), length(bad), length(sel))

    @printf("          RMS(-G_mix - Ψ)/max|Ψ| = %.4f   bins differing >20%%: %s\n", rms_norm, bad_range)
    @printf("          mean dM/dt over range = %+.4e   mean Ψ over range = %+.4e   |dMdt/Ψ| = %.3f\n",
            dMdt_mean[n], Ψ_meas[n], abs(dMdt_mean[n] / Ψ_meas[n]))
    if abs(dMdt_mean[n] / Ψ_meas[n]) > 0.2
        println("          ⚠ dM/dt is not small compared to Ψ here -- the quasi-steady assumption")
        println("            (G_mix ≈ -Ψ) is questionable over this range, not just discretization noise.")
    end
end

close(ds_g)

# ---- figure: small multiples, one column per hill, 2 rows -------------------
# colors consistent across this figure and the gmix_budget_combined_*.jl
# figures: Ψ_measured = purple, dM/dt = pink (matches those budget plots);
# -G_mix stays seagreen; Ψ_est / the B&N secant get a bright, distinct red so
# the "bulk estimate" reads as one consistent thing in both rows.
fig = Figure(size=(1500, 900))
axes_top = Axis[]
row1_handles = Any[]   # captured from hill1 only, for the one shared legend
for (i, n) in enumerate(hills)
    b_in, b_out = hill_contours[n].b_in, hill_contours[n].b_out
    lo, hi = minmax(b_in, b_out)

    ax = Axis(fig[1, i], xlabel=L"\Psi \;/\; (\kappa L_y)", ylabel=(i == 1 ? L"b \;/\; b^\star" : ""), title="hill$n")
    push!(axes_top, ax)
    p1 = lines!(ax, neg_Gmix[n],   b_centers, color=:seagreen, linewidth=2.5)
    p2 = lines!(ax, Ψ_measured[n], b_centers, color=:purple,   linewidth=2.5, linestyle=:dash)
    p3 = lines!(ax, dMdt_hill[n],  b_centers, color=:pink,     linewidth=2.5, linestyle=:dashdot)
    p4 = lines!(ax, fill(Ψ_est[n], 2), [lo, hi], color=:red, linewidth=2.5, linestyle=:dot)
    p5 = hlines!(ax, [b_in],  color=:gray50, linestyle=:dash,    linewidth=1)
    p6 = hlines!(ax, [b_out], color=:gray50, linestyle=:dashdot, linewidth=1)
    vlines!(ax, 0.0, color=:gray80, linestyle=:dot, linewidth=0.8)
    ylims!(ax, lo - 0.04, hi + 0.1)
    i == 1 && append!(row1_handles, (p1, p2, p3, p4, p5, p6))
end
Legend(fig[1, length(hills)+1], row1_handles,
       [L"-\mathcal{G}_{mix}(b)", L"\Psi_{measured}(b)", L"dM/dt(b)", L"\Psi_{est}\ (B\&N\ secant)", "b_in", "b_out"],
       labelsize=12)

row2_handles = Any[]
for (i, n) in enumerate(hills)
    b_in, b_out = hill_contours[n].b_in, hill_contours[n].b_out
    lo, hi = minmax(b_in, b_out)
    F0 = interp_at(b_edges, F_BN_col_fresh[n], b_in)
    F1 = interp_at(b_edges, F_BN_col_fresh[n], b_out)

    ax = Axis(fig[2, i], xlabel=L"F_{BN} \;/\; (\kappa L_y)", ylabel=(i == 1 ? L"b \;/\; b^\star" : ""))
    q1 = lines!(ax, F_BN_col_fresh[n], b_edges, color=:darkorange, linewidth=2.5)
    q2 = lines!(ax, [F0, F1], [b_in, b_out], color=:red, linewidth=2, linestyle=:dot)
    scatter!(ax, [F0, F1], [b_in, b_out], color=:red, markersize=8)
    ylims!(ax, lo - 0.04, hi + 0.1)
    i == 1 && append!(row2_handles, (q1, q2))
end
Legend(fig[2, length(hills)+1], row2_handles, [L"F_{BN}(b)", "B&N secant"], labelsize=12)

Label(fig[0, :], "Bryden & Nurser: buoyancy-resolved transport vs. bulk secant estimate (Seafloor, Ra=1e8, t=990.0–1000.0)",
      fontsize=18, font=:bold, tellwidth=false)

outpng = joinpath(plot_dir, "fbn_psi_bresolved.png")
save(outpng, fig; px_per_unit=2)
println("\nsaved figure -> $outpng")

println("\n=== final table (nondimensional: b -> b/b★, every flux -> /(κ·Ly)): Ψ_est vs ⟨Ψ_measured⟩, closure against ⟨∂M/∂t⟩ (all range-mean over [b_in,b_out], t=990.0-1000.0) ===")
println("ΔΨ ≡ Ψ_est - ⟨Ψ_measured⟩; ratio ≡ ΔΨ / (-⟨∂M/∂t⟩) -- should be ≈1 if the gap is explained by non-equilibrium alone")
@printf("%-6s %9s %9s %13s %13s %13s %13s %8s\n",
        "sill", "b_in/b★", "b_out/b★", "Ψ_est/(κLy)", "⟨Ψ_meas⟩/(κLy)", "ΔΨ/(κLy)", "⟨∂M/∂t⟩/(κLy)", "ratio")
for n in hills
    b_in, b_out = hill_contours[n].b_in, hill_contours[n].b_out
    ratio = ΔΨ[n] / -dMdt_mean[n]
    @printf("hill%-2d %9.4f %9.4f %+13.4e %+13.4e %+13.4e %+13.4e %7.2fx\n",
            n, b_in, b_out, Ψ_est[n], Ψ_meas[n], ΔΨ[n], dMdt_mean[n], ratio)
end

# ---- Part E: rendered comparison table for the talk (not just console) -----
# Same fields as the printed table just above, now as an actual Makie figure
# -- columns: Sill, b_in/b★, b_out/b★, Ψ_est/(κLy), ⟨Ψ_measured⟩/(κLy),
# ΔΨ/(κLy), ⟨∂M/∂t⟩/(κLy), ratio -- one row per hill, header row shaded.
col_headers = ["Sill", "b_in / b★", "b_out / b★", "Ψ_est / (κLy)", "⟨Ψ_meas⟩ / (κLy)",
               "ΔΨ / (κLy)", "⟨∂M/∂t⟩ / (κLy)", "ratio"]
row_strings = Vector{Vector{String}}()
for n in hills
    b_in, b_out = hill_contours[n].b_in, hill_contours[n].b_out
    ratio = ΔΨ[n] / -dMdt_mean[n]
    push!(row_strings, [
        "hill$n",
        @sprintf("%.4f", b_in), @sprintf("%.4f", b_out),
        @sprintf("%+.3e", Ψ_est[n]), @sprintf("%+.3e", Ψ_meas[n]),
        @sprintf("%+.3e", ΔΨ[n]), @sprintf("%+.3e", dMdt_mean[n]),
        @sprintf("%.2fx", ratio),
    ])
end

n_cols, n_rows = length(col_headers), length(row_strings)
fig_tbl = Figure(size=(1150, 90 + 40*(n_rows+1)))
gl = fig_tbl[1, 1] = GridLayout(tellwidth=false, halign=:center)

header_bg = colorant"#3b4a63"
row_bg    = (colorant"#eef1f6", colorant"#ffffff")   # alternating shading

for c in 1:n_cols
    Box(gl[1, c], color=header_bg, strokewidth=0)
    Label(gl[1, c], col_headers[c]; color=:white, font=:bold, fontsize=14, padding=(8,8,6,6))
end
for r in 1:n_rows, c in 1:n_cols
    Box(gl[r+1, c], color=row_bg[isodd(r) ? 1 : 2], strokewidth=0)
    Label(gl[r+1, c], row_strings[r][c]; fontsize=13, padding=(8,8,6,6))
end
rowgap!(gl, 0); colgap!(gl, 0)

Label(fig_tbl[0, 1], "F_BN vs. measured Ψ comparison, nondimensional (Seafloor, Ra=1e8, t=990.0–1000.0)",
      fontsize=16, font=:bold, tellwidth=false)

outpng_tbl = joinpath(plot_dir, "fbn_psi_bresolved_table.png")
save(outpng_tbl, fig_tbl; px_per_unit=2)
println("saved table -> $outpng_tbl")
