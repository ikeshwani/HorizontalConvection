# region_contours.jl -- generalizes single_contour_efficiency.jl's b_in/b_out
# contour selection from hill1 alone to all three hill regions, chained:
# hill_n's b_out is reused directly as hill_{n+1}'s b_in (skipping the
# intervening basin's own boundary columns entirely -- the whole point is
# tracking hill_n's own overflow forward, not independently re-measuring at
# the next hill's own edge).
#
# Config mirrors gmix_CODF.jl's own experiment/Ra -> data_dir/gmix_file
# derivation, so switching between the 4 experiments (Ra=1e8/1e6 x hill/flat)
# is just editing the two lines below. The SAME region boundaries are used in
# every experiment, including the flat ones -- the point of this whole
# exercise is comparing mixing efficiency at matching x-ranges across
# topographies, so the region list is intentionally NOT parametrized per
# experiment.

using TopographicHorizontalConvection
using NCDatasets
using CairoMakie
using Statistics
using Printf

# ============================================================================
# CONFIG -- EDIT THESE two lines to switch experiments
# ============================================================================
experiment = "hill"     # "hill" or "control"
Ra         = 1e8        # 1e8 or 1e6
# ============================================================================

if Ra == 1e8
    Ra_tag, Ra_str = "ra1e8", "RA1e8"
elseif Ra == 1e6
    Ra_tag, Ra_str = "ra1e6", "RA1e6"
else
    error("unsupported Ra: $Ra (use 1e8 or 1e6)")
end

if experiment == "control"
    topo, tag = "flat", "Control"
    segments = Ra == 1e8 ? (1:25) : (1:16)
elseif experiment == "hill"
    topo, tag = "threehill", "3hill"
    segments = Ra == 1e8 ? (1:34) : (1:19)
else
    error("unknown experiment: $experiment (use \"control\" or \"hill\")")
end

data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/$(Ra_tag)_4xstretch_$(topo)_baseforcing_zerostart/"
plot_dir = joinpath(@__DIR__)   # scripts/scratch/ -- keep scratch outputs next to the script
last_seg = last(segments)

gmix_file     = joinpath(data_dir, "Gmix_quantile_regions_CODF_$(tag)_$(Ra_str)_seg$(first(segments))to$(last(segments)).nc")
buoyancy_file = joinpath(data_dir, "buoyancy_seg$(last_seg).nc")

# ---- region boundaries: SAME in every experiment, including flat ----------
col_bounds = [
    ("basin0", -1.8,  -1.35), ("hill1", -1.35, -0.65), ("basin1", -0.65, -0.35),
    ("hill2",  -0.35,  0.35), ("basin2",  0.35,  0.65), ("hill3",   0.65,  1.35),
    ("basin3",  1.35,  Inf),
]
hill_names = ["hill1", "hill2", "hill3"]

# ---- grid + wet mask, matching gmix_CODF.jl's own Pass 0 setup ------------
ds1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
H  = ds1.attrib["H"]
x = ds1["x_caa"][:]; y = ds1["y_aca"][:]; z = ds1["z_aac"][:]
wet = ds1["b"][:, :, :, 2] .!= 0
close(ds1)

# boundary_threshold: coldest water at the column nearest x_target, across
# EVERY snapshot of the last segment -- same "does it hold throughout the
# whole segment" sweep as single_contour_efficiency.jl's b_in_t/b_out_t,
# generalized to any x. The conservative (min-over-time) value is the one
# fixed threshold guaranteed safe at every single timestep in the window: for
# "no water colder than B exists at time t" to hold at every t, B must be at
# most the smallest of the per-timestep minima.

function boundary_threshold(buoyancy_file, x_target, x, wet)
    ix = argmin(abs.(x .- x_target))
    ds = NCDataset(buoyancy_file)
    t_seg = ds["time"][:]
    b_col_seg = Array{Float64}(ds["b"][ix, :, :, :])   # [Ny, Nz, Nt]
    close(ds)

    wet_col = wet[ix, :, :]
    nt = length(t_seg)
    b_t = [minimum(@view(b_col_seg[:, :, k])[wet_col]) for k in 1:nt]

    return (; x_used=x[ix], t=t_seg, b_t, conservative=minimum(b_t))
end


# we only need to compute the bouyancy contours at four regions because
# I will be setting hill_n's b_out and b_{n+1}'s b_in as the same computed value
# so we only need to compute at hill 1 left boundary, hill 1 right boundary, 
# hill 2 right boundary, hill 3 right boundary
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#vector of hill name, hill right bound, hill left bound
hill_bounds = [(nm, xlo, xhi) for (nm, xlo, xhi) in col_bounds if nm in hill_names]

interface_xs = [hill_bounds[1][2]]                       # hill1's own left edge
append!(interface_xs, [xhi for (nm, xlo, xhi) in hill_bounds])  # every hill's own right edge
#the four locations where we compute boundary_thresholds

println("\ncomputing boundary thresholds at $(length(interface_xs)) interfaces: $interface_xs")
thresholds = [boundary_threshold(buoyancy_file, xt, x, wet) for xt in interface_xs]

println("\nb(t) at each interface, across all $(length(thresholds[1].t)) snapshots of segment $last_seg (t=$(thresholds[1].t[1]) to $(thresholds[1].t[end])):")
for (k, th) in enumerate(thresholds)
    @printf("  interface %d (x=%.3f):  min/median/max = %.4f / %.4f / %.4f   conservative (min over time) = %.4f\n",
            k, th.x_used, minimum(th.b_t), median(th.b_t), maximum(th.b_t), th.conservative)
end

hill_contours = Dict(hill_names[n] => (b_in=thresholds[n].conservative, b_out=thresholds[n+1].conservative)
                      for n in 1:length(hill_names))

println("\nchained contours per hill (conservative, min-over-time):")
for nm in hill_names
    c = hill_contours[nm]
    @printf("  %-6s  b_in=%.4f  b_out=%.4f  Δb=%.4f\n", nm, c.b_in, c.b_out, c.b_out - c.b_in)
end


# single_contour_efficiency.png-style shell plot, one per hill -- basin |
# hill | basin, with the [b_in,b_out] shell shaded. Generalizes
# single_contour_efficiency.jl's hill1-only plot; named with a per-hill
# suffix so these don't collide with that script's own output file.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ds_last = NCDataset(buoyancy_file)
it = length(ds_last["time"][:])
t_val = ds_last["time"][it]
b_full = Array{Float64}(ds_last["b"][:, :, :, it])
close(ds_last)
b_full[.!wet] .= NaN

iy = argmin(abs.(y .- y[end]/4))
b2d = b_full[:, iy, :]

function plot_hill_shell(plot_dir, hill_name, blo, bhi, left_nm, left_xlo, right_nm, right_xhi,
                          hill_xlo, hill_xhi, x, z, b2d, H, t_val)
    fig = Figure(size=(900, 550))
    ax = Axis(fig[1,1], xlabel="x", ylabel="z",
              title=@sprintf("%s | %s | %s  --  shell [%.3f, %.3f]  (t=%.1f)", left_nm, hill_name, right_nm, blo, bhi, t_val))
    hm = heatmap!(ax, x, z, b2d, colormap=:balance, nan_color=:black)
    contourf!(ax, x, z, b2d, levels=[blo, bhi], colormap=[:yellow], extendlow=:transparent, extendhigh=:transparent)
    contour!(ax, x, z, b2d, levels=[blo, bhi], color=:black, linewidth=2)
    vlines!(ax, [hill_xlo, hill_xhi], color=:magenta, linewidth=1.5, linestyle=:dot)
    Colorbar(fig[1,2], hm, label="b")
    xr = isfinite(right_xhi) ? right_xhi : x[end]
    xlims!(ax, left_xlo, xr); ylims!(ax, -H, 0.0)

    outpng = joinpath(plot_dir, "single_contour_efficiency_$(hill_name).png")
    save(outpng, fig)
    println("saved -> $outpng")
    return outpng
end

println()
for (n, nm) in enumerate(hill_names)
    idx = findfirst(t -> t[1] == nm, col_bounds)
    left_nm,  left_xlo,  _         = col_bounds[idx-1]
    _,        hill_xlo,  hill_xhi  = col_bounds[idx]
    right_nm, _,         right_xhi = col_bounds[idx+1]
    c = hill_contours[nm]
    plot_hill_shell(plot_dir, nm, c.b_in, c.b_out, left_nm, left_xlo, right_nm, right_xhi,
                     hill_xlo, hill_xhi, x, z, b2d, H, t_val)
end

# ============================================================================
# F_BN / Gmix / Ψ backsolve, generalized to all 3 hills -- same machinery as
# single_contour_efficiency.jl (divergence-theorem F_BN, interval-averaged to
# match the CODF file's own interval-mean convention, evaluated at
# b_mid = 0.5*(b_in+b_out) since a secant over [b_in,b_out] estimates the
# derivative at the interval's midpoint, not at either endpoint).
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ---- physics constants + grid spacing, needed for conv_dV_snapshot --------
ds1b = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
Lx = ds1b.attrib["Lx"]; Pr = ds1b.attrib["Pr"]; b★ = ds1b.attrib["b★"]
Δx_face = ds1b["Δx_faa"][:]; Δy_face = ds1b["Δy_afa"][:]; Δz_face = ds1b["Δz_aaf"][:]
Δx_center = reshape(ds1b["Δx_caa"][:], Nx, 1, 1)
Δy_center = reshape(ds1b["Δy_aca"][:], 1, Ny, 1)
Δz_center = reshape(ds1b["Δz_aac"][:], 1, 1, Nz)
close(ds1b)
vol = Δx_center .* Δy_center .* Δz_center
ν = sqrt(Pr * b★ * H^3 / Ra); κ = ν / Pr
zBL = boundary_layer_depth(Lx, Ra)

# ---- per-hill interior masks (z<zBL, x in the hill's own range -- matches
# single_contour_efficiency.jl's hill1-interior convention exactly) --------
hill_idxs = Dict{String, Vector{Int}}()
for (nm, xlo, xhi) in hill_bounds
    mask = (reshape(x, Nx,1,1) .>= xlo) .& (reshape(x, Nx,1,1) .< xhi) .& (reshape(z, 1,1,Nz) .< zBL)
    hill_idxs[nm] = findall(vec(mask .& wet))
end

# ---- per-cell divergence of the diffusive flux, matching gmix_CODF.jl -----
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

# ---- cell-centered |∇b|, for A_bound (bounding isopycnal area, co-area
# formula) -- reuses the same face-difference machinery as conv_dV_snapshot,
# but AVERAGES adjacent face differences to cell centers instead of
# differencing them again. Only averages over faces that are actually valid
# (not touching a dry/immersed-boundary neighbor) -- a blind /2 at a wall or
# hill-flank cell would systematically halve |∇b| (and thus A_bound) right
# where the efficiency signal is expected to peak.
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

# linear interpolation on a sorted (possibly non-uniform) axis, matching
# fbn_psi_comparison.jl:550-557 / single_contour_efficiency.jl
function interp_at(xs, ys, x0)
    x0 <= xs[1] && return ys[1]
    x0 >= xs[end] && return ys[end]
    i = searchsortedlast(xs, x0)
    i = clamp(i, 1, length(xs) - 1)
    frac = (x0 - xs[i]) / (xs[i+1] - xs[i])
    return ys[i] + frac * (ys[i+1] - ys[i])
end

# ---- CODF's own equilibrium interval bounds + measured budget terms -------
ds_gK = NCDataset(gmix_file)
t_start_int = ds_gK["t_start"][:]
t_end_int   = ds_gK["t_end"][:]
b_centers_g = Float64.(ds_gK["b"][:])
kk = length(t_start_int)   # last interval = equilibrium window
println("\nusing CODF interval $kk: t = $(t_start_int[kk]) to $(t_end_int[kk])")

Gmix_int_all  = Dict(nm => Float64.(ds_gK["Gmix_int_$nm"][:, kk])  for nm in hill_names)
psi_int_all   = Dict(nm => Float64.(ds_gK["psi_int_$nm"][:, kk])   for nm in hill_names)
Gsurf_int_all = Dict(nm => Float64.(ds_gK["Gsurf_int_$nm"][:, kk]) for nm in hill_names)
dMdt_all      = Dict(nm => Float64.(ds_gK["dMdt_$nm"][:, kk])      for nm in hill_names)
R_all         = Dict(nm => Float64.(ds_gK["R_$nm"][:, kk])         for nm in hill_names)
close(ds_gK)

# ---- interval-averaged F_BN at each hill's (b_in,b_out) -- ONE pass over
# every snapshot in the interval covers all 3 hills at once (conv_dV_snapshot
# is computed once per snapshot regardless of how many hills use it).
t_seg = thresholds[1].t   # buoyancy_file's own time vector, already loaded once
interval_idx = findall(t -> t_start_int[kk] - 1e-9 <= t <= t_end_int[kk] + 1e-9, t_seg)
println("averaging F_BN over $(length(interval_idx)) snapshots (t=$(t_seg[interval_idx[1]]) to $(t_seg[interval_idx[end]]))")

F_BN_lo_sum = Dict(nm => 0.0 for nm in hill_names)
F_BN_hi_sum = Dict(nm => 0.0 for nm in hill_names)
A_bound_sum = Dict(nm => 0.0 for nm in hill_names)

# per-hill cell volumes, precomputed once (idxs/vol don't change per snapshot)
vv_hill = Dict(nm => vec(vol)[hill_idxs[nm]] for nm in hill_names)

ds_seg = NCDataset(buoyancy_file)
for k in interval_idx
    bk = Array{Float64}(ds_seg["b"][:, :, :, k])
    bk[.!wet] .= NaN
    CONV_dV_k = conv_dV_snapshot(bk)
    gradmag_k = grad_b_mag(bk)
    for nm in hill_names
        c = hill_contours[nm]
        idxs = hill_idxs[nm]
        bk_h = vec(bk)[idxs]
        F_BN_lo_sum[nm] += -sum(@view CONV_dV_k[idxs][bk_h .< c.b_in])
        F_BN_hi_sum[nm] += -sum(@view CONV_dV_k[idxs][bk_h .< c.b_out])

        sel_bin = (bk_h .>= c.b_in) .& (bk_h .< c.b_out)
        gv_k = vec(gradmag_k)[idxs]
        A_bound_sum[nm] += sum((gv_k .* vv_hill[nm])[sel_bin])
    end
end
close(ds_seg)
n_int_snaps = length(interval_idx)

# range_average: the average of y(b) over [b_lo,b_hi), using whichever
# b_centers_g points fall in that range -- matches fbn_psi_comparison.jl's
# own nanmean(psi_int[b_sel]) pattern. This is the term that Gmix_from_FBN is
# EXACTLY equal to (by the FTC identity: ΔF_BN/Δb = average of Gmix over
# [b_in,b_out], given F_BN(b_in)≈0) -- so every term in the budget equation
# needs to be averaged the same way, not just Gmix, or the compared terms
# stop belonging to the same equation.
function range_average(b_centers, y, b_lo, b_hi)
    sel = findall(b -> b_lo <= b <= b_hi, b_centers)
    isempty(sel) && error("no b_centers points fall in [$b_lo, $b_hi) -- range too narrow for this axis")
    return mean(@view y[sel]), length(sel)
end

# break_at_gaps: don't let `lines!` draw a segment across bins that are wider
# than a real, resolved gap -- that's what visually manufactures a "cliff"
# out of two widely-separated real points on the CODF's quantile axis. Same
# fix as single_contour_efficiency.jl's Ψ plot, generalized here for Gmix.
function break_at_gaps(b_axis, y; gap_thresh=0.005)
    b_new = Float64[]; y_new = Float64[]
    for i in eachindex(b_axis)
        push!(b_new, b_axis[i]); push!(y_new, y[i])
        if i < length(b_axis) && abs(b_axis[i+1] - b_axis[i]) > gap_thresh
            push!(b_new, NaN); push!(y_new, NaN)
        end
    end
    return b_new, y_new
end

# ---- zoom_in_gmix_$(hill).png: the actual Gmix_int(b) curve shape around
# [b_in,b_out], with b_in/b_out/b_mid marked -- generalizes the original
# single_contour_efficiency.jl diagnostic so we can SEE why Gmix_from_FBN
# (the exact average over the whole interval) does or doesn't match
# Gmix_meas at any single point, rather than just inferring it from %diffs.
# Scatter shows every real point; the line is broken across gaps wider than
# gap_thresh so a sparse stretch of axis can't manufacture a fake cliff.
function plot_gmix_zoom(plot_dir, hill_name, b_centers, Gmix_curve, blo, bhi, bmid)
    pad = 3 * (bhi - blo)
    b_line, gmix_line = break_at_gaps(b_centers, Gmix_curve)

    fig = Figure()
    ax = Axis(fig[1,1], title="$hill_name gmix versus buoyancy", xlabel="gmix", ylabel="b")
    lines!(ax, gmix_line, b_line, color=:royalblue)
    scatter!(ax, Gmix_curve, b_centers, color=:royalblue, markersize=5)
    hlines!(ax, blo,  linestyle=:dash, color=:orange, linewidth=2, label="b_in")
    hlines!(ax, bhi,  linestyle=:dash, color=:red,    linewidth=2, label="b_out")
    hlines!(ax, bmid, linestyle=:dot,  color=:green,  linewidth=2, label="b_mid")
    axislegend(ax, position=:rb)
    ylims!(ax, blo - pad, bhi + pad)
    outpng = joinpath(plot_dir, "zoom_in_gmix_$(hill_name).png")
    save(outpng, fig)
    println("saved -> $outpng")
    return outpng
end

println("\nper-hill F_BN / Gmix / Ψ backsolve summary (interval-mean, range-averaged over [b_in,b_out]):")
efficiency_all = Dict{String, Float64}()
A_bound_all    = Dict{String, Float64}()
F_BN_mid_all   = Dict{String, Float64}()

for nm in hill_names
    c = hill_contours[nm]
    blo, bhi = c.b_in, c.b_out
    bmid = 0.5 * (blo + bhi)

    plot_gmix_zoom(plot_dir, nm, b_centers_g, Gmix_int_all[nm], blo, bhi, bmid)

    F_BN_lo_mean = F_BN_lo_sum[nm] / n_int_snaps
    F_BN_hi_mean = F_BN_hi_sum[nm] / n_int_snaps
    Gmix_from_FBN = (F_BN_hi_mean - F_BN_lo_mean) / (bhi - blo)   # exactly = average(Gmix) over [blo,bhi]

    # ---- mixing efficiency = F_BN / A_bound -- F_BN_mid is the edge-average
    # (a flux, matching gmix_CODF.jl's own edge->center convention for F_BN),
    # NOT the same as Gmix_from_FBN (a rate, ΔF_BN/Δb) -- these are two
    # different physical quantities built from the same two F_BN measurements.
    F_BN_mid  = 0.5 * (F_BN_lo_mean + F_BN_hi_mean)
    A_bound   = (A_bound_sum[nm] / n_int_snaps) / (bhi - blo)
    efficiency = F_BN_mid / max(A_bound, eps())
    efficiency_all[nm] = efficiency
    A_bound_all[nm]    = A_bound
    F_BN_mid_all[nm]   = F_BN_mid

    Gmix_meas,  n_bins = range_average(b_centers_g, Gmix_int_all[nm],  blo, bhi)
    psi_meas,   _       = range_average(b_centers_g, psi_int_all[nm],   blo, bhi)
    Gsurf_meas, _       = range_average(b_centers_g, Gsurf_int_all[nm], blo, bhi)
    dMdt_meas,  _       = range_average(b_centers_g, dMdt_all[nm],      blo, bhi)
    R_meas,     _       = range_average(b_centers_g, R_all[nm],         blo, bhi)

    Gmix_backsolve = dMdt_meas - psi_meas - Gsurf_meas
    psi_backsolve  = dMdt_meas - Gmix_meas - Gsurf_meas
    budget_rhs     = Gmix_meas + psi_meas + Gsurf_meas
    R_handcomputed = dMdt_meas - budget_rhs

    println()
    @printf("%s  (b_in=%.4f, b_out=%.4f, %d CODF bins averaged)\n", nm, blo, bhi, n_bins)
    @printf("  F_BN_mid (edge-avg flux)         = %.4e\n", F_BN_mid)
    @printf("  A_bound (bounding isopycnal area) = %.4e\n", A_bound)
    @printf("  efficiency = F_BN_mid / A_bound   = %.4e\n", efficiency)
    @printf("  Gmix_from_FBN (independent)      = %.4e\n", Gmix_from_FBN)
    @printf("  Gmix_int (measured, range-avg)   = %.4e   (%%diff vs Gmix_from_FBN = %+.1f%%)\n",
            Gmix_meas, 100*(Gmix_from_FBN-Gmix_meas)/abs(Gmix_meas))
    @printf("  psi_int (measured, range-avg)    = %.4e\n", psi_meas)
    @printf("  Gsurf_int (measured, range-avg)  = %.4e\n", Gsurf_meas)
    @printf("  dMdt (measured, range-avg)       = %.4e\n", dMdt_meas)
    @printf("  R (hand-computed, range-avg)     = %.4e   (R_%s saved, range-avg = %.4e)\n", R_handcomputed, nm, R_meas)
    @printf("  G_mix_backsolve = dMdt-Ψ-Gsurf   = %.4e   (%%diff vs Gmix_int = %+.1f%%)\n",
            Gmix_backsolve, 100*(Gmix_backsolve-Gmix_meas)/abs(Gmix_meas))
    @printf("  Ψ_backsolve = dMdt-Gmix-Gsurf    = %.4e   (%%diff vs psi_int  = %+.1f%%)\n",
            psi_backsolve, 100*(psi_backsolve-psi_meas)/abs(psi_meas))
end

println("\nmixing efficiency summary across all three hills:")
for nm in hill_names
    @printf("  %-6s  F_BN_mid=%.4e  A_bound=%.4e  efficiency=%.4e\n",
            nm, F_BN_mid_all[nm], A_bound_all[nm], efficiency_all[nm])
end
