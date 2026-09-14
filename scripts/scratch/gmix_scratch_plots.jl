# gmix_scratch_plots.jl
#
# Talk figure: a pared-down version of gmix_budget_combined_seafloor.jl
# showing ONLY the Hill 1 and Basin 1 region budgets, for the Seafloor Ra=1e8
# experiment, in one 1x2 figure -- the full 6-panel combined figure is too
# much info for a short talk slide, and Control (flat-bottom) isn't part of
# this analysis at all.
#
# Reuses the same three pieces of machinery built for the paper figures:
#   - nondimensionalization: b -> b/b★, every volume-flux term -> /(κ·Ly)
#     (see gmix_budget_combined_control.jl for the full reasoning: Ra here
#     was varied by shrinking ν at fixed b★, so raw-dimensional volume
#     fluxes conflate the shrinking diffusivity scale with the actual
#     forcing-driven strength).
#   - occupancy_from_axis: local-neighbor-relative empty-bin masking, plus a
#     synthetic zero-closing point at b_min - Δb, so the curves visibly go
#     to zero exactly where there's no data instead of drawing a long
#     straight line through an artificially wide, near-empty bin
#     (gmix_budget_combined_control.jl / _seafloor.jl).
#   - boundary_threshold: the conservative (min-over-time-in-the-window
#     column minimum) b_in/b_out convention used everywhere else in this
#     project (region_contours.jl / cascade_figure.jl), evaluated here at
#     each region's own x_in/x_out edges over the SAME [t_lo, t_hi] window as
#     the plotted budget curves (rather than the whole segment), so the
#     shaded b_in/b_out band lines up with what the curves are averaged over.

using TopographicHorizontalConvection   # physics: nearest_xi
using NCDatasets
using CairoMakie
using Statistics
using Printf
using NaNStatistics

t_lo, t_hi = 990.0, 1000.0   # same shared window as the paper combined-budget figures
occ_tol    = 10.0
close_Δb   = 0.02

# only these two regions for the talk
regions = [
    ("Hill 1",  "hill1",  -1.35, -0.65),
    ("Basin 1", "basin1", -0.65, -0.35),
]

# NOTE: using the ORIGINAL (non-actively-rebinned) CODF files here, not the
# "_rebinned_nbins101_rebin10.0" equilibrium variant. b_in/b_out (via
# boundary_threshold_window below) are independent of this file -- they're
# read straight from the raw buoyancy_seg<N>.nc -- but the curves plotted
# AGAINST those thresholds (psi_b, Gmix_col, dMdt, ...) do depend on it, and
# this is the axis region_contours.jl / cascade_figure.jl's own b_in/b_out
# values were established against, so using it here keeps the zoom figure
# consistent with that convention instead of the equilibrium file's much
# coarser common axis (which has a single ~0.3-wide unresolved gap spanning
# the entire b_in/b_out band for both hill1 and basin1 -- see prior
# diagnostics in this conversation).
experiments = [
    (; label="Seafloor", data_dir="/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/",
        gmix_file="Gmix_quantile_regions_CODF_3hill_RA1e8_seg1to34.nc", last_seg=34),
]

plot_dir = joinpath(experiments[1].data_dir, "figures")
mkpath(plot_dir)

# ---- occupancy_from_axis: verbatim from gmix_budget_combined_*.jl ----------
function occupancy_from_axis(b; factor=10.0)
    n = length(b)
    gaps = diff(b)
    m = length(gaps)
    is_spike = falses(m)
    for k in 1:m
        neighbors = Float64[]
        k > 1 && push!(neighbors, gaps[k - 1])
        k < m && push!(neighbors, gaps[k + 1])
        isempty(neighbors) && continue
        is_spike[k] = gaps[k] > factor * minimum(neighbors)
    end
    occ = trues(n)
    k = 1
    while k <= m && is_spike[k]
        occ[k] = false
        k += 1
    end
    k = m
    while k >= 1 && is_spike[k]
        occ[k + 1] = false
        k -= 1
    end
    return occ
end

# region_contours.jl's boundary_threshold, generalized only to scan a given
# [t_lo, t_hi] window instead of the whole segment -- same "conservative =
# min-over-every-snapshot-in-the-window column minimum" definition, just
# windowed to match what the budget curves themselves are averaged over.
function boundary_threshold_window(buoyancy_file, x_target, x, wet, t_lo, t_hi)
    ix = argmin(abs.(x .- x_target))
    ds = NCDataset(buoyancy_file)
    t_seg = Float64.(ds["time"][:])
    i_lo = nearest_xi(t_seg, t_lo)
    i_hi = nearest_xi(t_seg, t_hi)
    b_col_seg = Array{Float64}(ds["b"][ix, :, :, i_lo:i_hi])
    close(ds)
    wet_col = wet[ix, :, :]
    nt = size(b_col_seg, 3)
    b_t = [minimum(@view(b_col_seg[:, :, k])[wet_col]) for k in 1:nt]
    return minimum(b_t)
end

# fine_psi_at_columns: ψ(x, b, t) via get_ψb_sort (TopographicHorizontalConvection's
# src/analysis/streamfunction.jl), restricted to a handful of x-columns and
# evaluated on a caller-supplied, non-uniform b_bins axis that explicitly
# contains the region boundary thresholds as EXACT bin values.
#
# get_ψb_sort's sort+cumsum sweep guarantees ψ(x, b₀, t) = 0 for any snapshot
# whose column has no cell colder than b₀ -- this is exact, not statistical.
# Since b_in/b_out (boundary_threshold_window above) are defined so that no
# snapshot in [t_lo, t_hi] ever has a colder cell at that column, this holds
# for EVERY snapshot in the window, so the time-mean is exactly zero there
# too, as long as a bin actually sits at that value (hence "exact bin values"
# above) -- no interpolation-across-an-unresolved-gap artifact, which is
# exactly what was fabricating a ~31 ψ_in_left(b_in) reading for hill1 out of
# the production gmix file's much coarser common axis.
#
# Mirrors boundary_threshold_window's "single column, windowed nearest_xi
# t-range, read straight from the raw LAST-segment files" pattern, extended
# to also read u from the matching velocities_seg<N>.nc. u lives on the FACE
# grid (x_faa, length Nx+1 -- walls at both ends, Bounded x-topology), so its
# x-index needs no extra offset to line up with b's cell-centered x_caa index
# as long as we only ever index u[1:Nx_b, ...] -- confirmed directly against
# these files: buoyancy_seg34.nc's b is (512,32,128,150), velocities_seg34.nc's
# u is (513,32,128,150), i.e. Nx_u == Nx_b + 1 exactly.
function fine_psi_at_columns(buoyancy_file, velocity_file, x, x_targets::Vector{Float64},
                              b_bins::Vector{Float64}, t_lo, t_hi)
    bds = NCDataset(buoyancy_file)
    vds = NCDataset(velocity_file)

    t_seg_b = Float64.(bds["time"][:])
    t_seg_v = Float64.(vds["time"][:])
    @assert t_seg_b == t_seg_v "buoyancy/velocity segment time axes differ: $buoyancy_file vs $velocity_file"

    Nx_b, Nx_u = size(bds["b"], 1), size(vds["u"], 1)
    @printf("  fine_psi_at_columns: size(b)=%s size(u)=%s\n", size(bds["b"]), size(vds["u"]))
    @assert Nx_u == Nx_b + 1 "unexpected u/b x-dimension mismatch: Nx_u=$Nx_u, Nx_b=$Nx_b"

    i_lo = nearest_xi(t_seg_b, t_lo)
    i_hi = nearest_xi(t_seg_b, t_hi)
    Nt   = i_hi - i_lo + 1

    ix = [nearest_xi(x, xt) for xt in x_targets]   # valid into both b's x_caa and u's first Nx_b faces
    Nx_local = length(ix)

    Δy_vec = Float64.(bds["Δy_aca"][:])
    Δz_vec = Float64.(bds["Δz_aac"][:])
    Ny, Nz = length(Δy_vec), length(Δz_vec)

    b_all = Array{Float64}(undef, Nx_local, Ny, Nz, Nt)
    u_all = Array{Float64}(undef, Nx_local, Ny, Nz, Nt)
    for (k, i) in enumerate(ix)
        b_all[k, :, :, :] = Array{Float64}(bds["b"][i, :, :, i_lo:i_hi])
        u_all[k, :, :, :] = Array{Float64}(vds["u"][i, :, :, i_lo:i_hi])
    end
    close(bds); close(vds)

    ψ_b, _ = get_ψb_sort(b_all, u_all, Δy_vec, Δz_vec, Nx_local, Ny, Nz, Nt; b_bins=b_bins)
    ψ_mean = dropdims(nanmean(Float64.(ψ_b), dims=3), dims=3)   # [Nx_local, n_bins], RAW units
    return ψ_mean
end

# linear interpolation of a b-indexed series onto one query point; flat
# extrapolation past the ends (same convention as gmix_CODF_equilibrium.jl's
# interp_series). Requires b sorted ascending -- true of every d.b built by
# region_budget below (occupancy-masked real bins plus one synthetic point
# below the real minimum, all in ascending order).
function interp_at(b, y, bq)
    n = length(b)
    bq <= b[1] && return y[1]
    bq >= b[n] && return y[n]
    j = clamp(searchsortedlast(b, bq), 1, n - 1)
    frac = (bq - b[j]) / (b[j+1] - b[j])
    return y[j] + frac * (y[j+1] - y[j])
end

# region_budget: same as gmix_budget_combined_*.jl's region_budget (per-region
# volume-flux curves, empty-bin masking + zero-closing), unchanged.
function region_budget(ds_g, region_key, region_in, region_out, x, t_lo, t_hi, occ, close_Δb)
    b    = Float64.(ds_g["b"][:])
    time = Float64.(ds_g["time"][:])
    i_lo = nearest_xi(time, t_lo)
    i_hi = nearest_xi(time, t_hi)
    rng  = i_lo:i_hi

    Gmix_col = Float64.(ds_g["Gmix_col_$region_key"][:, rng])
    Gsurf    = Float64.(ds_g["Gsurf_$region_key"][:, rng])
    psi_col  = Float64.(ds_g["psi_col_$region_key"][:, rng])

    gmix      = vec(mean(Gmix_col, dims=2))
    gBF       = vec(mean(Gsurf,    dims=2))
    transport = vec(mean(psi_col,  dims=2))

    # dM/dt is already computed (per "interval" chunk, on the b-axis common to
    # every other variable read above) by gmix_CODF_equilibrium.jl -- read it
    # straight from the file instead of re-deriving it from the M endpoints.
    # Pick the interval whose [t_start, t_end] best matches the [t_lo, t_hi]
    # window used for everything else here.
    t_start = Float64.(ds_g["t_start"][:])
    t_end   = Float64.(ds_g["t_end"][:])
    k_int   = argmin(abs.(t_start .- t_lo) .+ abs.(t_end .- t_hi))
    dMdt    = Float64.(ds_g["dMdt_$region_key"][:, k_int])
    @printf("    dMdt interval used for %-8s: t = %.2f → %.2f (requested %.1f → %.1f)\n",
            region_key, t_start[k_int], t_end[k_int], t_lo, t_hi)

    R         = dMdt .- gmix .- gBF .- transport

    index_in  = nearest_xi(x, region_in)
    index_out = nearest_xi(x, region_out)
    ψ = nanmean(Float64.(ds_g["psi_b"][:, :, rng]), dims=3)[:, :]
    ψ_in_left  = -ψ[index_in, :]
    ψ_in_right =  ψ[index_out, :]

    # keep only unmasked points, then close the budget with one synthetic
    # zero point at b_min - Δb (see file header / gmix_budget_combined_*.jl).
    b_real = b[occ]
    b_close = b_real[1] - close_Δb
    close!(arr) = vcat(0.0, arr[occ])
    b_new     = vcat(b_close, b_real)
    gmix      = close!(gmix)
    gBF       = close!(gBF)
    transport = close!(transport)
    dMdt      = close!(dMdt)
    R         = close!(R)
    ψ_in_left  = close!(ψ_in_left)
    ψ_in_right = close!(ψ_in_right)

    return (; b=b_new, gmix, gBF, transport, dMdt, R, ψ_in_left, ψ_in_right,
             tstart=time[i_lo], tend=time[i_hi])
end

# ---- load everything, per experiment -------------------------------------
cases = Any[]   # (exp_label, region_title, d, b_in_hat, b_out_hat)
raw_thr      = Dict{String,Float64}()   # "hill1_in"/"hill1_out"/"basin1_in"/"basin1_out" -> raw (dimensional) b
seafloor_ctx = Ref{Any}(nothing)        # (; x, b★, κ, Ly, buoyancy_file, velocity_file) -- for the fine ψ rebuild below
for exp in experiments
    ds1 = NCDataset(joinpath(exp.data_dir, "buoyancy_seg1.nc"))
    x   = Float64.(ds1["x_caa"][:])
    ν   = Float64(ds1.attrib["ν"])
    Pr  = Float64(ds1.attrib["Pr"])
    b★  = Float64(ds1.attrib["b★"])
    Ly  = sum(Float64.(ds1["Δy_aca"][:]))
    wet = ds1["b"][:, :, :, 2] .!= 0
    close(ds1)
    κ = ν / Pr
    @printf("%-8s  ν=%.4e  Pr=%.4e  κ=%.4e  Ly=%.4e  κ·Ly=%.4e  b★=%.4e\n", exp.label, ν, Pr, κ, Ly, κ*Ly, b★)

    buoyancy_file = joinpath(exp.data_dir, "buoyancy_seg$(exp.last_seg).nc")
    seafloor_ctx[] = (; x, b★, κ, Ly, buoyancy_file,
        velocity_file = joinpath(exp.data_dir, "velocities_seg$(exp.last_seg).nc"))

    ds_g = NCDataset(joinpath(exp.data_dir, exp.gmix_file))
    occ = occupancy_from_axis(Float64.(ds_g["b"][:]); factor=occ_tol)
    @printf("  axis-gap mask: %d / %d bins flagged sparse\n", count(!, occ), length(occ))

    for (title, key, xin, xout) in regions
        d = region_budget(ds_g, key, xin, xout, x, t_lo, t_hi, occ, close_Δb)

        b_in  = boundary_threshold_window(buoyancy_file, xin,  x, wet, t_lo, t_hi)
        b_out = boundary_threshold_window(buoyancy_file, xout, x, wet, t_lo, t_hi)
        @printf("  %-10s b_in=%.4f  b_out=%.4f  (t = %.1f → %.1f)\n", key, b_in, b_out, d.tstart, d.tend)
        raw_thr[key * "_in"]  = b_in
        raw_thr[key * "_out"] = b_out

        # nondimensionalize: b -> b/b★, every volume-flux term -> /(κ·Ly)
        d̂ = (; b         = d.b ./ b★,
               gmix      = d.gmix      ./ (κ * Ly),
               gBF       = d.gBF       ./ (κ * Ly),
               transport = d.transport ./ (κ * Ly),
               dMdt      = d.dMdt      ./ (κ * Ly),
               R         = d.R         ./ (κ * Ly),
               ψ_in_left  = d.ψ_in_left  ./ (κ * Ly),
               ψ_in_right = d.ψ_in_right ./ (κ * Ly),
               tstart=d.tstart, tend=d.tend)

        push!(cases, (exp.label, title, d̂, b_in / b★, b_out / b★))
    end
    close(ds_g)
end

@assert raw_thr["hill1_out"] == raw_thr["basin1_in"] "hill1/basin1 shared boundary threshold mismatch: $(raw_thr["hill1_out"]) vs $(raw_thr["basin1_in"])"

# ---- shared symmetric x-range across both panels ---------------------------
allvals = Float64[]
for (_, _, d, _, _) in cases
    append!(allvals, d.gmix .+ d.gBF, d.transport, d.dMdt, d.R, d.ψ_in_left, d.ψ_in_right)
end
xmax = 1.05 * maximum(abs, filter(isfinite, allvals))
xlims_shared = (-xmax, xmax)

# ---- talk-sized fonts (1x2 grid on a slide, not a manuscript multi-panel) --
fs_title       = 26
fs_panel_title = 20
fs_axislabel   = 20
fs_ticklabel   = 14
fs_legend      = 16
lw_bold        = 4.5   # gmix, transport
lw_thin        = 1.6   # dMdt, residual, ψ_in_left/right

fig = Figure(size=(1100, 560))
axes = Dict{Tuple{String,String}, Axis}()
plt  = Any[]   # legend handles, filled once

lookup = Dict((lab, ti) => (dd, bi, bo) for (lab, ti, dd, bi, bo) in cases)

for (i, exp) in enumerate(experiments), (j, (title, key, xin, xout)) in enumerate(regions)
    d, b_in, b_out = lookup[(exp.label, title)]
    ax = Axis(fig[i, j];
        title = title,
        titlesize = fs_panel_title,
        xticklabelsize = fs_ticklabel, yticklabelsize = fs_ticklabel,
        xticklabelrotation = π/4,
    )
    axes[(exp.label, title)] = ax

    # b_in / b_out: shaded band + horizontal lines, drawn first (behind the curves)
    lo, hi = minmax(b_in, b_out)
    hspan!(ax, lo, hi; color=(:gray, 0.18))
    hlines!(ax, [b_in, b_out]; color=:gray30, linestyle=:dash, linewidth=1.5)

    p1 = lines!(ax, d.gmix .+ d.gBF, d.b, color=:green,  linewidth=lw_bold)
    p2 = lines!(ax, d.transport,     d.b, color=:purple, linewidth=lw_bold)
    p3 = lines!(ax, d.dMdt,          d.b, color=:pink,     linewidth=lw_thin)
    p4 = lines!(ax, d.R,             d.b, color=:navyblue, linewidth=lw_thin, linestyle=:dash)
    p5 = lines!(ax, d.ψ_in_right,    d.b, color=:red,        linewidth=lw_thin)
    p6 = lines!(ax, d.ψ_in_left,     d.b, color=:dodgerblue, linewidth=lw_thin)
    isempty(plt) && append!(plt, (p1, p2, p3, p4, p5, p6))

    xlims!(ax, xlims_shared)
    j > 1 && hideydecorations!(ax; ticks=false, grid=false)
end
linkyaxes!(values(axes)...)
linkxaxes!(values(axes)...)

Label(fig[2, 1:2], L"\text{Volume flux} \;/\; (\kappa L_y)", fontsize=fs_axislabel)
Label(fig[1, 0], L"b \;/\; b^\star", fontsize=fs_axislabel, rotation=π/2, tellheight=false)

labels = [L"\mathcal{G}_{mix} + \mathcal{G}^{bf}", L"\text{transport in}", L"\frac{dM}{dt}",
          L"\text{residual}", L"\psi_{in, right}", L"\psi_{in, left}"]
Legend(fig[1, 3], collect(plt), labels; labelsize=fs_legend)

Label(fig[0, 1:2], "Hill 1 / Basin 1 volume budget, " * @sprintf("t = %.1f to %.1f", t_lo, t_hi),
      fontsize=fs_title, font=:bold, tellwidth=false)

outpath = joinpath(plot_dir, "gmix_budget_talk_hill1_basin1.png")
save(outpath, fig; px_per_unit=2)
println("saved → $outpath")

# ---- second figure: zoomed on the b_in/b_out boundary band -----------------
# The b_in/b_out band sits right at the bottom of the full b range (a couple
# hundredths of b/b★ wide -- see the printed b_in/b_out values above), so in
# the overview figure above it's a sliver a few pixels tall. Here every panel
# shares ONE y-range -- the union of all four panels' own padded bands -- so
# the four are directly comparable, the same way the overview figure's panels
# share one y-axis. x-limits stay per-panel, rescaled to the data inside that
# shared window (across all six series, so nothing runs off the frame edge).
zoom_pad_frac = 0.3    # padding as a fraction of each (b_in, b_out) band width
zoom_pad_min  = 0.02   # floor on that padding, in b / b★ units, for bands this thin

y_windows = [(minmax(b_in, b_out) .+ (-1, 1) .* max(zoom_pad_frac * abs(b_out - b_in), zoom_pad_min))
             for (_, _, _, b_in, b_out) in cases]
ylo_shared = minimum(first, y_windows)
yhi_shared = maximum(last,  y_windows)

# ---- rebuild ψ_in_left/ψ_in_right fresh, on a fine axis that contains
# b_in/b_out as EXACT bin values, for just this zoom figure ------------------
# The production gmix file's psi_b lives on a coarse, whole-domain volume-
# quantile axis with a huge unresolved gap right where these thresholds sit
# (see file header discussion / this session's diagnostics) -- the crossing
# markers below would otherwise be reading a fabricated interpolation, not a
# measurement. fine_psi_at_columns (above) recomputes ψ directly from the raw
# per-segment fields at just the 3 unique boundary columns needed here.
ctx = seafloor_ctx[]
b_in_hill1, b_shared, b_out_basin1 = raw_thr["hill1_in"], raw_thr["hill1_out"], raw_thr["basin1_out"]

n_fine_grid   = 300
pad_frac_fine = 0.15   # extra margin beyond the (already-padded) zoom window
span_lo = ylo_shared * ctx.b★ - pad_frac_fine * (yhi_shared - ylo_shared) * ctx.b★
span_hi = yhi_shared * ctx.b★ + pad_frac_fine * (yhi_shared - ylo_shared) * ctx.b★
b_bins_fine = sort(unique(vcat(collect(range(span_lo, span_hi, length=n_fine_grid)),
                                [b_in_hill1, b_shared, b_out_basin1])))
@printf("fine b-axis: %d points, span raw b = [%.4f, %.4f], thresholds = %.4f, %.4f, %.4f\n",
        length(b_bins_fine), span_lo, span_hi, b_in_hill1, b_shared, b_out_basin1)

ψ_fine_raw = fine_psi_at_columns(ctx.buoyancy_file, ctx.velocity_file, ctx.x,
                                  [-1.35, -0.65, -0.35], b_bins_fine, t_lo, t_hi)

κLy_seafloor = ctx.κ * ctx.Ly
b_fine_hat   = b_bins_fine ./ ctx.b★
# column mapping: 1 = x=-1.35 (hill1 in), 2 = x=-0.65 (hill1 out = basin1 in,
# shared boundary), 3 = x=-0.35 (basin1 out) -- matches region_budget's own
# index_in/index_out split and ψ_in_left = -ψ[in,:], ψ_in_right = ψ[out,:].
fine = Dict(
    "hill1"  => (; b = b_fine_hat, ψ_in_left = -ψ_fine_raw[1,:] ./ κLy_seafloor, ψ_in_right = ψ_fine_raw[2,:] ./ κLy_seafloor),
    "basin1" => (; b = b_fine_hat, ψ_in_left = -ψ_fine_raw[2,:] ./ κLy_seafloor, ψ_in_right = ψ_fine_raw[3,:] ./ κLy_seafloor),
)

fig2  = Figure(size=(1100, 560))
axes2 = Dict{Tuple{String,String}, Axis}()
plt2  = Any[]

for (i, exp) in enumerate(experiments), (j, (title, key, xin, xout)) in enumerate(regions)
    d, b_in, b_out = lookup[(exp.label, title)]
    fk = fine[key]   # freshly-rebuilt, finely-resolved ψ_in_left/right for THIS region

    ax = Axis(fig2[i, j];
        title = title,
        titlesize = fs_panel_title,
        xticklabelsize = fs_ticklabel, yticklabelsize = fs_ticklabel,
        xticklabelrotation = π/4,
    )
    axes2[(exp.label, title)] = ax

    lo, hi = minmax(b_in, b_out)
    hspan!(ax, lo, hi; color=(:gray, 0.18))
    hlines!(ax, [b_in, b_out]; color=:gray30, linestyle=:dash, linewidth=1.5)

    p1 = lines!(ax, d.gmix .+ d.gBF, d.b, color=:green,  linewidth=lw_bold)
    p2 = lines!(ax, d.transport,     d.b, color=:purple, linewidth=lw_bold)
    p3 = lines!(ax, d.dMdt,          d.b, color=:pink,     linewidth=lw_thin)
    p4 = lines!(ax, d.R,             d.b, color=:navyblue, linewidth=lw_thin, linestyle=:dash)
    p5 = lines!(ax, fk.ψ_in_right,   fk.b, color=:red,        linewidth=lw_thin)
    p6 = lines!(ax, fk.ψ_in_left,    fk.b, color=:dodgerblue, linewidth=lw_thin)
    isempty(plt2) && append!(plt2, (p1, p2, p3, p4, p5, p6))

    ylims!(ax, ylo_shared, yhi_shared)

    # rescale x to just the data actually inside the SHARED zoom window,
    # across ALL six series (including the fine ψ_in_left/right) so every
    # curve's peak stays visible in-frame -- nothing gets clipped off the edge.
    in_window      = ylo_shared .<= d.b .<= yhi_shared
    fine_in_window = ylo_shared .<= fk.b .<= yhi_shared
    zoomvals  = Float64[]
    for series in (d.gmix .+ d.gBF, d.transport, d.dMdt, d.R)
        append!(zoomvals, series[in_window])
    end
    append!(zoomvals, fk.ψ_in_left[fine_in_window], fk.ψ_in_right[fine_in_window])
    zoomvals = filter(isfinite, zoomvals)
    xmax_j = isempty(zoomvals) ? xmax : 1.05 * maximum(abs, zoomvals)
    xlims!(ax, -xmax_j, xmax_j)

    # mark where ψ_in_left / ψ_in_right cross b_in and b_out, with a dotted
    # drop-line down to the x-axis, so the inflow-vs-outflow magnitude at each
    # boundary buoyancy reads directly off the x ticks instead of by eye
    # against the curve -- this is the "how much less left than entered"
    # comparison at the region's own boundary threshold. The value itself is
    # also written right at the foot of each drop-line, in the matching
    # color; left/right text alignment (away from the line) separates the two
    # colors' labels, and a pixel-space vertical stagger between the b_in and
    # b_out label of the SAME color keeps those two from overlapping too --
    # necessary because a thin band (the usual case) puts the two crossing
    # values close together in x, sometimes close enough to land on the same
    # spot in data coordinates.
    # fk.b was built to contain b_in/b_out as EXACT values (see fine_psi_at_columns
    # call above), so this should always be an exact array hit, not an
    # interpolation -- interp_at is kept only as a defensive fallback.
    for (side, series, col, xalign) in (("left", fk.ψ_in_left, :dodgerblue, :left), ("right", fk.ψ_in_right, :red, :right))
        for (row, (b_name, b_target)) in enumerate((("b_in", b_in), ("b_out", b_out)))
            idx = findfirst(==(b_target), fk.b)
            val = idx === nothing ? interp_at(fk.b, series, b_target) : series[idx]
            idx === nothing && @printf("    WARNING: no exact fine-axis hit for %s %s (falling back to interp_at)\n", key, b_name)
            @printf("    CROSSING  %-8s %-10s ψ_in_%-5s @ %s=%.4f  ->  %.6f  (exact=%s)\n",
                    exp.label, title, side, b_name, b_target, val, idx !== nothing)
            lines!(ax, [val, val], [ylo_shared, b_target]; color=col, linestyle=:dot, linewidth=1.2)
            scatter!(ax, [val], [b_target]; color=col, markersize=10, strokewidth=1, strokecolor=:black)
            # stagger UPWARD (not down) -- the anchor already sits right at the
            # bottom of the axis (align=:bottom), so any downward pixel offset
            # pushes the second label past the axis's clip boundary and it
            # silently vanishes.
            text!(ax, val, ylo_shared; text=@sprintf("%.1f", val), color=col,
                  fontsize=fs_ticklabel - 2, align=(xalign, :bottom), offset=(0, (row - 1) * 13))
        end
    end

    j > 1 && hideydecorations!(ax; ticks=false, grid=false)
end
linkyaxes!(values(axes2)...)

Label(fig2[2, 1:2], L"\text{Volume flux} \;/\; (\kappa L_y)", fontsize=fs_axislabel)
Label(fig2[1, 0], L"b \;/\; b^\star", fontsize=fs_axislabel, rotation=π/2, tellheight=false)

Legend(fig2[1, 3], collect(plt2), labels; labelsize=fs_legend)

Label(fig2[0, 1:2], "Hill 1 / Basin 1 volume budget (zoom on b_in/b_out), " *
      @sprintf("t = %.1f to %.1f", t_lo, t_hi),
      fontsize=fs_title, font=:bold, tellwidth=false)

outpath2 = joinpath(plot_dir, "gmix_budget_talk_hill1_basin1_zoom.png")
save(outpath2, fig2; px_per_unit=2)
println("saved → $outpath2")
