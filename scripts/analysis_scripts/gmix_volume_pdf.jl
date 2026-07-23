# Diagnostic: volume PDF in buoyancy coordinates, p(b) = dM/db.
#
# gmix_CODF.jl bins a region's cells by buoyancy and takes ONE d/db of a cumulative
# sum.  A bin holding no cells leaves the cumulative sum flat, so G_mix reads exactly
# 0 there while the neighbouring occupied bin absorbs the whole jump — the derivative
# of a staircase, which is the "0 ↔ negative" oscillation seen in basin 3.
#
# A "hole" here means an EMPTY bin sitting inside the b range the region does occupy
# (water exists either side of it, just not in it) — as opposed to the trivially empty
# bins beyond the region's b range.  Holes appear when the b gap between adjacent
# z-levels exceeds the bin width, i.e. when the bins are too FINE for the local
# stratification.  M_<region> in the CODF file is the volume CDF, so p(b) = dM/db is
# the histogram that shows exactly where that happens.
#
# Read-only: nothing here writes back to the CODF output.

using NCDatasets
using CairoMakie
using NaNStatistics
using Statistics
using Printf

# ---- config ----
experiment     = "hill"        # "control" (flat bottom) or "hill" (3-hill GRC)
Ra             = 1e8           # 1e8 or 1e6
b_range        = (-1, 1)       # must match the run that produced the CODF file
n_b_bins       = 101           # ditto (101 edges → 100 centers)
n_tmean        = 11            # trailing snapshots to time-average (matches budget plots)
focus_region   = "basin3"      # region for the zoom / case study
b_zoom         = :auto         # :auto → the focus region's occupied b range, or (lo, hi)
do_cell_counts = true          # re-read one buoyancy snapshot to count cells per bin

if Ra == 1e8
    Ra_tag, Ra_str = "ra1e8", "RA1e8"
elseif Ra == 1e6
    Ra_tag, Ra_str = "ra1e6", "RA1e6"
else
    error("unsupported Ra: $Ra (use 1e8 or 1e6)")
end

if experiment == "control"
    topo, tag = "flat", "Control"
    segments  = Ra == 1e8 ? (1:15) : (1:7)
elseif experiment == "hill"
    topo, tag = "threehill", "3hill"
    segments  = Ra == 1e8 ? (1:23) : (1:10)
else
    error("unknown experiment: $experiment (use \"control\" or \"hill\")")
end

data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/$(Ra_tag)_4xstretch_$(topo)_baseforcing_zerostart/"
plot_dir = joinpath(data_dir, "figures")
mkpath(plot_dir)

gmix_file = joinpath(data_dir,
    "Gmix_regions_CODF_$(tag)_$(Ra_str)_seg$(first(segments))to$(last(segments)).nc")

# interior column x-bounds — mirror col_bounds in gmix_CODF.jl
col_bounds = [
    ("basin0", -1.8,  -1.35), ("hill1", -1.35, -0.65), ("basin1", -0.65, -0.35),
    ("hill2",  -0.35,  0.35), ("basin2",  0.35,  0.65), ("hill3",   0.65,  1.35),
    ("basin3",  1.35,  Inf),
]
col_names = [nm for (nm, _, _) in col_bounds]

# b_edges as built in gmix_CODF.jl; the file's `b` axis is the centers
b_edges = collect(range(b_range[1], b_range[2], length=n_b_bins))
db_bin  = b_edges[2] - b_edges[1]

# ---- load the volume CDF M(b) per full-depth column -------------------------
println("reading $gmix_file")
ds = NCDataset(gmix_file)
b_g = Float64.(ds["b"][:])
t_g = Float64.(ds["time"][:])
Nt  = length(t_g)
win = max(1, Nt - n_tmean + 1):Nt
@printf("  %d b-centers, %d time steps; averaging over last %d (t = %.1f → %.1f)\n",
        length(b_g), Nt, length(win), t_g[win[1]], t_g[end])

# full-depth column = interior + its BL strip (same partition the budget uses)
tmean_col(prefix, nm) = vec(nanmean(Float64.(ds["$(prefix)_$(nm)"][:, win]), dims=2))
M_col    = Dict(nm => tmean_col("M", nm) .+ tmean_col("M_bl", nm) for nm in col_names)
Gmix_col = Dict(nm => vec(nanmean(Float64.(ds["Gmix_col_$(nm)"][:, win]), dims=2))
                for nm in col_names)
close(ds)

# p(b) = dM/db, evaluated at the midpoints between b-centers
b_mid   = 0.5 .* (b_g[1:end-1] .+ b_g[2:end])
pdf_col = Dict(nm => diff(M_col[nm]) ./ diff(b_g) for nm in col_names)

# A region only occupies part of the b axis; bins beyond that range are empty for the
# trivial reason that the region holds no water there.  The pathology is an empty bin
# INSIDE the occupied range — a hole — so report against that range, not the whole axis.
function occupied_range(nm)
    dM  = diff(M_col[nm])
    tol = 1e-6 * maximum(abs, dM)
    occ = findall(>(tol), dM)
    isempty(occ) && return nothing
    return (first(occ), last(occ))
end

# The well-mixed bottom layer piles ~40x the typical bin population into a single bin;
# on a full-range axis it flattens everything else to a flat line.  Clip to a robust
# upper bound so the bin-to-bin structure is legible (titles say "y clipped").
clip_hi(v; q=0.95, pad=1.25) =
    (pos = filter(>(0), abs.(v)); isempty(pos) ? 1.0 : quantile(pos, q) * pad)

# ---- Fig A: volume PDF histogram per column ---------------------------------
figA = Figure(size=(1000, 140 * length(col_names) + 90))
axesA = Axis[]
for (i, nm) in enumerate(col_names)
    ax = Axis(figA[i, 1], ylabel=nm,
              xlabel = i == length(col_names) ? "buoyancy b" : "")
    i < length(col_names) && hidexdecorations!(ax; ticks=false, grid=false)
    barplot!(ax, b_mid, pdf_col[nm], gap=0, color=:royalblue, strokewidth=0)
    ylims!(ax, 0, clip_hi(pdf_col[nm]))
    push!(axesA, ax)
end
linkxaxes!(axesA...)
Label(figA[0, :], "volume PDF  p(b) = dM/db  per column (y clipped) — $(tag) Ra=$(Ra_str)",
      fontsize=18)
outA = joinpath(plot_dir, "gmix_volume_pdf_columns.png")
save(outA, figA); println("saved figure → $outA")

# ---- optional: exact per-bin cell counts from one snapshot -------------------
# The PDF above differences a Float32 cumulative sum, so a genuinely empty bin can
# read as small-but-nonzero roundoff.  Histogramming a snapshot directly is exact and
# settles whether a bin truly holds zero cells.
counts_focus = nothing
if do_cell_counts
    ds1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
    x   = Float64.(ds1["x_caa"][:])
    wet = ds1["b"][:, :, :, 2] .!= 0          # dry cells are 0 at t=0 (see CLAUDE.md)
    close(ds1)

    # full-depth column mask = interior ∪ BL strip = the x-strip itself
    xlo, xhi = only((lo, hi) for (nm, lo, hi) in col_bounds if nm == focus_region)
    colmask  = (reshape(x, :, 1, 1) .>= xlo) .& (reshape(x, :, 1, 1) .< xhi)
    idxs     = findall(vec(colmask .& wet))

    dsb = NCDataset(joinpath(data_dir, "buoyancy_seg$(last(segments)).nc"))
    b_snap = Array{Float64}(dsb["b"][:, :, :, end])
    t_snap = Float64(dsb["time"][end])
    close(dsb)
    b_snap[.!wet] .= NaN                       # keep dry cells out of the statistics
    b_focus_blk = b_snap[findall((x .>= xlo) .& (x .< xhi)), :, :]   # [ncol, Ny, Nz]

    counts_focus = zeros(Int, length(b_edges) - 1)
    for v in vec(b_snap)[idxs]
        (isnan(v) || v < b_edges[1] || v >= b_edges[end]) && continue
        counts_focus[clamp(searchsortedlast(b_edges, v), 1, length(counts_focus))] += 1
    end
    @printf("  cell counts from buoyancy_seg%d at t = %.1f (%d wet cells in %s)\n",
            last(segments), t_snap, length(idxs), focus_region)
end

# ---- Fig B: focus-region zoom, three histograms sharing the b axis ------------
if b_zoom === :auto
    r = occupied_range(focus_region)
    r === nothing && error("focus region $focus_region holds no volume")
    blo, bhi = b_g[r[1]], b_g[r[2] + 1]
else
    blo, bhi = b_zoom
end
@printf("\nzoom range for %s: b ∈ [%.2f, %.2f]\n", focus_region, blo, bhi)

b_bincent = 0.5 .* (b_edges[1:end-1] .+ b_edges[2:end])
inzoom(v) = findall(x -> blo <= x <= bhi, v)
sel_mid, sel_cent, sel_bin = inzoom(b_mid), inzoom(b_g), inzoom(b_bincent)

# holes = empty bins inside the occupied range; shade them so the same b values can be
# followed down all three panels — every red stripe lines up with a G_mix zero.
holes = do_cell_counts ? sel_bin[counts_focus[sel_bin] .== 0] : Int[]
shade_holes!(ax) = for k in holes
    vspan!(ax, b_edges[k], b_edges[k+1], color=(:crimson, 0.22))
end

nrow = do_cell_counts ? 3 : 2
figB = Figure(size=(1100, 240 * nrow + 90))
axesB = Axis[]
row = 0

if do_cell_counts
    global row += 1
    ax = Axis(figB[row, 1], ylabel="cells per bin", title="grid cells per b-bin (y clipped)")
    shade_holes!(ax)
    barplot!(ax, b_bincent[sel_bin], counts_focus[sel_bin], gap=0, color=:darkorange)
    ylims!(ax, 0, clip_hi(counts_focus[sel_bin]))
    hidexdecorations!(ax; ticks=false, grid=false)
    push!(axesB, ax)
end

global row += 1
ax = Axis(figB[row, 1], ylabel="dM/db", title="volume PDF (y clipped)")
shade_holes!(ax)
barplot!(ax, b_mid[sel_mid], pdf_col[focus_region][sel_mid], gap=0, color=:royalblue)
ylims!(ax, 0, clip_hi(pdf_col[focus_region][sel_mid]))
hidexdecorations!(ax; ticks=false, grid=false)
push!(axesB, ax)

global row += 1
ax = Axis(figB[row, 1], ylabel="G_mix", xlabel="buoyancy b", title="G_mix (y clipped)")
shade_holes!(ax)
barplot!(ax, b_g[sel_cent], Gmix_col[focus_region][sel_cent], gap=0, color=:seagreen)
let g = clip_hi(Gmix_col[focus_region][sel_cent])
    ylims!(ax, -g, g)
end
push!(axesB, ax)

linkxaxes!(axesB...)
xlims!(axesB[end], blo - db_bin, bhi + db_bin)
Label(figB[0, :],
      @sprintf("%s — red stripes are HOLES: empty bins inside the occupied range (%s Ra=%s)",
               focus_region, tag, Ra_str), fontsize=18)
outB = joinpath(plot_dir, "gmix_volume_pdf_$(focus_region)_zoom.png")
save(outB, figB); println("saved figure → $outB")

# ---- quantify the undersampling ---------------------------------------------
println("\nholes inside each column's occupied b range (empty bins where the region DOES hold water):")
for nm in col_names
    rng = occupied_range(nm)
    if rng === nothing
        @printf("  %-8s  no volume anywhere\n", nm)
        continue
    end
    dM  = diff(M_col[nm])[rng[1]:rng[2]]
    tol = 1e-6 * maximum(abs, diff(M_col[nm]))
    nz  = count(<=(tol), dM)
    @printf("  %-8s  b ∈ [%+.2f, %+.2f]   %3d / %3d bins empty (%.0f%%)\n",
            nm, b_g[rng[1]], b_g[rng[2]+1], nz, length(dM), 100nz / length(dM))
end

if do_cell_counts
    nz = count(==(0), counts_focus[sel_bin])
    @printf("\n%s snapshot: %d / %d bins in the zoom hold ZERO grid cells (%.0f%%)\n",
            focus_region, nz, length(sel_bin), 100nz / length(sel_bin))
    occ = counts_focus[sel_bin][counts_focus[sel_bin] .> 0]
    if !isempty(occ)
        @printf("  occupied bins: median %d cells, min %d, max %d\n",
                round(Int, median(occ)), minimum(occ), maximum(occ))
    end

    # Why the staircase: in a quiescent, horizontally homogeneous column b ≈ b(z), so
    # each z-level's cells collapse into ONE bin.  Holes appear wherever the b gap
    # between ADJACENT z-levels exceeds the bin width — the b-resolution floor.
    lvl = [mean(filter(!isnan, vec(b_focus_blk[:, :, k]))) for k in 1:size(b_focus_blk, 3)]
    spread = [maximum(filter(!isnan, vec(b_focus_blk[:, :, k]))) -
              minimum(filter(!isnan, vec(b_focus_blk[:, :, k]))) for k in 1:size(b_focus_blk, 3)]
    gaps = filter(!isnan, abs.(diff(lvl)))
    @printf("\n%s stratification vs bin width (db = %.3f):\n", focus_region, db_bin)
    @printf("  horizontal b-spread within a z-level: median %.2e, max %.2e\n",
            median(spread), maximum(spread))
    @printf("  b gap between adjacent z-levels:      median %.2e, max %.2e\n",
            median(gaps), maximum(gaps))
    @printf("  → bins finer than ~%.3f leave holes here; %.0f%% of level gaps exceed db\n",
            maximum(gaps), 100 * count(>(db_bin), gaps) / length(gaps))
end
