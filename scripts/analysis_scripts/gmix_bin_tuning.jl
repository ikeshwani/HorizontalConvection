# Tune and verify the blended buoyancy axis used by gmix_CODF.jl.
#
# The CODF G_mix bins cells by buoyancy and takes one d/db of a cumulative sum.  In a
# quiescent region b ≈ b(z): each z-level's cells share one buoyancy and collapse into
# a single bin, while adjacent levels can sit further apart in b than one bin is wide.
# Bins then land between levels ("holes"), the cumulative sum goes flat, and G_mix
# reads exactly 0 there — the 0 ↔ negative oscillation.
#
# blended_b_edges places edges evenly in φ(b) = (1-λ)·b_norm + λ·F(b), widening bins
# where volume is thin.  λ is NOT universal — it depends on how concentrated the run's
# volume distribution is — so re-run this after changing Ra, topography or resolution.
#
# The axis is built from ONE snapshot and then frozen for the whole time series, so the
# score that matters is OUT-OF-SAMPLE: build on one snapshot, count holes on others.
#
# Read-only: writes one figure, touches no data.

using TopographicHorizontalConvection   # blended_b_edges
using NCDatasets
using CairoMakie
using Statistics
using Printf

# ---- config ----
experiment  = "hill"                     # "control" (flat bottom) or "hill" (3-hill GRC)
Ra          = 1e8                        # 1e8 or 1e6
n_b_bins    = 101                        # bin EDGES, matching gmix_CODF.jl
λ_list      = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
λ_plot      = [0.0, 0.5, 0.7, 1.0]       # λ values drawn on the edge figure
λ_recommend = 0.7                        # highlighted in the figure
ml_range    = (-0.35, -0.20)             # "mixed layer" window, for a resolution score
tail_b      = 0.3                        # bins with center above this count as "tail"

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

# gmix_CODF.jl builds its axis from the last step of the final segment — build here too
build_snap  = (last(segments), :last)
# earlier times, to score the frozen axis out-of-sample
verify_snaps = [(last(segments), 1), (last(segments) - 3, :last), (last(segments) - 6, :last)]

# ---- grid, wet mask, cell volumes -------------------------------------------
ds1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
x   = Float64.(ds1["x_caa"][:])
vol = reshape(Float64.(ds1["Δx_caa"][:]), Nx, 1, 1) .*
      reshape(Float64.(ds1["Δy_aca"][:]), 1, Ny, 1) .*
      reshape(Float64.(ds1["Δz_aac"][:]), 1, 1, Nz)
wet = ds1["b"][:, :, :, 2] .!= 0          # dry cells are 0 at t=0 (see CLAUDE.md)
close(ds1)
vol_full = vol .* ones(Nx, Ny, Nz)

# full-depth columns = interior ∪ BL strip = the x-strip itself; mirrors col_bounds in
# gmix_CODF.jl, plus the plume so this covers the whole domain
cols = [("plume", -Inf, -1.8), ("basin0", -1.8, -1.35), ("hill1", -1.35, -0.65),
        ("basin1", -0.65, -0.35), ("hill2", -0.35, 0.35), ("basin2", 0.35, 0.65),
        ("hill3", 0.65, 1.35), ("basin3", 1.35, Inf)]
region_idx = Dict(nm => findall(vec(((reshape(x, :, 1, 1) .>= lo) .&
                                     (reshape(x, :, 1, 1) .< hi)) .& wet))
                  for (nm, lo, hi) in cols)

function load_snap(seg, which)
    ds = NCDataset(joinpath(data_dir, "buoyancy_seg$(seg).nc"))
    k  = which === :last ? length(ds["time"][:]) : which
    b  = Array{Float64}(ds["b"][:, :, :, k])
    t  = Float64(ds["time"][k])
    close(ds)
    return b, t
end

b_build, t_build = load_snap(build_snap...)
snaps = [(load_snap(s, k)...,) for (s, k) in verify_snaps]
@printf("axis built from t = %.1f ; verified at t = %s\n",
        t_build, join([@sprintf("%.1f", t) for (_, t) in snaps], ", "))

wet_idx = findall(vec(wet))
b_all   = vec(b_build)[wet_idx]
v_all   = vec(vol_full)[wet_idx]

# A hole is an EMPTY bin inside the range the region occupies — bins beyond that range
# are empty only because the region holds no water there, which is not a defect.
function count_holes(b_region, edges)
    c = zeros(Int, length(edges) - 1)
    for v in b_region
        (v < edges[1] || v > edges[end]) && continue
        c[clamp(searchsortedlast(edges, v), 1, length(c))] += 1
    end
    occ = findall(>(0), c)
    isempty(occ) && return (0, 0)
    inner = c[first(occ):last(occ)]
    return (count(==(0), inner), length(inner))
end

# ---- λ sweep -----------------------------------------------------------------
println("\nholes = empty bins inside each region's occupied b range, summed over all $(length(cols)) regions")
@printf("%-8s %-6s %-4s %-7s", "λ", "nbins", "ML", "tail_w")
@printf(" %-12s", "build")
for (_, t) in snaps; @printf(" %-12s", @sprintf("t=%.0f", t)); end
println()

for λ in λ_list
    edges = blended_b_edges(b_all, v_all, n_b_bins; λ=λ)
    w     = diff(edges)
    ctr   = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    n_ml  = count(v -> ml_range[1] <= v <= ml_range[2], ctr)
    ti    = findall(>(tail_b), ctr)
    tw    = isempty(ti) ? NaN : maximum(w[ti])
    @printf("%-8.2f %-6d %-4d %-7.3f", λ, length(w), n_ml, tw)
    for (b_s, _) in [(b_build, t_build); snaps]
        h = 0; tot = 0
        for (nm, _, _) in cols
            a, b2 = count_holes(vec(b_s)[region_idx[nm]], edges); h += a; tot += b2
        end
        @printf(" %-12s", @sprintf("%d/%d", h, tot))
    end
    println()
end
@printf("\nML = bins centered in [%.2f, %.2f]; tail_w = widest bin centered above %.2f\n",
        ml_range[1], ml_range[2], tail_b)
println("pick the smallest λ with 0 holes out-of-sample; larger λ buys ML bins but widens the tail")

# ---- figure: where the cuts land vs where the water is ------------------------
# fine reference PDF, independent of any scheme under test (log y: the density spans
# ~4 decades, which is exactly why one uniform width cannot serve the whole axis)
bmin, bmax = minimum(b_all), maximum(b_all)
nfine = 400
efine = collect(range(bmin, bmax, length=nfine + 1))
vfine = zeros(nfine)
for (v, dv) in zip(b_all, v_all)
    vfine[clamp(searchsortedlast(efine, v), 1, nfine)] += dv
end
pdf_fine = vfine ./ diff(efine)
ctr_fine = 0.5 .* (efine[1:end-1] .+ efine[2:end])
pdf_plot = [p <= 0 ? NaN : p for p in pdf_fine]      # log axis cannot take zeros

fig = Figure(size=(1250, 190 * length(λ_plot) + 380))
axp = Axis(fig[1, 1], ylabel="volume PDF  dM/db", yscale=log10,
           title="where the water is (log scale — spans ~4 decades)")
stairs!(axp, ctr_fine, pdf_plot, color=:black, linewidth=1.4, step=:center)
hidexdecorations!(axp; ticks=false, grid=false)
axes_all = Any[axp]

for (i, λ) in enumerate(λ_plot)
    edges = blended_b_edges(b_all, v_all, n_b_bins; λ=λ)
    w     = diff(edges)
    ctr   = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    n_ml  = count(v -> ml_range[1] <= v <= ml_range[2], ctr)
    ti    = findall(>(tail_b), ctr)
    tw    = isempty(ti) ? NaN : maximum(w[ti])
    note  = λ == 0 ? " — uniform in b" :
            λ == 1 ? " — pure equal-volume (quantile)" :
            λ == λ_recommend ? " — recommended" : ""
    col   = λ == λ_recommend ? :seagreen : λ == 1 ? :crimson : λ == 0 ? :gray30 : :royalblue
    ax = Axis(fig[1 + i, 1],
              title=@sprintf("λ = %.1f%s   —   %d bins in the mixed layer, widest tail bin %.3f",
                             λ, note, n_ml, tw),
              titlesize=12, titlealign=:left, titlecolor=col,
              xlabel = i == length(λ_plot) ? "buoyancy b" : "")
    vlines!(ax, edges, color=col, linewidth=0.7)
    hideydecorations!(ax)
    i < length(λ_plot) && hidexdecorations!(ax; ticks=false, grid=false)
    ylims!(ax, 0, 1)
    push!(axes_all, ax)
end

linkxaxes!(axes_all...)
xlims!(axp, bmin, bmax)
rowsize!(fig.layout, 1, Relative(0.42))
Label(fig[0, :], "Each tick is one bin edge — the cuts follow the water as λ rises  ($(tag) Ra=$(Ra_str))",
      fontsize=18)

outfig = joinpath(plot_dir, "gmix_bin_edges_vs_pdf.png")
save(outfig, fig)
println("\nsaved figure → $outfig")
