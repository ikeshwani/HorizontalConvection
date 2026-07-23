# gmix_chi.jl
#
# Regional/column G_mix(b,t) via the ORIGINAL chi-based sorted-binning
# estimator (G_mix_calc in src/analysis/gmix.jl: sort cells by buoyancy,
# accumulate χ·dV, evaluate the cumulative integral at each buoyancy bin edge,
# then 0.5·d²/db² — no smoothing, per the removal of gaussian_smooth from that
# path).
#
# This mirrors gmix_CODF.jl's setup — same experiment/Ra config, same quantile
# buoyancy axis (blended_b_edges, λ=1 ⇒ pure equal-volume bins), same
# region/column decomposition (interior column + its boundary-layer strip,
# summed into a full-depth column), same interval-mean time averaging — but
# swaps CODF's diffusive-flux-convergence G_mix estimator for the chi
# estimator. Ψ, G_surface, and ∂M/∂t do NOT depend on the G_mix estimator, so
# they are NOT recomputed here: load them from the CODF output file
# (Gmix_quantile_regions_CODF_*.nc, variables psi_int_*/Gsurf_int_*/dMdt_*)
# and combine with this file's Gmix_int_* yourself for the chi-based residual.
#
# One structural difference from CODF: G_mix_calc's 0.5·d²/db² needs TWO
# b-derivatives of a once-integrated quantity (χ·dV), so it naturally lands on
# the INTERIOR bin edges b_edges[2:end-1] — not the bin centers CODF uses for
# M/Ψ/G_surface (CODF's CONV_dV is already once-differentiated in space, so it
# only needs one more b-derivative, landing on centers). `b` in this file's
# output is b_edges[2:end-1]; interpolate onto CODF's b_centers if you need an
# exact per-b combination for the residual.
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/gmix_chi.jl

using TopographicHorizontalConvection   # region masks + blended_b_edges physics
using NCDatasets
using CairoMakie
using NaNStatistics
using Statistics
using Printf

# ---- config (mirrors gmix_CODF.jl) ----
experiment = "control"          # "control" (flat bottom) or "hill" (3-hill GRC)

Ra       = 1e8                  # 1e8 or 1e6
n_b_bins = 101                  # number of bin EDGES (→ n_b_bins-2 interior-edge points)

bin_mode = :blended             # quantile axis only (uniform not supported here)
λ_bins   = 1.0                  # pure quantile (equal-volume) bins

# Chunk the record into blocks of this many output steps (steps_per_interval+1
# snapshots, boundary snapshot shared between consecutive blocks) and take a
# plain snapshot MEAN of G_mix over each block — same tiling as gmix_CODF.jl's
# interval_mean, so intervals line up 1:1 with that file's Gmix_int_*/
# psi_int_*/Gsurf_int_*/dMdt_* for a term-by-term residual. No dMdt here:
# unlike M(b), the chi estimator has no natural cumulative "mass" whose
# endpoint difference gives an exact ∂/∂t — G_mix(b,t) is just averaged.
steps_per_interval = 100

# Ra → path tag + filename tag
if Ra == 1e8
    Ra_tag, Ra_str = "ra1e8", "RA1e8"
elseif Ra == 1e6
    Ra_tag, Ra_str = "ra1e6", "RA1e6"
else
    error("unsupported Ra: $Ra (use 1e8 or 1e6)")
end

if experiment == "control"
    topo     = "flat"
    tag      = "Control"
    segments = Ra == 1e8 ? (1:21) : (1:7)
elseif experiment == "hill"
    topo     = "threehill"
    tag      = "3hill"
    segments = Ra == 1e8 ? (1:28) : (1:10)   # bump as more segments land
else
    error("unknown experiment: $experiment (use \"control\" or \"hill\")")
end

data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/$(Ra_tag)_4xstretch_$(topo)_baseforcing_zerostart/"
plot_dir = joinpath(data_dir, "figures")
mkpath(plot_dir)

outfile = joinpath(data_dir, "Gmix_chi_quantile_regions_$(tag)_$(Ra_str)_seg$(first(segments))to$(last(segments)).nc")

# ---- load grid info from seg1 ----
ds1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))

Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
Lx = ds1.attrib["Lx"]

x = ds1["x_caa"][:]
z = ds1["z_aac"][:]

Δx_center = reshape(ds1["Δx_caa"][:], Nx, 1, 1)
Δy_center = reshape(ds1["Δy_aca"][:], 1, Ny, 1)
Δz_center = reshape(ds1["Δz_aac"][:], 1, 1, Nz)
vol = Δx_center .* Δy_center .* Δz_center

# wet mask from the second time step (immersed cells are b=0 at t=0, same as IC)
wet = ds1["b"][:, :, :, 2] .!= 0
close(ds1)

# ---- Pass 0: build the buoyancy axis (same recipe as gmix_CODF.jl) ----------
# Frozen for every segment/timestep, built from one quasi-steady snapshot (the
# last step of the final segment) — see gmix_CODF.jl for the out-of-sample
# stability check.
if bin_mode != :blended
    error("gmix_chi.jl only supports bin_mode = :blended (quantile axis)")
end
println("Pass 0: building blended buoyancy axis (λ = $λ_bins) from seg$(last(segments))...")
ds_ax  = NCDataset(joinpath(data_dir, "buoyancy_seg$(last(segments)).nc"))
b_ax   = Array{Float64}(ds_ax["b"][:, :, :, end])
t_ax   = Float64(ds_ax["time"][end])
close(ds_ax)
wet_idx = findall(vec(wet))
b_edges = blended_b_edges(vec(b_ax)[wet_idx], vec(vol)[wet_idx], n_b_bins; λ=λ_bins)
@printf("  axis from t = %.1f: %d edges, width min %.4f / median %.4f / max %.4f\n",
        t_ax, length(b_edges), minimum(diff(b_edges)),
        median(diff(b_edges)), maximum(diff(b_edges)))
b_ax = nothing; GC.gc()

n_b_bins = length(b_edges)             # blended_b_edges drops ties, so re-read the count
b_out    = b_edges[2:end-1]            # G_mix_calc's native output axis (interior edges)
n_b_out  = length(b_out)

# ---- region geometry (same split as gmix_CODF.jl: interior column + its BL strip) ----
ΔA_2d = dropdims(Δx_center .* Δy_center, dims=3)
zBL   = boundary_layer_depth(Lx, Ra)
X = reshape(x, :, 1, 1);  Z = reshape(z, 1, 1, :)

col_bounds = [
    ("basin0", -1.8,  -1.35), ("hill1", -1.35, -0.65), ("basin1", -0.65, -0.35),
    ("hill2",  -0.35,  0.35), ("basin2",  0.35,  0.65), ("hill3",   0.65,  1.35),
    ("basin3",  1.35,  Inf),
]
bl_masks = [("bl_$(nm)", (X .>= xlo) .& (X .< xhi) .& (Z .> zBL))
            for (nm, xlo, xhi) in col_bounds]

base_masks     = gmix_region_masks(x, z, Lx, Ra)
region_masks   = vcat([(nm, m) for (nm, m) in base_masks if nm != "boundary_layer"],
                      bl_masks)
region_precomp = precompute_regions(region_masks, ΔA_2d, wet)
region_names   = [r.name for r in region_precomp]
col_names      = [nm for (nm, _, _) in col_bounds]

# ---- non-uniform generalization of G_mix_calc's 0.5·d²/db² -----------------
# y'' at the interior points of x, via the standard 3-point non-uniform
# stencil. Reduces EXACTLY to G_mix_calc's `diff(diff(y)) ./ db^2` when x is
# evenly spaced (h1 = h2 = db): 2*((y2-y1)/db - (y1-y0)/db)/(2db)
#                              = (y2 - 2y1 + y0)/db^2.
function second_deriv_nonuniform(y::AbstractVector, x::AbstractVector)
    n = length(x)
    out = zeros(Float64, n - 2)
    for i in 2:n-1
        h1 = x[i]   - x[i-1]
        h2 = x[i+1] - x[i]
        out[i-1] = 2 * ((y[i+1] - y[i]) / h2 - (y[i] - y[i-1]) / h1) / (h1 + h2)
    end
    return out
end

# ---- G_mix(b) over one region's cells, on the blended axis ------------------
# Same sort-and-accumulate-χ·dV algorithm as G_mix_calc: cumulative ∫χ·dV
# evaluated at each bin edge via binary search on the sorted buoyancy
# (searchsortedlast), then 0.5·d²/db² — just fed the pre-built non-uniform
# b_edges instead of a uniform range, and second_deriv_nonuniform instead of
# the constant-db stencil. No smoothing.
function gmix_chi_region(b, χdV, idxs)
    bg = vec(b)[idxs]
    cg = vec(χdV)[idxs]
    perm     = sortperm(bg)
    b_sorted = bg[perm]
    cum_χdV  = cumsum(cg[perm])

    M = zeros(Float64, n_b_bins)
    for (n, edge) in enumerate(b_edges)
        idx = searchsortedlast(b_sorted, edge)
        M[n] = idx > 0 ? cum_χdV[idx] : 0.0
    end
    return 0.5 .* second_deriv_nonuniform(M, b_edges)
end

# plain snapshot mean over each interval block (see steps_per_interval above)
interval_mean(A, i_start, i_end) = reduce(hcat,
    [vec(nanmean(A[:, i_start[k]:i_end[k]], dims=2)) for k in eachindex(i_start)])

# ---- Pass 1: build the global time vector, deduplicating segment overlaps ---
println("Pass 1: scanning time vectors from segments $(first(segments))–$(last(segments))...")
time_all = Float64[]
let t_last = -Inf
    for s in segments
        bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
        t_seg = Float64.(bfile["time"][:])
        close(bfile)
        valid = findall(t_seg .> t_last)
        # clip to the velocity file's time length, exactly as gmix_CODF.jl does,
        # so this file's time vector (and interval boundaries) line up 1:1 with
        # the CODF file's — even though this script never reads velocities.
        vfile0 = NCDataset(joinpath(data_dir, "velocities_seg$(s).nc"))
        nt_v   = length(vfile0["time"][:])
        close(vfile0)
        valid  = filter(≤(nt_v), valid)
        isempty(valid) && continue
        append!(time_all, t_seg[valid])
        t_last = t_seg[valid[end]]
    end
end
times = time_all
Nt    = length(times)
println("total time steps: $Nt  (t = $(times[1]) → $(times[end]))")

Gmix_regions = Dict(name => zeros(Float32, n_b_out, Nt) for name in region_names)

# ---- Pass 2: compute chi G_mix one segment at a time ------------------------
println("Pass 2: computing chi-based G_mix segment by segment...")
let t_last = -Inf, t_offset = 0
for s in segments
    bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
    t_seg = Float64.(bfile["time"][:])
    valid = findall(t_seg .> t_last)

    vfile0 = NCDataset(joinpath(data_dir, "velocities_seg$(s).nc"))
    nt_v   = length(vfile0["time"][:])
    close(vfile0)
    valid  = filter(≤(nt_v), valid)

    if isempty(valid)
        @printf("  seg %d: all steps are duplicates — skipping\n", s)
        close(bfile)
        continue
    end

    n_skip = valid[1] - 1
    n_skip > 0 && @printf("  seg %d: skipping first %d overlapping step(s)\n", s, n_skip)

    t_range = valid[1]:valid[end]
    nt      = length(t_range)
    gi      = t_offset+1 : t_offset+nt
    @printf("  seg %d: loading %d steps (t = %.2f → %.2f)...\n",
            s, nt, t_seg[t_range[1]], t_seg[t_range[end]])

    b_seg = Array(bfile["b"][:, :, :, t_range])       # native (Float32) precision
    χ_seg = Array(bfile["chi"][:, :, :, t_range])
    close(bfile)

    for (ti, g) in enumerate(gi)
        b   = Float64.(@view b_seg[:, :, :, ti])
        χdV = Float64.(@view(χ_seg[:, :, :, ti])) .* vol
        for r in region_precomp
            Gmix_regions[r.name][:, g] .= gmix_chi_region(b, χdV, r.idxs)
        end
    end

    t_last   = t_seg[t_range[end]]
    t_offset += nt
    b_seg = nothing; χ_seg = nothing; GC.gc()
    @printf("  seg %d: done\n", s)
end  # for s
end  # let

# ---- interval-mean per full column (interior + its BL strip) ----------------
n_int = (Nt - 1) ÷ steps_per_interval
if n_int == 0
    i_start, i_end, n_int = [1], [Nt], 1
else
    i_start = [(k - 1) * steps_per_interval + 1 for k in 1:n_int]
    i_end   = i_start .+ steps_per_interval
    if i_end[end] < Nt   # leftover steps → shorter final interval, always kept
        push!(i_start, i_end[end]);  push!(i_end, Nt);  n_int += 1
    end
end
t_start = times[i_start]
t_end   = times[i_end]
t_mid   = 0.5 .* (t_start .+ t_end)
@printf("interval mean: %d interval(s), t = %.1f → %.1f; final interval: %d step(s), t = %.1f → %.1f\n",
        n_int, t_start[1], t_end[end], i_end[end] - i_start[end], t_start[end], t_end[end])

G_cols = Dict{String,Matrix{Float64}}()   # full time resolution [n_b_out, Nt]
G_int  = Dict{String,Matrix{Float64}}()   # per-interval mean    [n_b_out, n_int]
for (nm, _, _) in col_bounds
    G_col      = Float64.(Gmix_regions[nm] .+ Gmix_regions["bl_"*nm])
    G_cols[nm] = G_col
    G_int[nm]  = interval_mean(G_col, i_start, i_end)
end

# time-mean per region over the whole run (sanity check: bl_* should dominate)
Gmix_region_mean = Dict(name => vec(nanmean(Gmix_regions[name], dims=2))
                        for name in region_names)

# ---- save G_mix(b, t) per region + per-column interval means to NetCDF -----
NCDataset(outfile, "c") do dsout
    defDim(dsout, "b",      n_b_out)
    defDim(dsout, "b_edge", n_b_bins)
    defDim(dsout, "time",   Nt)
    defDim(dsout, "interval", n_int)
    defVar(dsout, "b",       collect(b_out),   ("b",))
    defVar(dsout, "b_edges", collect(b_edges), ("b_edge",))
    defVar(dsout, "time",    times,            ("time",))
    defVar(dsout, "t_start", t_start, ("interval",))
    defVar(dsout, "t_end",   t_end,   ("interval",))
    defVar(dsout, "t_mid",   t_mid,   ("interval",))
    for name in region_names
        defVar(dsout, "Gmix_$(name)", Gmix_regions[name], ("b", "time"))
    end
    for nm in col_names   # full-depth column (interior + BL strip)
        defVar(dsout, "Gmix_col_$(nm)", Float32.(G_cols[nm]), ("b", "time"))
        defVar(dsout, "Gmix_int_$(nm)", Float32.(G_int[nm]),  ("b", "interval"))
    end
    dsout.attrib["method"] = "chi-based sorted-binning G_mix (G_mix_calc: 0.5*d2/db2 of cumulative chi*dV, no smoothing), quantile buoyancy axis, non-uniform-edge stencil, interval-mean averaged"
    dsout.attrib["steps_per_interval"] = steps_per_interval
    dsout.attrib["Ra"] = Ra
    dsout.attrib["segments"] = "$(first(segments))-$(last(segments))"
    dsout.attrib["bin_mode"] = String(bin_mode)
    dsout.attrib["lambda"] = λ_bins
    dsout.attrib["note"] = "b axis is b_edges[2:end-1] (interior edges), NOT bin centers -- see file header. psi/G_surface/dMdt are not recomputed here; combine with the CODF file's *_int_* variables for the residual."
end
println("saved → $outfile")

# per-region magnitude check: boundary-layer strips should dominate
for name in region_names
    @printf("  %-16s  max|G_mix| = %.3e\n", name, maximum(abs.(Gmix_region_mean[name])))
end

fig2  = Figure(size=(230 * length(region_names), 520))
axes2 = Axis[]
for (i, name) in enumerate(region_names)
    ax = Axis(fig2[1, i], xlabel="⟨G_mix⟩", ylabel="b", title=name)
    push!(axes2, ax)
    i > 1 && hideydecorations!(ax; ticks=false, grid=false)
    lines!(ax, Gmix_region_mean[name], b_out, color=:seagreen, linewidth=2)
    vlines!(ax, 0.0, color=:gray, linestyle=:dash, linewidth=0.8)
end
linkyaxes!(axes2...)

outpng2 = joinpath(plot_dir, "gmix_chi_regions_tmean.png")
save(outpng2, fig2)
println("saved figure → $outpng2")

# ---- quick look: per-column G_mix for the last (equilibrium) interval -------
kk = n_int
fig3  = Figure(size=(230 * length(col_names), 520))
axes3 = Axis[]
for (i, nm) in enumerate(col_names)
    ax = Axis(fig3[1, i], xlabel="G_mix (chi)", ylabel="b",
              title=@sprintf("%s  t = %.0f–%.0f", nm, t_start[kk], t_end[kk]))
    push!(axes3, ax)
    i > 1 && hideydecorations!(ax; ticks=false, grid=false)
    lines!(ax, G_int[nm][:, kk], b_out, color=:seagreen, linewidth=2)
    vlines!(ax, 0.0, color=:gray, linestyle=:dash, linewidth=0.8)
end
linkyaxes!(axes3...)
outpng3 = joinpath(plot_dir, "gmix_chi_columns_lastinterval.png")
save(outpng3, fig3)
println("saved figure → $outpng3")
