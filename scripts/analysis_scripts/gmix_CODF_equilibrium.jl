using TopographicHorizontalConvection
using NCDatasets
using NaNStatistics
using Statistics
using Printf

# ==============================================================================
# gmix_CODF_equilibrium.jl
#
# Same CODF (convergence-of-diffusive-flux) budget as gmix_CODF.jl, but scoped
# to ONLY the final segment (the equilibrium period lives entirely there) and
# with the buoyancy axis actively rebuilt every `steps_per_interval` output
# steps instead of frozen once from a single snapshot.
#
# This targets the sparse-tail binning bug found in last week's b0/b1 case
# study (a real point at b=-0.355 falling into an oversized terminal bin,
# ~30% error between Gmix_int and Gmix_from_FBN while averaging over only 3
# bins): the old full-run axis was built from ONE instantaneous snapshot (the
# last step of the last segment) and never saw buoyancy extremes that only
# existed near the actual analysis window.
#
# Restricting the active rebinning to just the final segment (~150 snapshots,
# not all ~9800 across the whole run) keeps this cheap: a few axis-pooling
# sorts plus one segment's worth of per-snapshot region binning, instead of
# hours for the full run.
#
# Two buoyancy axes are used:
#   - a LOCAL axis per steps_per_interval chunk, built by pooling every
#     snapshot in that chunk. This is what the region binning is actually
#     computed on, so each chunk gets resolution matched to its own buoyancy
#     distribution at that moment.
#   - one COMMON axis, built by pooling every snapshot in the final segment,
#     used only as the storage/output grid. Each chunk's local-axis result is
#     linearly interpolated onto it before writing, so the output file has one
#     shared, comparable `b` dimension across chunks (same requirement as the
#     full CODF file: dM/dt differences M between chunk endpoints).
#
# The full-run Hovmöller (gmix_CODF.jl, all segments) is untouched by this
# script and keeps its existing single-snapshot axis -- that's fine for the
# broad time-evolution picture. This script only serves the equilibrium-time
# consumers: the volume budget plots and the Bryden & Nurser analysis.
# ==============================================================================

# ---- config ----
experiment = "hill"          # "control" (flat bottom) or "hill" (3-hill GRC)
Ra         = 1e8             # 1e8 or 1e6
n_b_bins   = 801             # bin EDGES, both LOCAL (per chunk) and COMMON axes
λ_bins     = 1.0             # blended-axis λ (1.0 = pure volume quantile)

steps_per_interval = 100     # rebuild the local axis every this many output steps

if Ra == 1e8
    Ra_tag, Ra_str = "ra1e8", "RA1e8"
elseif Ra == 1e6
    Ra_tag, Ra_str = "ra1e6", "RA1e6"
else
    error("unsupported Ra: $Ra (use 1e8 or 1e6)")
end

if experiment == "control"
    topo, tag = "flat", "Control"
    final_segment = Ra == 1e8 ? 28 : 19
elseif experiment == "hill"
    topo, tag = "threehill", "3hill"
    final_segment = Ra == 1e8 ? 34 : 22
else
    error("unknown experiment: $experiment (use \"control\" or \"hill\")")
end

data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/$(Ra_tag)_4xstretch_$(topo)_baseforcing_zerostart/"
plot_dir = joinpath(data_dir, "figures")
mkpath(plot_dir)

outfile = joinpath(data_dir, "Gmix_quantile_regions_CODF_$(tag)_$(Ra_str)_seg$(final_segment)_equilibrium_nbins$(n_b_bins).nc")
# never clobber an existing file (this must never touch the full-run CODF output)
isfile(outfile) && error("refusing to run: output file already exists: $outfile")

# ---- load grid + metadata (mirrors gmix_CODF.jl) ----
ds1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
Ra_meta, Pr, b★, H = ds1.attrib["Ra"], ds1.attrib["Pr"], ds1.attrib["b★"], ds1.attrib["H"]
Lx = ds1.attrib["Lx"]

x = ds1["x_caa"][:]
z = ds1["z_aac"][:]

Δx_face = ds1["Δx_faa"][:]
Δy_face = ds1["Δy_afa"][:]
Δz_face = ds1["Δz_aaf"][:]

Δx_center = reshape(ds1["Δx_caa"][:], Nx, 1, 1)
Δy_center = reshape(ds1["Δy_aca"][:], 1, Ny, 1)
Δz_center = reshape(ds1["Δz_aac"][:], 1, 1, Nz)

Δy_vec = ds1["Δy_aca"][:]     # un-reshaped, for get_ψb_sort
Δz_vec = ds1["Δz_aac"][:]

vol = Δx_center .* Δy_center .* Δz_center
ν = sqrt(Pr * b★ * H^3 / Ra)
κ = ν / Pr

wet = ds1["b"][:, :, :, 2] .!= 0
close(ds1)
wet_idx = findall(vec(wet))
@printf("grid: %d x %d x %d, wet fraction = %.3f\n", Nx, Ny, Nz, length(wet_idx) / (Nx*Ny*Nz))

# ---- region geometry (mirrors gmix_CODF.jl exactly, so region names line up
# with the ones in the full-run CODF file) ----
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
col_iranges    = Dict(nm => findall((x .>= xlo) .& (x .< xhi))
                      for (nm, xlo, xhi) in col_bounds)

b_surf  = b★ .* tanh.(3 .* (x .+ Lx/3))
dz_half = 0.5 * Δz_center[1, 1, Nz]

# ---- per-snapshot per-cell convergence  ∫_cell(-∇·F)dV  [Nx,Ny,Nz] -----------
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

# G_mix(b) over one region's cells, on an EXPLICIT local axis
function gmix_region(b, CONV_dV, idxs, b_edges)
    bg = vec(b)[idxs]; cg = vec(CONV_dV)[idxs]
    M  = [sum(@view cg[bg .< be]) for be in b_edges]
    return -diff(M) ./ diff(b_edges)
end

# cumulative volume M(b) below each b_center, on an explicit local axis
function volume_below(b, idxs, b_centers)
    bg = vec(b)[idxs]; vg = vec(vol)[idxs]
    return [sum(@view vg[bg .< bc]) for bc in b_centers]
end

# surface-forcing transformation G_surface(b), on an explicit local axis
function gsurface_col(b_top, irange, b_edges)
    blk   = b_top[irange, :]
    Tsurf = κ .* (reshape(b_surf[irange], :, 1) .- blk) ./ dz_half .* ΔA_2d[irange, :]
    bt = vec(blk); tg = vec(Tsurf)
    M  = [sum(@view tg[bt .< be]) for be in b_edges]
    return -diff(M) ./ diff(b_edges)
end

# linear interpolation of a bin-center-indexed series y(x) onto new points xq;
# flat extrapolation past the ends. x must be sorted ascending.
function interp_series(x, y, xq)
    n  = length(x)
    yq = zeros(Float64, length(xq))
    for (i, xi) in enumerate(xq)
        if xi <= x[1]
            yq[i] = y[1]
        elseif xi >= x[n]
            yq[i] = y[n]
        else
            j = clamp(searchsortedlast(x, xi), 1, n - 1)
            frac = (xi - x[j]) / (x[j+1] - x[j])
            yq[i] = y[j] + frac * (y[j+1] - y[j])
        end
    end
    return yq
end

# ---- load the final segment in full -------------------------------------------
println("loading final segment $(final_segment)...")
bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(final_segment).nc"))
t_seg_raw = Float64.(bfile["time"][:])
b_seg_raw = Array(bfile["b"][:, :, :, :])
close(bfile)

vfile = NCDataset(joinpath(data_dir, "velocities_seg$(final_segment).nc"))
t_v_raw   = Float64.(vfile["time"][:])
u_seg_raw = Array(vfile["u"][1:Nx, :, :, :])
close(vfile)

# a segment can die after writing one more buoyancy snapshot than velocity
nt    = min(length(t_seg_raw), length(t_v_raw))
times = t_seg_raw[1:nt]
b_seg = b_seg_raw[:, :, :, 1:nt]
u_seg = u_seg_raw[:, :, :, 1:nt]
b_seg_raw = nothing; u_seg_raw = nothing; GC.gc()
@printf("  segment %d: %d steps, t = %.2f -> %.2f\n", final_segment, nt, times[1], times[end])

# ---- chunk boundaries within this segment (mirrors gmix_CODF.jl Step 5) -------
n_int = (nt - 1) ÷ steps_per_interval
if n_int == 0
    i_start, i_end, n_int = [1], [nt], 1
else
    i_start = [(k - 1) * steps_per_interval + 1 for k in 1:n_int]
    i_end   = i_start .+ steps_per_interval
    if i_end[end] < nt   # leftover steps -> shorter final chunk, no data dropped
        push!(i_start, i_end[end]); push!(i_end, nt); n_int += 1
    end
end
t_start = times[i_start]; t_end = times[i_end]; t_mid = 0.5 .* (t_start .+ t_end)
@printf("%d chunk(s) within segment %d, t = %.2f -> %.2f\n",
        n_int, final_segment, t_start[1], t_end[end])
for k in 1:n_int
    @printf("  chunk %d: steps %d:%d, t = %.2f -> %.2f (%d steps)\n",
            k, i_start[k], i_end[k], t_start[k], t_end[k], i_end[k]-i_start[k]+1)
end

# ---- COMMON output axis: pool every snapshot in the final segment -------------
println("building common output axis (pooling all $nt snapshots of segment $(final_segment))...")
b_pool   = reduce(vcat, [vec(Float64.(b_seg[:, :, :, ti]))[wet_idx] for ti in 1:nt])
vol_pool = repeat(vec(vol)[wet_idx], nt)
b_edges_common = blended_b_edges(b_pool, vol_pool, n_b_bins; λ=λ_bins)
b_pool = nothing; vol_pool = nothing; GC.gc()
n_b_common       = length(b_edges_common)
b_centers_common = 0.5 .* (b_edges_common[1:end-1] .+ b_edges_common[2:end])
nb_common        = n_b_common - 1
@printf("  common axis: %d edges, width min %.4f / median %.4f / max %.4f\n",
        n_b_common, minimum(diff(b_edges_common)), median(diff(b_edges_common)),
        maximum(diff(b_edges_common)))

# ---- pre-allocate full-time-resolution + interval outputs on the common axis --
Gmix_regions = Dict(name => zeros(Float64, nb_common, nt) for name in region_names)
M_regions    = Dict(name => zeros(Float64, nb_common, nt) for name in region_names)
Gsurf_cols   = Dict(nm => zeros(Float64, nb_common, nt) for nm in col_names)
ψ_b          = zeros(Float32, Nx, nb_common, nt)

G_int     = Dict(nm => zeros(Float64, nb_common, n_int) for nm in col_names)
Ψ_int     = Dict(nm => zeros(Float64, nb_common, n_int) for nm in col_names)
Gsurf_int = Dict(nm => zeros(Float64, nb_common, n_int) for nm in col_names)
dMdt_cols = Dict(nm => zeros(Float64, nb_common, n_int) for nm in col_names)
R_cols    = Dict(nm => zeros(Float64, nb_common, n_int) for nm in col_names)

# ---- per-chunk: fresh local axis, compute, interpolate onto the common axis ---
println("computing CODF budget per chunk, with an actively-rebuilt local axis...")
for k in 1:n_int
    idx_range = i_start[k]:i_end[k]
    nt_chunk  = length(idx_range)
    # skip the FIRST step of every chunk after the first: it's the shared
    # endpoint already written by the previous chunk (same overlap-dedup
    # convention gmix_CODF.jl uses across segments)
    write_range = k == 1 ? idx_range : (i_start[k]+1):i_end[k]

    @printf("  chunk %d/%d: pooling %d snapshots for a fresh local axis...\n", k, n_int, nt_chunk)
    b_pool_k   = reduce(vcat, [vec(Float64.(b_seg[:, :, :, ti]))[wet_idx] for ti in idx_range])
    vol_pool_k = repeat(vec(vol)[wet_idx], nt_chunk)
    b_edges_k  = blended_b_edges(b_pool_k, vol_pool_k, n_b_bins; λ=λ_bins)
    b_pool_k = nothing; vol_pool_k = nothing; GC.gc()
    nb_k        = length(b_edges_k) - 1
    b_centers_k = 0.5 .* (b_edges_k[1:end-1] .+ b_edges_k[2:end])
    @printf("    local axis: %d edges, width min %.4f / median %.4f / max %.4f\n",
            length(b_edges_k), minimum(diff(b_edges_k)), median(diff(b_edges_k)), maximum(diff(b_edges_k)))

    Gmix_chunk  = Dict(name => zeros(Float64, nb_k, nt_chunk) for name in region_names)
    M_chunk     = Dict(name => zeros(Float64, nb_k, nt_chunk) for name in region_names)
    Gsurf_chunk = Dict(nm => zeros(Float64, nb_k, nt_chunk) for nm in col_names)

    for (li, gi) in enumerate(idx_range)
        b = Float64.(b_seg[:, :, :, gi])
        b[.!wet] .= NaN
        CONV_dV = conv_dV_snapshot(b)
        b_top   = b[:, :, Nz]
        for r in region_precomp
            Gmix_chunk[r.name][:, li] .= gmix_region(b, CONV_dV, r.idxs, b_edges_k)
            M_chunk[r.name][:, li]    .= volume_below(b, r.idxs, b_centers_k)
        end
        for (nm, ir) in col_iranges
            Gsurf_chunk[nm][:, li] .= gsurface_col(b_top, ir, b_edges_k)
        end
    end

    # ψ_b(x, b, t), evaluated directly at this chunk's local b_centers
    ψ_chunk, _ = get_ψb_sort(b_seg[:, :, :, idx_range], u_seg[:, :, :, idx_range],
                             Δy_vec, Δz_vec, Nx, Ny, Nz, nt_chunk; b_bins=b_centers_k)

    # interpolate this chunk's full-time-resolution fields onto the common axis
    for name in region_names, (li, gi) in enumerate(idx_range)
        gi in write_range || continue
        Gmix_regions[name][:, gi] .= interp_series(b_centers_k, Gmix_chunk[name][:, li], b_centers_common)
        M_regions[name][:, gi]    .= interp_series(b_centers_k, M_chunk[name][:, li],    b_centers_common)
    end
    for nm in col_names, (li, gi) in enumerate(idx_range)
        gi in write_range || continue
        Gsurf_cols[nm][:, gi] .= interp_series(b_centers_k, Gsurf_chunk[nm][:, li], b_centers_common)
    end
    for (li, gi) in enumerate(idx_range)
        gi in write_range || continue
        for ix in 1:Nx
            ψ_b[ix, :, gi] .= Float32.(interp_series(b_centers_k, Float64.(ψ_chunk[ix, :, li]), b_centers_common))
        end
    end

    # per-column chunk-mean budget, on the LOCAL axis first, then interpolated
    for (nm, xlo, xhi) in col_bounds
        G_col_k = Gmix_chunk[nm] .+ Gmix_chunk["bl_"*nm]
        M_col_k = M_chunk[nm]    .+ M_chunk["bl_"*nm]
        iL = nearest_xi(x, xlo); iR = nearest_xi(x, min(xhi, x[end]))
        ψ_in_left  = -ψ_chunk[iL, :, :]
        ψ_in_right = ψ_chunk[iR, :, :]
        Ψ_col_k    = Float64.(ψ_in_left .+ ψ_in_right)

        dMdt_k      = (M_col_k[:, end] .- M_col_k[:, 1]) ./ (times[idx_range[end]] - times[idx_range[1]])
        G_int_k     = vec(nanmean(G_col_k, dims=2))
        Ψ_int_k     = vec(nanmean(Ψ_col_k, dims=2))
        Gsurf_int_k = vec(nanmean(Gsurf_chunk[nm], dims=2))
        R_k         = dMdt_k .- G_int_k .- Ψ_int_k .- Gsurf_int_k

        G_int[nm][:, k]     .= interp_series(b_centers_k, G_int_k, b_centers_common)
        Ψ_int[nm][:, k]     .= interp_series(b_centers_k, Ψ_int_k, b_centers_common)
        Gsurf_int[nm][:, k] .= interp_series(b_centers_k, Gsurf_int_k, b_centers_common)
        dMdt_cols[nm][:, k] .= interp_series(b_centers_k, dMdt_k, b_centers_common)
        R_cols[nm][:, k]    .= interp_series(b_centers_k, R_k, b_centers_common)

        # telescoping sanity check, on the common axis post-interpolation
        ΔM_sum = dMdt_cols[nm][:, k] .* (t_end[k] - t_start[k])
        ΔM_tot = interp_series(b_centers_k, M_col_k[:, end] .- M_col_k[:, 1], b_centers_common)
        @printf("    %-8s  telescoping max|err| = %.3e\n", nm, maximum(abs.(ΔM_sum .- ΔM_tot)))
    end
    println("  chunk $k/$n_int done")
    Gmix_chunk = nothing; M_chunk = nothing; Gsurf_chunk = nothing; ψ_chunk = nothing; GC.gc()
end

# ---- full-time-resolution column fields, derived once from the assembled
# common-axis region fields (no need to duplicate this per chunk) -------------
G_cols = Dict{String,Matrix{Float64}}()
M_cols = Dict{String,Matrix{Float64}}()
Ψ_cols = Dict{String,Matrix{Float64}}()
for (nm, xlo, xhi) in col_bounds
    G_cols[nm] = Gmix_regions[nm] .+ Gmix_regions["bl_"*nm]
    M_cols[nm] = M_regions[nm]    .+ M_regions["bl_"*nm]
    iL = nearest_xi(x, xlo); iR = nearest_xi(x, min(xhi, x[end]))
    ψ_in_left  = -Float64.(ψ_b[iL, :, :])
    ψ_in_right =  Float64.(ψ_b[iR, :, :])
    Ψ_cols[nm] = ψ_in_left .+ ψ_in_right
end

# ---- save to NetCDF, same variable layout as gmix_CODF.jl's output ------------
NCDataset(outfile, "c") do dsout
    defDim(dsout, "b",      nb_common)
    defDim(dsout, "b_edge", n_b_common)
    defDim(dsout, "time",   nt)
    defDim(dsout, "x",      Nx)
    defDim(dsout, "interval", n_int)
    defVar(dsout, "b",       collect(b_centers_common), ("b",))
    defVar(dsout, "b_edges", collect(b_edges_common),   ("b_edge",))
    defVar(dsout, "time",    times,                     ("time",))
    defVar(dsout, "x",       Float64.(x),                ("x",))
    defVar(dsout, "t_start", t_start, ("interval",))
    defVar(dsout, "t_end",   t_end,   ("interval",))
    defVar(dsout, "t_mid",   t_mid,   ("interval",))
    for name in region_names
        defVar(dsout, "Gmix_$(name)", Float32.(Gmix_regions[name]), ("b", "time"))
        defVar(dsout, "M_$(name)",    Float32.(M_regions[name]),    ("b", "time"))
    end
    defVar(dsout, "psi_b", ψ_b, ("x", "b", "time"))
    for nm in col_names
        defVar(dsout, "Gsurf_$(nm)",     Float32.(Gsurf_cols[nm]), ("b", "time"))
        defVar(dsout, "Gmix_col_$(nm)",  Float32.(G_cols[nm]),     ("b", "time"))
        defVar(dsout, "psi_col_$(nm)",   Float32.(Ψ_cols[nm]),     ("b", "time"))
        defVar(dsout, "Gmix_int_$(nm)",  Float32.(G_int[nm]),      ("b", "interval"))
        defVar(dsout, "psi_int_$(nm)",   Float32.(Ψ_int[nm]),      ("b", "interval"))
        defVar(dsout, "Gsurf_int_$(nm)", Float32.(Gsurf_int[nm]),  ("b", "interval"))
        defVar(dsout, "dMdt_$(nm)",      Float32.(dMdt_cols[nm]),  ("b", "interval"))
        defVar(dsout, "R_$(nm)",         Float32.(R_cols[nm]),     ("b", "interval"))
    end
    dsout.attrib["method"]   = "CODF budget restricted to the final segment (equilibrium period), buoyancy axis actively rebuilt every steps_per_interval steps (pooled over all snapshots in the chunk) and interpolated onto one common per-segment axis (also pooled). See gmix_CODF_equilibrium.jl."
    dsout.attrib["steps_per_interval"] = steps_per_interval
    dsout.attrib["Ra"]       = Ra
    dsout.attrib["segment"]  = final_segment
    dsout.attrib["n_b_bins"] = n_b_bins
    dsout.attrib["bin_mode"] = "blended"
    dsout.attrib["lambda"]   = λ_bins
end
println("saved -> $outfile")

for name in region_names
    @printf("  %-16s  max|G_mix| = %.3e\n", name, maximum(abs.(Gmix_regions[name])))
end
