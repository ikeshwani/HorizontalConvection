using TopographicHorizontalConvection   # region masks: gmix_region_masks, precompute_regions
using NCDatasets
using CairoMakie
using NaNStatistics
using Printf

# ---- config ----
experiment = "hill"          # "control" (flat bottom) or "hill" (3-hill GRC)

Ra       = 1e8                  # 1e8 or 1e6
b_range  = (-1, 1)
n_b_bins = 101                  # coarse b-bins: CODF takes one d/db, fewer bins = less noise

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
    segments = Ra == 1e8 ? (1:15) : (1:7)
elseif experiment == "hill"
    topo     = "threehill"
    tag      = "3hill"
    segments = Ra == 1e8 ? (1:23) : (1:10) #ill have to change this as the segments increase 
else
    error("unknown experiment: $experiment (use \"control\" or \"hill\")")
end

data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/$(Ra_tag)_4xstretch_$(topo)_baseforcing_zerostart/"
plot_dir = joinpath(data_dir, "figures")   # figures live inside the run folder
mkpath(plot_dir)

outfile = joinpath(data_dir, "Gmix_regions_CODF_$(tag)_$(Ra_str)_seg$(first(segments))to$(last(segments)).nc")

# load in global attribs from seg 1
ds1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))

Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
Ra, Pr, b★, H = ds1.attrib["Ra"], ds1.attrib["Pr"], ds1.attrib["b★"], ds1.attrib["H"]
Lx = ds1.attrib["Lx"]

x = ds1["x_caa"][:]
y = ds1["y_aca"][:]
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

#create a wet mask from the second time in segment 1
wet = ds1["b"][:, :, :, 2] .!= 0
close(ds1)

# the goal is to recalculate gmix using the convergence of a diffusive flux,
# split BY REGION so boundary-layer mixing is separated from the interior.
# Gmix = ∂/∂b ∫_(V(b'<b)) -∇ ⋅ (-κ ∇b) dV

# buoyancy axis for the b-coordinate integral (time-independent, build once)
b_edges   = collect(range(b_range[1], b_range[2], length=n_b_bins))
b_centers = 0.5 .* (b_edges[1:end-1] .+ b_edges[2:end])

# region geometry.  Step 1: split the single lumped "boundary_layer" into 7
# per-column x-strips (same x-cuts as the interior columns, but above zBL), so a
# full-depth column budget = interior column + its BL strip.  CODF G_mix and
# volume M are additive over this partition.  v2/v3 scripts are unaffected —
# gmix_region_masks in src/ is left untouched; the split is local here.
ΔA_2d = dropdims(Δx_center .* Δy_center, dims=3)   # [Nx, Ny] cell footprint

zBL = boundary_layer_depth(Lx, Ra)
X = reshape(x, :, 1, 1);  Z = reshape(z, 1, 1, :)

# interior column x-bounds (name, xlo, xhi) — mirror gmix_region_masks
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

# ---- per-snapshot per-cell convergence  ∫_cell(-∇·F)dV  [Nx,Ny,Nz] ----------
# b: [Nx,Ny,Nz] Float64 snapshot, already NaN'd in solid cells.
function conv_dV_snapshot(b)

    # X (solid walls at both ends: no-flux)
    flux_x = -κ .* diff(b, dims=1) ./ reshape(Δx_face[2:Nx], Nx-1, 1, 1)
    flux_x_full = zeros(Nx+1, Ny, Nz)
    flux_x_full[2:Nx, :, :] .= flux_x
    flux_x_full[isnan.(flux_x_full)] .= 0.0
    convX = -1 .* diff(flux_x_full, dims=1) ./ Δx_center

    # Y (periodic: face 1 == face Ny+1 == wrap face)
    flux_y = -κ .* diff(b, dims=2) ./ reshape(Δy_face[2:Ny], 1, Ny-1, 1)
    flux_y_wrap = -κ .* (b[:, 1:1, :] .- b[:, Ny:Ny, :]) ./ Δy_face[1]
    flux_y_full = zeros(Nx, Ny+1, Nz)
    flux_y_full[:, 2:Ny, :] .= flux_y
    flux_y_full[:, 1,    :] .= flux_y_wrap[:, 1, :]
    flux_y_full[:, Ny+1, :] .= flux_y_wrap[:, 1, :]
    flux_y_full[isnan.(flux_y_full)] .= 0.0
    convY = -1 .* diff(flux_y_full, dims=2) ./ Δy_center

    # Z (bottom wall + top surface: no-flux)
    flux_z = -κ .* diff(b, dims=3) ./ reshape(Δz_face[2:Nz], 1, 1, Nz-1)
    flux_z_full = zeros(Nx, Ny, Nz+1)
    flux_z_full[:, :, 2:Nz] .= flux_z
    flux_z_full[isnan.(flux_z_full)] .= 0.0
    convZ = -1 .* diff(flux_z_full, dims=3) ./ Δz_center

    CONV_dV = (convX .+ convY .+ convZ) .* vol
    CONV_dV[.!wet] .= 0.0
    return CONV_dV
end

# ---- G_mix(b) over ONE region's cells --------------------------------------
# `idxs` are the region's wet linear indices (from precompute_regions).
# M(b) = ∫_{b'<b} (-∇·F) dV over just those cells, then one d/db at the end.
function gmix_region(b, CONV_dV, idxs)
    bg = vec(b)[idxs]
    cg = vec(CONV_dV)[idxs]
    M  = zeros(n_b_bins)
    for n in eachindex(b_edges)
        M[n] = sum(@view cg[bg .< b_edges[n]])
    end
    return -diff(M) ./ diff(b_edges)
end

# ---- Step 2: cumulative volume M(b) below each b_center over a region --------
# Co-located on b_centers (queried directly), so ∂M/∂t lines up with G_mix.
function volume_below(b, idxs)
    bg = vec(b)[idxs]
    vg = vec(vol)[idxs]
    M  = zeros(length(b_centers))
    for n in eachindex(b_centers)
        M[n] = sum(@view vg[bg .< b_centers[n]])
    end
    return M
end

# ---- Step 4: surface-forcing transformation G_surface(b) per column ----------
# Keep G_mix pure interior mixing (no-flux top stays); add G_surface separately.
# Steady baseforcing ⇒ surface buoyancy is time-independent (seasonal_period=0).
b_surf  = b★ .* tanh.(3 .* (x .+ Lx/3))         # [Nx]
dz_half = 0.5 * Δz_center[1, 1, Nz]             # top-cell center → surface
col_iranges = Dict(nm => findall((x .>= xlo) .& (x .< xhi))
                   for (nm, xlo, xhi) in col_bounds)

# Diffusive surface flux into each top cell (Dirichlet BC: κ(b_surf−b_top)/dz_half),
# as a convergence into the cell (same units as CONV_dV), binned by the top cell's
# buoyancy with one −∂/∂b — so it enters the budget exactly like gmix_region.
function gsurface_col(b_top, irange)
    blk   = b_top[irange, :]                                       # [ncol, Ny]
    Tsurf = κ .* (reshape(b_surf[irange], :, 1) .- blk) ./ dz_half .* ΔA_2d[irange, :]
    bt = vec(blk);  tg = vec(Tsurf)
    M  = zeros(n_b_bins)
    for n in eachindex(b_edges)
        M[n] = sum(@view tg[bt .< b_edges[n]])
    end
    return -diff(M) ./ diff(b_edges)                               # on b_centers
end

# central time derivative along dim 2 (uneven dt handled)
function ddt_time(A, t)
    dA = similar(A, Float64)
    dA[:, 1]   .= (A[:, 2]   .- A[:, 1])     ./ (t[2] - t[1])
    dA[:, end] .= (A[:, end] .- A[:, end-1]) ./ (t[end] - t[end-1])
    for k in 2:length(t)-1
        dA[:, k] .= (A[:, k+1] .- A[:, k-1]) ./ (t[k+1] - t[k-1])
    end
    return dA
end

# ---- Pass 1: build the global time vector, deduplicating segment overlaps ---
println("Pass 1: scanning time vectors from segments $(first(segments))–$(last(segments))...")
time_all = Float64[]
let t_last = -Inf
    for s in segments
        bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
        t_seg = Float64.(bfile["time"][:])
        close(bfile)
        valid = findall(t_seg .> t_last)
        # clip to the velocity file's time length (see Pass 2); a step dropped
        # from a short segment is recovered from the next segment's overlap.
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

# ---- pre-allocate per-region G_mix(b, t), volume M(b, t), and budget fields ----
Gmix_regions = Dict(name => zeros(Float32, n_b_bins - 1, Nt) for name in region_names)
M_regions    = Dict(name => zeros(Float32, length(b_centers), Nt) for name in region_names)
ψ_b          = zeros(Float32, Nx, length(b_centers), Nt)
Gsurf_cols   = Dict(nm => zeros(Float32, length(b_centers), Nt) for (nm, _, _) in col_bounds)

# ---- Pass 2: compute CODF G_mix one segment at a time ----------------------
println("Pass 2: computing CODF G_mix segment by segment...")
let t_last = -Inf, t_offset = 0
for s in segments
    bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
    t_seg = Float64.(bfile["time"][:])
    valid = findall(t_seg .> t_last)

    # a segment can die after writing one more buoyancy snapshot than velocity;
    # clip so t_range never overruns the velocity file's time dimension (we need
    # matching b/u steps for ψ_b anyway).
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
    gi      = t_offset+1 : t_offset+nt          # global time index range
    @printf("  seg %d: loading %d steps (t = %.2f → %.2f)...\n",
            s, nt, t_seg[t_range[1]], t_seg[t_range[end]])

    b_seg = Array(bfile["b"][:, :, :, t_range])
    close(bfile)

    vfile = NCDataset(joinpath(data_dir, "velocities_seg$(s).nc"))
    u_seg = Array(vfile["u"][1:Nx, :, :, t_range])
    close(vfile)

    for (ti, g) in enumerate(gi)
        b = Array{Float64}(b_seg[:, :, :, ti])
        b[.!wet] .= NaN
        CONV_dV = conv_dV_snapshot(b)
        b_top   = b[:, :, Nz]                       # top-row buoyancy (all wet)
        for r in region_precomp
            Gmix_regions[r.name][:, g] .= gmix_region(b, CONV_dV, r.idxs)
            M_regions[r.name][:, g]    .= volume_below(b, r.idxs)
        end
        for (nm, ir) in col_iranges
            Gsurf_cols[nm][:, g] .= gsurface_col(b_top, ir)
        end
    end

    # ψ_b(x, b, t) evaluated DIRECTLY at b_centers (matches G_mix / M / G_surface).
    # Do NOT edge-average: ψ(b) has near-steps where a column is well-mixed (e.g.
    # the homogeneous b≈−0.3 layer at hill edges), and a 0.5·(edge+edge) chord
    # across a step injects a spurious half-height value — the exact cumulative at
    # each center avoids it (≈10× smaller error vs the fine-grid ψ_b reference).
    ψ_seg, _ = get_ψb_sort(b_seg, u_seg, Δy_vec, Δz_vec, Nx, Ny, Nz, nt;
                           b_range=(b_centers[1], b_centers[end]),
                           n_b_bins=length(b_centers))
    ψ_b[:, :, gi] .= ψ_seg

    t_last   = t_seg[t_range[end]]
    t_offset += nt
    b_seg = nothing; u_seg = nothing; GC.gc()
    @printf("  seg %d: done\n", s)
end  # for s
end  # let

# ---- Step 5: per-column full-depth budget + residual -------------------------
# full column = interior (sub-zBL) + its BL strip.  Ψ = convergent transport into
# the column = ψ_in_left + ψ_in_right = ψ_iR − ψ_iL (finalized sign convention,
# see scratch.jl).  Residual R = ∂M/∂t − G_mix − Ψ − G_surface ≈ 0 (quasi-steady).
col_names = [nm for (nm, _, _) in col_bounds]
G_cols    = Dict{String,Matrix{Float64}}()
M_cols    = Dict{String,Matrix{Float64}}()
Ψ_cols    = Dict{String,Matrix{Float64}}()
dMdt_cols = Dict{String,Matrix{Float64}}()
R_cols    = Dict{String,Matrix{Float64}}()

for (nm, xlo, xhi) in col_bounds
    G_col = Float64.(Gmix_regions[nm] .+ Gmix_regions["bl_"*nm])
    M_col = Float64.(M_regions[nm]    .+ M_regions["bl_"*nm])
    iL = nearest_xi(x, xlo)
    iR = nearest_xi(x, min(xhi, x[end]))
    # convergent transport INTO the column (finalized sign, see scratch.jl):
    # Ψ = ψ_in_left + ψ_in_right = −ψ(iL) + ψ(iR) = ψ_iR − ψ_iL
    ψ_in_left = -ψ_b[iL, :, :]
    ψ_in_right = -(-ψ_b[iR, :, :])
    Ψ_col = Float64.(ψ_in_left .+ ψ_in_right)
    dMdt  = ddt_time(M_col, times)
    R_col = dMdt .- G_col .- Ψ_col .- Float64.(Gsurf_cols[nm])

    G_cols[nm] = G_col;  M_cols[nm] = M_col;  Ψ_cols[nm] = Ψ_col
    dMdt_cols[nm] = dMdt;  R_cols[nm] = R_col
end

# time-mean per region over the whole run
Gmix_region_mean = Dict(name => vec(nanmean(Gmix_regions[name], dims=2))
                        for name in region_names)

# ---- save G_mix(b, t) per region to NetCDF ---------------------------------
NCDataset(outfile, "c") do dsout
    defDim(dsout, "b",    length(b_centers))
    defDim(dsout, "time", Nt)
    defDim(dsout, "x",    Nx)
    defVar(dsout, "b",    collect(b_centers), ("b",))
    defVar(dsout, "time", times,             ("time",))
    defVar(dsout, "x",    Float64.(x),        ("x",))
    for name in region_names
        defVar(dsout, "Gmix_$(name)", Gmix_regions[name],        ("b", "time"))
        defVar(dsout, "M_$(name)",    Float32.(M_regions[name]), ("b", "time"))
    end
    defVar(dsout, "psi_b", ψ_b, ("x", "b", "time"))
    for nm in col_names   # full-depth column budget terms
        defVar(dsout, "Gsurf_$(nm)",    Float32.(Gsurf_cols[nm]), ("b", "time"))
        defVar(dsout, "Gmix_col_$(nm)", Float32.(G_cols[nm]),     ("b", "time"))
        defVar(dsout, "psi_col_$(nm)",  Float32.(Ψ_cols[nm]),     ("b", "time"))
        defVar(dsout, "dMdt_$(nm)",     Float32.(dMdt_cols[nm]),  ("b", "time"))
        defVar(dsout, "R_$(nm)",        Float32.(R_cols[nm]),     ("b", "time"))
    end
    dsout.attrib["method"]   = "CODF interior G_mix + surface forcing G_surface; ∂M/∂t = G_mix + Ψ + G_surface"
    dsout.attrib["Ra"]       = Ra
    dsout.attrib["segments"] = "$(first(segments))-$(last(segments))"
end
println("saved → $outfile")

# # ---- plot: one G_mix(b, t) heatmap per region ------------------------------
# clim = 0.008
# fig  = Figure(size=(300 * length(region_names), 420))
# for (i, name) in enumerate(region_names)
#     ax = Axis(fig[1, 2i-1], xlabel="time", ylabel="b", title=name)
#     hm = heatmap!(ax, times, b_centers, permutedims(Gmix_regions[name]),
#                   colormap=:balance, colorrange=(-clim, clim))
#     Colorbar(fig[1, 2i], hm)
# end

# seg_tag = "seg$(first(segments))to$(last(segments))"
# outpng = joinpath(plot_dir, "gmix_CODF_regions_$(seg_tag).png")
# mkpath(dirname(outpng))
# save(outpng, fig)
# println("saved figure → $outpng")

# per-region magnitude check: boundary_layer should dominate
for name in region_names
    @printf("  %-16s  max|G_mix| = %.3e\n", name, maximum(abs.(Gmix_region_mean[name])))
end

fig2  = Figure(size=(230 * length(region_names), 520))
axes2 = Axis[]
for (i, name) in enumerate(region_names)
    ax = Axis(fig2[1, i], xlabel="⟨G_mix⟩", ylabel="b", title=name)
    push!(axes2, ax)
    i > 1 && hideydecorations!(ax; ticks=false, grid=false)
    lines!(ax, Gmix_region_mean[name], b_centers, color=:seagreen, linewidth=2)
    vlines!(ax, 0.0, color=:gray, linestyle=:dash, linewidth=0.8)
end
linkyaxes!(axes2...)

outpng2 = joinpath(plot_dir, "gmix_CODF_regions_tmean.png")
save(outpng2, fig2)
println("saved figure → $outpng2")

# ---- Step 6: per-column volume budget (time-mean over last window) -----------
bwin  = max(1, Nt-10):Nt
tmean(A) = vec(nanmean(A[:, bwin], dims=2))

@printf("\nper-column residual check (max|·| over b, time-mean last %d steps):\n", length(bwin))
for nm in col_names
    @printf("  %-8s  max|R| = %.3e   max|G_mix| = %.3e   max|Ψ| = %.3e   max|Gsurf| = %.3e\n",
            nm, maximum(abs, tmean(R_cols[nm])), maximum(abs, tmean(G_cols[nm])),
            maximum(abs, tmean(Ψ_cols[nm])), maximum(abs, tmean(Gsurf_cols[nm])))
end

function budget_figure(sel_b, fname, suffix)
    fig  = Figure(size=(300 * length(col_names), 560))
    axes = Axis[]
    for (i, nm) in enumerate(col_names)
        ax = Axis(fig[1, i], xlabel="transport", ylabel="b", title="$nm$suffix")
        push!(axes, ax);  i > 1 && hideydecorations!(ax; ticks=false, grid=false)
        lines!(ax, tmean(G_cols[nm])[sel_b],    b_centers[sel_b], color=:seagreen,  linewidth=2, label="G_mix")
        lines!(ax, tmean(Ψ_cols[nm])[sel_b],    b_centers[sel_b], color=:royalblue, linewidth=2, label="Ψ")
        lines!(ax, tmean(Gsurf_cols[nm])[sel_b], b_centers[sel_b], color=:orange,   linewidth=2, label="G_surface")
        lines!(ax, tmean(dMdt_cols[nm])[sel_b], b_centers[sel_b], color=:gray,      linewidth=1.5, linestyle=:dash, label="∂M/∂t")
        lines!(ax, tmean(R_cols[nm])[sel_b],    b_centers[sel_b], color=:black,     linewidth=2, linestyle=:dot,  label="residual")
        vlines!(ax, 0.0, color=:gray, linewidth=0.6)
    end
    linkyaxes!(axes...)
    axislegend(axes[end], position=:rb, labelsize=9)
    outpng = joinpath(plot_dir, fname)
    save(outpng, fig);  println("saved figure → $outpng")
end

budget_figure(eachindex(b_centers), "gmix_CODF_budget_columns.png", "")
dense = findall(b_centers .< -0.3)                  # dense-class (AABW) zoom
!isempty(dense) && budget_figure(dense, "gmix_CODF_budget_columns_dense.png", " (dense)")
