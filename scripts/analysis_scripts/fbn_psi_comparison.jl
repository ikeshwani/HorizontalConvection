using TopographicHorizontalConvection   # region masks + nearest_xi + boundary_layer_depth
using NCDatasets
using CairoMakie
using NaNStatistics
using Statistics
using Printf

# ==============================================================================
# Bryden & Nurser (2003) bulk-formula test.
#
# B&N estimate the turbulent transformation at a topographic sill from a bulk
# formula Q·Δρ.  Here we build the direct analog for this simulation and check
# it against the fully resolved transport.
#
# Definitions (already derived — see task notes, not re-derived here):
#   F_BN(b) ≡ ∫_{A(b)} (-κ∇b)·n̂ dS         diffusive buoyancy flux through the b-isosurface
#           = -M(b)                          where M(b) = Σ_{cells: b'<b} CONV_dV  (gmix_region's M)
#   G_mix(b) = ∂F_BN/∂b                      exact, via the divergence theorem + FTC
#   in a quasi-steady interval, away from the surface:  G_mix(b) ≈ -Ψ(b)
#   ⇒ Ψ ≈ -∂F_BN/∂b ≈ -(F_BN(b1) - F_BN(b0)) / (b1 - b0)   (a bulk, F_BN-only estimate of Ψ)
#
# This script computes F_BN(b) for each hill's full column (interior + its bl_
# strip) two independent ways over the equilibrium interval only:
#   (a) freshly, by integrating -κ∇b·n̂ directly from the raw buoyancy field
#       (a sibling of gmix_CODF.jl's gmix_region that returns M instead of -diff(M)),
#   (b) reconstructed from the ALREADY-SAVED Gmix_$(name)(b,t) arrays in the
#       CODF output NetCDF, via a reverse cumulative sum.
# These should agree to numerical precision — printed explicitly below.
#
# Scope note: "freshly computed" is restricted to the equilibrium interval
# (the last budget interval, t = t_start[end]..t_end[end]) rather than the
# full 26-segment record.  Everything downstream (b0, b1, the Ψ comparison)
# only ever uses that interval anyway (it's what budget_figure in gmix_CODF.jl
# reports too), and that interval falls entirely inside one segment file, so
# recomputing it directly here avoids re-running the full heavy CONV_dV pass.
# ==============================================================================

# ---- config ----
Ra_tag, Ra_str   = "ra1e8", "RA1e8"
segments         = 1:26                 # must match the CODF file being read
b1_quantile      = 0.075                # 5-10% volume quantile of downstream basin (B&N Q proxy)

data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/$(Ra_tag)_4xstretch_threehill_baseforcing_zerostart/"
plot_dir = joinpath(data_dir, "figures")
mkpath(plot_dir)

gmix_file = joinpath(data_dir, "Gmix_quantile_regions_CODF_3hill_$(Ra_str)_seg$(first(segments))to$(last(segments)).nc")

# ---- load grid + metadata (mirrors gmix_CODF.jl Pass 0 setup) ----
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

vol = Δx_center .* Δy_center .* Δz_center
ν = sqrt(Pr * b★ * H^3 / Ra)
κ = ν / Pr

wet = ds1["b"][:, :, :, 2] .!= 0
close(ds1)

# ---- region geometry (mirrors gmix_CODF.jl exactly, so region names line up
# with the ones already saved in gmix_file) ----

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

# ---- pull the buoyancy axis + equilibrium interval window from the CODF file ----
ds_g = NCDataset(gmix_file)
b_edges  = Float64.(ds_g["b_edges"][:])
b_centers = Float64.(ds_g["b"][:])
n_b_bins  = length(b_edges)
time_g    = Float64.(ds_g["time"][:])
t_start_g = Float64.(ds_g["t_start"][:])
t_end_g   = Float64.(ds_g["t_end"][:])

t0_eq, t1_eq = t_start_g[end], t_end_g[end]   # last interval == equilibrium period
time_idx_eq  = findall((time_g .>= t0_eq - 1e-9) .& (time_g .<= t1_eq + 1e-9))
@printf("equilibrium interval: t = %.2f -> %.2f  (%d snapshots, global time indices %d:%d)\n",
        t0_eq, t1_eq, length(time_idx_eq), time_idx_eq[1], time_idx_eq[end])

# ---- Step 1a: fresh F_BN(b) over the equilibrium interval, per region ----
# `M_region` is the sibling of gmix_CODF.jl's `gmix_region`: it returns the
# cumulative M(b) at bin EDGES instead of the final -diff(M)/diff(b_edges).
function M_region(b, CONV_dV, idxs, b_edges)
    bg = vec(b)[idxs]
    cg = vec(CONV_dV)[idxs]
    M  = zeros(length(b_edges))
    for n in eachindex(b_edges)
        M[n] = sum(@view cg[bg .< b_edges[n]])
    end
    return M
end

# same per-cell diffusive-convergence formula as gmix_CODF.jl's conv_dV_snapshot
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

# find which segment file(s) cover the equilibrium window
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

seg_hits = segments_covering(data_dir, segments, t0_eq, t1_eq)
@printf("equilibrium interval covered by segment(s): %s\n", [s for (s, _) in seg_hits])

hills        = 1:3
hill_regions = vcat(["hill$n" for n in hills], ["bl_hill$n" for n in hills])

M_fresh_sum   = Dict(name => zeros(Float64, n_b_bins) for name in region_names)
b_min_region  = Dict(name => Inf for name in region_names)
# cache the exact (single snapshot, single y-column) 2D slice whenever a region's
# running minimum improves -- lets us later plot the real instant/location where
# the coldest water actually occurred, instead of a time+y average that can
# smooth a genuinely transient event out of existence (see WMT discussion below).
b_min_slice = Dict(name => (s=0, ti=0, iy=0, b2d=zeros(Nx, Nz)) for name in hill_regions)
n_snap_fresh  = 0

for (s, local_range) in seg_hits
    bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
    b_seg = Array{Float64}(bfile["b"][:, :, :, local_range])
    close(bfile)
    nt = size(b_seg, 4)
    @printf("  seg %d: loading %d snapshot(s) for fresh F_BN...\n", s, nt)
    for ti in 1:nt
        b = b_seg[:, :, :, ti]
        b[.!wet] .= NaN
        CONV_dV = conv_dV_snapshot(b)
        for r in region_precomp
            M_fresh_sum[r.name] .+= M_region(b, CONV_dV, r.idxs, b_edges)
            bg = vec(b)[r.idxs]
            im = argmin(bg)
            m  = bg[im]
            if m < b_min_region[r.name]
                b_min_region[r.name] = m
                if r.name in hill_regions
                    iy = Tuple(CartesianIndices((Nx, Ny, Nz))[r.idxs[im]])[2]
                    b_min_slice[r.name] = (s=s, ti=ti, iy=iy, b2d=copy(b[:, iy, :]))
                end
            end
        end
        global n_snap_fresh += 1
    end
    b_seg = nothing; GC.gc()
end

M_fresh   = Dict(name => M_fresh_sum[name] ./ n_snap_fresh for name in region_names)
F_BN_fresh = Dict(name => -M_fresh[name] for name in region_names)
@printf("fresh F_BN averaged over %d snapshot(s)\n", n_snap_fresh)

# ---- Step 1b: reconstruct F_BN from the already-saved Gmix_$(name)(b,t) ----
# M[n+1] = M[n] - Gmix[n]*(b_edges[n+1]-b_edges[n]), anchored at M[1] = 0.
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
    Gmix_mean = nanmean(Float64.(ds_g["Gmix_$(name)"][:, time_idx_eq]), dims=2)[:]
    M_recon[name] = reconstruct_M(Gmix_mean, b_edges)
end
F_BN_recon = Dict(name => -M_recon[name] for name in region_names)

println("\nfresh vs reconstructed F_BN agreement (per region, max|Δ| over b_edges):")
for name in region_names
    err = maximum(abs.(F_BN_fresh[name] .- F_BN_recon[name]))
    scale = max(maximum(abs.(F_BN_fresh[name])), 1e-30)
    @printf("  %-10s  max|Δ| = %.3e   (%.2f%% of max|F_BN|)\n", name, err, 100*err/scale)
end

# ---- combine interior + bl_ into full-depth hill columns (fresh + reconstructed) ----
# (hills, hill_regions defined earlier, alongside the fresh loop)
F_BN_col_fresh = Dict(n => F_BN_fresh["hill$n"] .+ F_BN_fresh["bl_hill$n"] for n in hills)
F_BN_col_recon = Dict(n => F_BN_recon["hill$n"] .+ F_BN_recon["bl_hill$n"] for n in hills)
b0_hill = Dict(n => min(b_min_region["hill$n"], b_min_region["bl_hill$n"]) for n in hills)

# ---- upstream/downstream basin definition, per hill (Bryden & Nurser's own
# two-reservoir picture).
#   b0 := the exact buoyancy boundary below which the DOWNSTREAM basin has zero
#         volume of its own -- the coldest buoyancy class that basin actually
#         contains. By construction M_downstream(b) = 0 for every b <= b0, which
#         IS the B&N invariant: a density that dense flowing into the hill cannot
#         already be sitting downstream, so any transport there is a genuinely
#         fresh crossing of the sill, not water that was already there.
#   b1 := the b1_quantile volume quantile of that SAME downstream basin -- the
#         bulk density that basin has actually filled up with after crossing.
# (An earlier version picked b0 as a volume quantile of the UPSTREAM basin
# instead. That's not equivalent: nothing in "upstream basin's own quantile"
# constrains what's in the downstream basin, and checking it directly showed
# 0.33-0.36% of the downstream basin's volume leaking through at that b0 for
# hill1/hill3 -- a real violation of the B&N picture, not just noise.)
function basin_M_col(nm)
    dropdims(nanmean(Float64.(ds_g["M_$(nm)"][:, time_idx_eq]) .+
                      Float64.(ds_g["M_bl_$(nm)"][:, time_idx_eq]), dims=2), dims=2)
end

function volume_quantile(M_col, b_centers, q)
    V_tot  = M_col[end]
    target = q * V_tot
    ib = clamp(searchsortedfirst(M_col, target), 2, length(M_col))
    frac = (target - M_col[ib-1]) / (M_col[ib] - M_col[ib-1])
    return b_centers[ib-1] + frac * (b_centers[ib] - b_centers[ib-1])
end

# coldest b at which M_col first becomes nonzero: the basin's own true
# zero-support boundary -- no water of the basin is colder than this.
function zero_support(M_col, b_centers)
    i0 = findfirst(>(0), M_col)
    i0 === nothing && return b_centers[end]
    return b_centers[max(i0 - 1, 1)]
end

neighbors = Dict(1 => ("basin0", "basin1"), 2 => ("basin1", "basin2"), 3 => ("basin2", "basin3"))

b0_up   = Dict{Int, Float64}()
b1_down = Dict{Int, Float64}()
nm_up   = Dict{Int, String}()
nm_down = Dict{Int, String}()
M_basin_cache = Dict{String, Vector{Float64}}()
for n in hills
    nm_left, nm_right = neighbors[n]
    haskey(M_basin_cache, nm_left)  || (M_basin_cache[nm_left]  = basin_M_col(nm_left))
    haskey(M_basin_cache, nm_right) || (M_basin_cache[nm_right] = basin_M_col(nm_right))

    # whichever basin's own water reaches colder is upstream (the source
    # reservoir); the other, whose own coldest water is warmer, is downstream
    zs_left  = zero_support(M_basin_cache[nm_left],  b_centers)
    zs_right = zero_support(M_basin_cache[nm_right], b_centers)
    nm_up[n], nm_down[n] = zs_left < zs_right ? (nm_left, nm_right) : (nm_right, nm_left)

    b0_up[n]   = zero_support(M_basin_cache[nm_down[n]], b_centers)
    b1_down[n] = volume_quantile(M_basin_cache[nm_down[n]], b_centers, b1_quantile)

    @printf("hill%d: upstream = %s, downstream = %s  (b0 = %+.4f, b1 = %+.4f)\n",
            n, nm_up[n], nm_down[n], b0_up[n], b1_down[n])
end

# ---- sanity check: does the downstream basin actually hold zero mass at b0_up? ----
# should now read exactly 0.00% for every hill, by construction.
println("\nsanity check: does b0_up exist in the downstream basin? (should be exactly 0%)")
for n in hills
    M_down = M_basin_cache[nm_down[n]]
    ib0 = clamp(searchsortedfirst(b_centers, b0_up[n]), 1, length(b_centers))
    frac_down = M_down[ib0] / M_down[end]
    @printf("  hill%d  M_%s(b0_up=%.4f) = %.3e  (%.2f%% of %s's total volume)\n",
            n, nm_down[n], b0_up[n], M_down[ib0], 100*frac_down, nm_down[n])
end

# ---- equilibrium-mean (x,z) buoyancy field, y-averaged, for the background heatmap ----
(s, local_range) = only(seg_hits)   # equilibrium window sits entirely in one segment
bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
b_seg = Array{Float64}(bfile["b"][:, :, :, local_range])
close(bfile)
b_xz_sum = zeros(Nx, Nz)
n_xz = 0
for ti in 1:size(b_seg, 4)
    b = b_seg[:, :, :, ti]
    b[.!wet] .= NaN
    b_xz_sum .+= dropdims(nanmean(b, dims=2), dims=2)
    global n_xz += 1
end
b_xz = b_xz_sum ./ n_xz

# true topography footprint, from the actual (channel-modulated, y-varying) wet
# mask -- NOT the analytic seafloor_profile(x), which is x-only and therefore
# blind to the y=0 "channel" notch (see Seafloor3D in src/simulation.jl: hill
# height is multiplied by a y-dependent factor that shrinks it near y=0).
# dry_frac(x,z) = fraction of y-columns that are dry at that (x,z): 1 = solid
# rock for every y (truly opaque), 0 = open water for every y, in between =
# partially blocked -- exactly the cells where a y-average can have real data
# even though the nominal, unmodulated hill profile would call it "inside the hill."
dry_frac = dropdims(mean(Float64.(.!wet), dims=2), dims=2)   # [Nx, Nz]

# the old, analytic, x-only profile -- kept only as a dashed reference line to
# show explicitly how it disagrees with the true (dry_frac) footprint.
h₀_frac, numhill = 0.5, 3
seafloor_nominal = seafloor_profile(x, H, Lx, h₀_frac, numhill)

fig = Figure(size=(900, 400))
ax = Axis(fig[1, 1], xlabel="x", ylabel="z",
          title="hill1: upstream (b0) vs downstream (b1) buoyancy contours")
hm = heatmap!(ax, x, z, b_xz, colormap=:balance, nan_color=:transparent)
Colorbar(fig[1, 2], hm, label="b")

# true topography: opaque where always dry, fading to transparent where only
# partially dry across y -- the "haze" is exactly where the channel lets some
# y-columns stay wet even though the nominal profile says solid ground.
heatmap!(ax, x, z, dry_frac,
         colormap=cgrad([RGBAf(0.15, 0.15, 0.15, 0.0), RGBAf(0.15, 0.15, 0.15, 1.0)]),
         colorrange=(0, 1))
lines!(ax, x, seafloor_nominal, color=:black, linestyle=:dash, linewidth=1.5,
       label="nominal profile (x-only, no channel)")

contour!(ax, x, z, b_xz, levels=[b0_up[1]], color=:crimson, linewidth=2.5,
         label="b0 (upstream, $(nm_up[1])) = $(round(b0_up[1], digits=3))")
contour!(ax, x, z, b_xz, levels=[b1_down[1]], color=:dodgerblue, linewidth=2.5,
         label="b1 (downstream, $(nm_down[1])) = $(round(b1_down[1], digits=3))")
axislegend(ax, position=:rb, merge=true)

xlims!(ax, x[1], x[end]); ylims!(ax, -H, 0)
save(joinpath(plot_dir, "hill1_b0_b1_contours.png"), fig)
fig

# ---- y-z cross-section at each hill's peak x: the channel notch, directly ----
# bottom_height(x,y) is the true seafloor elevation the grid actually saved --
# the exact Seafloor3D geometry, no derivation from the wet mask needed.
ds_bh = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
y = Float64.(ds_bh["y_aca"][:])
bottom_height = Float64.(ds_bh["bottom_height"][:, :])   # [Nx, Ny]
close(ds_bh)

hill_peak_x = [-Lx/4, 0.0, Lx/4]
hill_ix = [argmin(abs.(x .- xp)) for xp in hill_peak_x]

fig_yz = Figure(size=(1000, 300))
for (i, n) in enumerate(1:3)
    axyz = Axis(fig_yz[1, i], xlabel="y", ylabel = i == 1 ? "z" : "",
                title="hill$n  (x = $(round(x[hill_ix[i]], digits=2)))")
    band!(axyz, y, fill(-H, length(y)), bottom_height[hill_ix[i], :], color=(:gray30, 1.0))
    xlims!(axyz, y[1], y[end]); ylims!(axyz, -H, 0)
end
Label(fig_yz[0, :], "seafloor elevation in the (y,z) plane at each hill's peak x — the channel notch", fontsize=14)
save(joinpath(plot_dir, "hill_channel_yz.png"), fig_yz)
fig_yz

# ---- instantaneous, single-(y,t) view of hill1 ----
# Every WMT quantity here (M, F_BN, G_mix) is computed per-snapshot on the raw
# 3D field and only time-averaged AFTER the fact -- never on a time- or
# y-averaged buoyancy field. A time+y mean background can visually erase a real
# but transient event: the very first version of this plot showed the old,
# true-minimum b0 contour landing near the plume, nowhere close to hill1 --
# not because it was wrong, but because the single snapshot/single y-column
# that set that minimum got smoothed away by the average. Here we instead plot
# the exact (x,z) slice, at the exact snapshot and y-column, where hill1's own
# true minimum buoyancy actually occurred -- a real instant, not an average of one.
b0_hill_slice = Dict{Int, NamedTuple}()
for n in hills
    nm = b_min_region["hill$n"] <= b_min_region["bl_hill$n"] ? "hill$n" : "bl_hill$n"
    b0_hill_slice[n] = b_min_slice[nm]
end

ex = b0_hill_slice[1]
@printf("hill1's true minimum occurred at segment %d, snapshot %d, y-index %d (y=%.4f)\n",
        ex.s, ex.ti, ex.iy, y[ex.iy])

seafloor_exact = bottom_height[:, ex.iy]   # exact seafloor at this one y -- no averaging ambiguity

fig_inst = Figure(size=(900, 400))
axi = Axis(fig_inst[1, 1], xlabel="x", ylabel="z",
           title="hill1: instantaneous snapshot (seg $(ex.s), step $(ex.ti), y=$(round(y[ex.iy], digits=3)))")
hmi = heatmap!(axi, x, z, ex.b2d, colormap=:balance, nan_color=:transparent)
Colorbar(fig_inst[1, 2], hmi, label="b")
band!(axi, x, fill(-H, Nx), seafloor_exact, color=(:gray30, 1.0))

contour!(axi, x, z, ex.b2d, levels=[b0_hill[1]], color=:black, linewidth=2, linestyle=:dot,
         label="old b0 (true min, this instant) = $(round(b0_hill[1], digits=3))")
contour!(axi, x, z, ex.b2d, levels=[b0_up[1]], color=:crimson, linewidth=2.5,
         label="b0 (upstream, $(nm_up[1])) = $(round(b0_up[1], digits=3))")
contour!(axi, x, z, ex.b2d, levels=[b1_down[1]], color=:dodgerblue, linewidth=2.5,
         label="b1 (downstream, $(nm_down[1])) = $(round(b1_down[1], digits=3))")
axislegend(axi, position=:rb, merge=true)

xlims!(axi, x[1], x[end]); ylims!(axi, -H, 0)
save(joinpath(plot_dir, "hill1_instantaneous_snapshot.png"), fig_inst)
fig_inst

# ---- same instantaneous view, generalized to all 3 hills: the 6 chosen
# contours (b0_up/b1_down per hill) side by side, each on its OWN hill's real,
# non-averaged snapshot (own segment/step/y-column, from b_min_slice) -- not a
# shared/common instant, since each hill's true minimum occurs at its own time
# and y. Real bottom_height topography at that exact y, same as the hill1 panel.
fig_inst_all = Figure(size=(1500, 420))
hm_panels = Vector{Any}(undef, length(hills))
for (i, n) in enumerate(hills)
    ex_n = b0_hill_slice[n]
    seafloor_n = bottom_height[:, ex_n.iy]

    axn = Axis(fig_inst_all[1, i], xlabel="x", ylabel=(i == 1 ? "z" : ""),
               title="hill$n (seg $(ex_n.s), step $(ex_n.ti), y=$(round(y[ex_n.iy], digits=3)))")
    hm_panels[i] = heatmap!(axn, x, z, ex_n.b2d, colormap=:balance, nan_color=:transparent, colorrange=(-0.5, 0.5))
    band!(axn, x, fill(-H, Nx), seafloor_n, color=(:gray30, 1.0))

    contour!(axn, x, z, ex_n.b2d, levels=[b0_up[n]], color=:crimson, linewidth=2.5,
             label="b0 (upstream, $(nm_up[n])) = $(round(b0_up[n], digits=3))")
    contour!(axn, x, z, ex_n.b2d, levels=[b1_down[n]], color=:dodgerblue, linewidth=2.5,
             label="b1 (downstream, $(nm_down[n])) = $(round(b1_down[n], digits=3))")
    Legend(fig_inst_all[2, i], axn, labelsize=9)
    xlims!(axn, x[1], x[end]); ylims!(axn, -H, 0)
end
Colorbar(fig_inst_all[1, 4], hm_panels[end], label="b")
Label(fig_inst_all[0, :], "b0 (upstream) / b1 (downstream) contours, at each hill's own true-minimum instant", fontsize=14)
save(joinpath(plot_dir, "hill_all_instantaneous_snapshots.png"), fig_inst_all)
fig_inst_all

# ---- Ψ(b): bulk (F_BN-derived) vs measured, as full curves rather than one
# bulk (b0,b1)-secant number. Ψ_bulk(b) = -ΔF_BN/Δb taken at EVERY adjacent
# bin pair is exactly G_mix(b) = ∂F_BN/∂b discretized, i.e. this tests the
# quasi-steady relation Ψ(b) ≈ -G_mix(b) pointwise, not just its two-point
# secant approximation between b0 and b1.
Ψ_bulk_hill1     = -diff(F_BN_col_fresh[1]) ./ diff(b_edges)   # on b_centers
Ψ_measured_hill1 = Float64.(ds_g["psi_int_hill1"][:, end])     # already-saved, direct-from-u, equilibrium interval

# ---- decompose the Ψ_bulk vs Ψ_measured offset -----------------------------
# The per-column budget (Step 5 of gmix_CODF.jl) is the EXACT relation
#   ∂M/∂t = G_mix + Ψ + G_surface + R
# where R is whatever endpoint-difference ∂M/∂t doesn't balance (interval-mean
# G_mix/Ψ/G_surface vs the exact ∂M/∂t over that same interval — R is not
# "leftover physics", it's the numerical residual of that finite-interval
# balance). Ψ_bulk here is just -G_mix_fresh(b) (verified to agree with the
# production Gmix_int_hill1 to ~1e-13), so rearranging:
#   Ψ_measured − Ψ_bulk = Ψ − (−G_mix) = ∂M/∂t − G_surface − R
# G_surface ≈ 0 this deep below the surface (hill1 is entirely sub-boundary-
# layer), so the offset should be explained almost entirely by ∂M/∂t (i.e. by
# how far the equilibrium interval is from truly steady, ∂M/∂t=0) plus R.
# Pulling both saved terms lets us see which one actually accounts for the gap.
dMdt_hill1 = Float64.(ds_g["dMdt_hill1"][:, end])   # equilibrium interval, on b_centers
R_hill1    = Float64.(ds_g["R_hill1"][:, end])
offset_hill1 = Ψ_measured_hill1 .- Ψ_bulk_hill1

fig_curve = Figure(size=(700, 700))
axc = Axis(fig_curve[1, 1], xlabel="Ψ (volume flux)", ylabel="b",
           title="hill1: Ψ(b) — bulk (F_BN-derived) vs measured (from velocities)")
lines!(axc, Ψ_bulk_hill1, b_centers, color=:seagreen, linewidth=2.5, label="Ψ_bulk(b) = -∂F_BN/∂b")
lines!(axc, Ψ_measured_hill1, b_centers, color=:royalblue, linewidth=2.5, label="Ψ_measured(b) (direct)")
lines!(axc, offset_hill1, b_centers, color=:black, linewidth=1.5, linestyle=:solid,
       label="offset = Ψ_measured - Ψ_bulk")
lines!(axc, dMdt_hill1, b_centers, color=:darkorange, linewidth=1.5, linestyle=:dash,
       label="∂M/∂t (non-steady-state)")
lines!(axc, R_hill1, b_centers, color=:purple, linewidth=1.5, linestyle=:dash,
       label="R (budget residual)")
hlines!(axc, [b0_up[1]], color=:gray40, linestyle=:dash, linewidth=1.2,
        label="b0 (upstream, $(nm_up[1]), denser) = $(round(b0_up[1], digits=3))")
hlines!(axc, [b1_down[1]], color=:gray40, linestyle=:dashdot, linewidth=1.2,
        label="b1 (downstream, $(nm_down[1]), lighter) = $(round(b1_down[1], digits=3))")
vlines!(axc, 0.0, color=:gray70, linestyle=:dot, linewidth=1)
Legend(fig_curve[1,2], axc, labelsize=10)
ylims!(axc, b0_up[1] - 0.04, b1_down[1] + 0.1)
save(joinpath(plot_dir, "hill1_psi_of_b_comparison.png"), fig_curve)
fig_curve

# quick numeric check over the b0->b1 range actually used by the bulk estimate
let (lo, hi) = minmax(b0_up[1], b1_down[1])
    sel = findall(lo .<= b_centers .<= hi)
    @printf("\nhill1 offset decomposition, averaged over b in [%.3f, %.3f] (%d bins):\n",
            lo, hi, length(sel))
    @printf("  offset (Ψ_meas - Ψ_bulk) = %+.4e\n", nanmean(offset_hill1[sel]))
    @printf("  ∂M/∂t                    = %+.4e\n", nanmean(dMdt_hill1[sel]))
    @printf("  R (residual)             = %+.4e\n", nanmean(R_hill1[sel]))
    @printf("  ∂M/∂t - R                = %+.4e   (should ≈ offset since G_surface≈0 here)\n",
            nanmean(dMdt_hill1[sel]) - nanmean(R_hill1[sel]))
end



# linear interpolation helper on a sorted (possibly non-uniform) axis
function interp_at(xs, ys, x0)
    x0 <= xs[1] && return ys[1]
    x0 >= xs[end] && return ys[end]
    i = searchsortedlast(xs, x0)
    i = clamp(i, 1, length(xs)-1)
    frac = (x0 - xs[i]) / (xs[i+1] - xs[i])
    return ys[i] + frac * (ys[i+1] - ys[i])
end

println("\nb0 (true zero-support buoyancy) per hill, and F_BN(b0) sanity check:")
for n in hills
    fbn0 = interp_at(b_edges, F_BN_col_fresh[n], b0_hill[n])
    scale = maximum(abs.(F_BN_col_fresh[n]))
    @printf("  hill%d  b0 = %+.4f   F_BN(b0) = %+.3e   (%.2f%% of max|F_BN| = %.3e)\n",
            n, b0_hill[n], fbn0, 100*abs(fbn0)/max(scale,1e-30), scale) 
end

# ---- determine downstream basin per hill from the actual flow, not x-position ----
# A single-column ψ(x,b) reading (e.g. at the hill's peak) can land on a dry cell at
# the hill's own dense b0 and read an exact, spurious 0 -- not evidence of "no flow."
# Instead ask a direct, physically unambiguous question: which of the two neighboring
# basins actually CONTAINS water as dense as this hill's own b0?  Only the upstream
# source basin can (mass has to come from somewhere at least that dense); the
# downstream basin, having already been warmed/mixed crossing the sill, cannot.
neighbors = Dict(1 => ("basin0", "basin1"), 2 => ("basin1", "basin2"), 3 => ("basin2", "basin3"))

M_col_basin_mean(nm) = dropdims(nanmean(Float64.(ds_g["M_$(nm)"][:, time_idx_eq]) .+
                                         Float64.(ds_g["M_bl_$(nm)"][:, time_idx_eq]), dims=2), dims=2)

downstream  = Dict{Int, String}()
M_col_basin = Dict{String, Vector{Float64}}()   # cache so the b1 step below can reuse it
println("\nflow-direction check per hill (which neighbor basin holds water as dense as b0):")
for n in hills
    nm_left, nm_right = neighbors[n]
    haskey(M_col_basin, nm_left)  || (M_col_basin[nm_left]  = M_col_basin_mean(nm_left))
    haskey(M_col_basin, nm_right) || (M_col_basin[nm_right] = M_col_basin_mean(nm_right))

    ib0 = argmin(abs.(b_centers .- b0_hill[n]))
    M_left_b0, M_right_b0 = M_col_basin[nm_left][ib0], M_col_basin[nm_right][ib0]

    if M_left_b0 > M_right_b0
        downstream[n] = nm_right
        @printf("  hill%d  b0=%+.4f   M_%s(b0)=%.3e (source)   M_%s(b0)=%.3e  -> downstream = %s\n",
                n, b0_hill[n], nm_left, M_left_b0, nm_right, M_right_b0, nm_right)
    else
        downstream[n] = nm_left
        @printf("  hill%d  b0=%+.4f   M_%s(b0)=%.3e   M_%s(b0)=%.3e (source)  -> downstream = %s\n",
                n, b0_hill[n], nm_left, M_left_b0, nm_right, M_right_b0, nm_left)
    end
end

# ---- b1: 5-10% volume quantile of the downstream basin's own M_col(b) ----
b1_hill = Dict{Int, Float64}()
println("\nb1 (downstream-basin $(100*b1_quantile)% volume quantile) per hill:")
for n in hills
    basin = downstream[n]
    M_col = M_col_basin[basin]
    V_tot = M_col[end]
    target = b1_quantile * V_tot
    ib = searchsortedfirst(M_col, target)
    ib = clamp(ib, 2, length(M_col))
    frac = (target - M_col[ib-1]) / (M_col[ib] - M_col[ib-1])
    b1 = b_centers[ib-1] + frac * (b_centers[ib] - b_centers[ib-1])
    b1_hill[n] = b1
    @printf("  hill%d  downstream = %-7s  V_tot = %.4e  target = %.4e  b1 = %+.4f\n",
            n, basin, V_tot, target, b1)
end

# ---- Ψ_est: bulk F_BN-derived estimate, and Ψ_measured: direct from psi_int ----
println("\nΨ_est (bulk, F_BN-derived) vs Ψ_measured (direct, from velocity field):")
results = NamedTuple[]
for n in hills
    b0, b1 = b0_up[n], b1_down[n]
    F0 = interp_at(b_edges, F_BN_col_fresh[n], b0)
    F1 = interp_at(b_edges, F_BN_col_fresh[n], b1)
    Ψ_est = -(F1 - F0) / (b1 - b0)

    psi_int = Float64.(ds_g["psi_int_hill$n"][:, end])   # last interval == equilibrium period
    lo, hi = minmax(b0, b1)
    b_sel = findall(lo .<= b_centers .<= hi)
    Ψ_measured = nanmean(psi_int[b_sel])

    pct_diff = 100 * (Ψ_est - Ψ_measured) / abs(Ψ_measured)
    sign_match = sign(Ψ_est) == sign(Ψ_measured)

    push!(results, (hill=n, b0=b0, b1=b1, downstream=nm_down[n],
                     Ψ_est=Ψ_est, Ψ_measured=Ψ_measured, pct_diff=pct_diff, sign_match=sign_match))
    @printf("  hill%d  Ψ_est = %+.4e   Ψ_measured = %+.4e   %%diff = %+.1f%%   sign match: %s  (%d b-bins averaged)\n",
            n, Ψ_est, Ψ_measured, pct_diff, sign_match, length(b_sel))
end

close(ds_g)

# ---- plot: Ψ_est vs Ψ_measured, one pair per hill ----
fig = Figure(size=(700, 480))
ax = Axis(fig[1, 1], xlabel="hill", ylabel="Ψ (volume flux)",
          title="Bryden & Nurser bulk estimate vs resolved transport",
          xticks=(1:3, ["hill1", "hill2", "hill3"]))
barplot!(ax, (1:3) .- 0.15, [r.Ψ_est for r in results], width=0.3, color=:seagreen, label="Ψ_est (F_BN bulk)")
barplot!(ax, (1:3) .+ 0.15, [r.Ψ_measured for r in results], width=0.3, color=:royalblue, label="Ψ_measured (direct)")
hlines!(ax, 0.0, color=:gray, linestyle=:dash, linewidth=0.8)
axislegend(ax, position=:rb)
outpng = joinpath(plot_dir, "fbn_psi_comparison.png")
save(outpng, fig)
println("\nsaved figure -> $outpng")

# ---- summary ----
println("\n=== summary ===")
for r in results
    @printf("hill%d: Ψ_est = %+.3e, Ψ_measured = %+.3e, %%diff = %+.1f%%, sign match = %s, b-range = [%.3f, %.3f], downstream = %s\n",
            r.hill, r.Ψ_est, r.Ψ_measured, r.pct_diff, r.sign_match, min(r.b0,r.b1), max(r.b0,r.b1), r.downstream)
end
