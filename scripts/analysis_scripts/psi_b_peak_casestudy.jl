# psi_b_peak_casestudy.jl
#
# Case study of where the four region curves in `psi_b_flux_withGmix_hill.png`
# peak in buoyancy.  For each region (Hill 1 … Basin 3) and each curve
#   ψ_in  = ψ at the region's left  x-boundary
#   ψ_out = ψ at the region's right x-boundary
#   Δψ    = ψ_out − ψ_in
#   G_mix = regional diapycnal mixing / water-mass transformation
# we find the SIGNED GLOBAL EXTREMA (largest-magnitude max and min) over the
# physically realized buoyancy band, then map each peak's b/b★ to a physical
# depth z via the y-averaged buoyancy field and tag the nearest landmark.
#
# Reuses the data-loading / averaging / region structure of
# scripts/analysis_scripts/psi_b_region_flux.jl so the curves are identical to
# the figure.  Reads only existing analysis NetCDFs — no recomputation.
#
# Run from scripts/ :   julia --project=../ analysis_scripts/psi_b_peak_casestudy.jl

using TopographicHorizontalConvection   # boundary_layer_depth, nearest_xi, seafloor_profile
using NCDatasets
using Statistics
using Printf
using DelimitedFiles

# =========================================================
# PARAMETERS  (match psi_b_region_flux.jl)
# =========================================================
avg_window = 10.0

data_dir       = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/"
psib_file      = joinpath(data_dir, "psi_b_t385.nc")
gmix_file      = joinpath(data_dir, "Gmix_regions_v2_RA1e8_seg1to14.nc")
ctrl_psib_file = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_flat_baseforcing_zerostart/psi_b_Control_RA1e8_seg1to12.nc"
buoy_segments  = 1:23                                              # scanned for the b→z averaging window
plot_dir       = joinpath(data_dir, "figures")
out_csv        = joinpath(plot_dir, "psi_b_peaks_casestudy.csv")
mkpath(plot_dir)

# =========================================================
# grid metadata
# =========================================================
ds1  = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
Lx   = Float64(ds1.attrib["Lx"])
H    = Float64(ds1.attrib["H"])
Ra   = Float64(ds1.attrib["Ra"])
bstar = Float64(ds1.attrib["b★"])
h₀    = Float64(ds1.attrib["h₀"])          # absolute tallest-hill height
close(ds1)
h₀_frac = h₀ / H
zBL = boundary_layer_depth(Lx, Ra)
@printf("Lx=%.2f  H=%.2f  Ra=%.0e  b★=%.2f  h₀=%.2f  zBL=%.3f\n", Lx, H, Ra, bstar, h₀, zBL)

# =========================================================
# shared averaging window anchored to control's last time
# =========================================================
ds_ctrl_ref = NCDataset(ctrl_psib_file)
t_end = Float64(ds_ctrl_ref["time"][end])
close(ds_ctrl_ref)
@printf("averaging window: t = %.1f → %.1f\n", t_end - avg_window, t_end)

# =========================================================
# load ψ(x, b, t) — hill, average over window
# =========================================================
println("loading ψ(x, b) ...")
ds_psi = NCDataset(psib_file)
x_psi  = Float64.(ds_psi["x"][:])
t_psi  = Float64.(ds_psi["time"][:])
i_avg  = searchsortedfirst(t_psi, t_end - avg_window):length(t_psi)
ψ_mean = dropdims(mean(Float64.(ds_psi["ψ_b"][:, :, i_avg]), dims=3), dims=3)  # [Nx, n_b]
close(ds_psi)
@printf("  ψ averaged over %d steps (t = %.1f → %.1f)\n", length(i_avg), t_psi[i_avg[1]], t_end)

# =========================================================
# load G_mix per region — same window
# =========================================================
println("loading G_mix by region ...")
ds_g = NCDataset(gmix_file)
t_g  = Float64.(ds_g["time"][:])
b_g  = Float64.(ds_g["b"][:])    # 499 inner bins, matches b_psi[2:end-1]
i_g  = searchsortedfirst(t_g, t_end - avg_window):length(t_g)
region_keys = ["Gmix_hill1", "Gmix_basin1", "Gmix_hill2",
               "Gmix_basin2", "Gmix_hill3", "Gmix_basin3"]
gmix_means = Dict(rk => dropdims(mean(Float64.(ds_g[rk][:, i_g]), dims=2), dims=2)
                  for rk in region_keys)
close(ds_g)
@printf("  G_mix averaged over %d steps\n", length(i_g))

# align ψ to the 499-bin b_g grid
ψ_inner = ψ_mean[:, 2:end-1]     # [Nx, 499]

# =========================================================
# regions (same x-bounds as psi_b_region_flux.jl)
# =========================================================
regions = [
    ("Hill 1",  "Gmix_hill1",  -1.35, -0.65),
    ("Basin 1", "Gmix_basin1", -0.65, -0.35),
    ("Hill 2",  "Gmix_hill2",  -0.35,  0.35),
    ("Basin 2", "Gmix_basin2",  0.35,  0.65),
    ("Hill 3",  "Gmix_hill3",   0.65,  1.35),
    ("Basin 3", "Gmix_basin3",  1.35,  x_psi[end]),
]

# =========================================================
# b → z mapping: y-averaged buoyancy, time-averaged over the SAME window
# (t_end-avg_window → t_end) used for ψ and G_mix.  Built from the full
# buoyancy_seg*.nc files so all three axes share one averaging window instead
# of labeling window-averaged curves with a single-time zonal snapshot.
# =========================================================
@printf("building b → z mapping from full buoyancy over t = %.1f → %.1f ...\n",
        t_end - avg_window, t_end)

# x/z grid from seg1
dsb1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
xz   = Float64.(dsb1["x_caa"][:])
zz   = Float64.(dsb1["z_aac"][:])
close(dsb1)

# accumulate the y-average of every snapshot whose time falls in the window,
# skipping duplicated steps at segment boundaries (t_last guard)
bz_sum = zeros(Float64, length(xz), length(zz))
n_win  = 0
let t_last = -Inf
    for s in buoy_segments
        fpath = joinpath(data_dir, "buoyancy_seg$(s).nc")
        isfile(fpath) || continue
        dsb = NCDataset(fpath)
        tb  = Float64.(dsb["time"][:])
        for (k, tk) in enumerate(tb)
            tk <= t_last && continue                       # segment overlap
            t_last = tk
            (tk >= t_end - avg_window && tk <= t_end) || continue
            b4 = Float64.(dsb["b"][:, :, :, k])            # [Nx, Ny, Nz]
            bz_sum .+= dropdims(mean(b4, dims=2), dims=2)  # y-average → [Nx, Nz]
            n_win += 1
        end
        close(dsb)
    end
end
n_win == 0 && error("no buoyancy timesteps found in window t = $(t_end - avg_window) → $t_end")
bz = bz_sum ./ n_win                                       # [Nx, Nz] window- & y-averaged
@printf("  averaged %d y-averaged buoyancy snapshots over the window\n", n_win)

# Region-mean wet buoyancy profile b̄(z), then invert to z(b) by linear interp.
# A cell is wet where z > seafloor_profile(x).
function region_profile(xL, xR)
    cols = findall(x -> (x >= xL) && (x <= xR), xz)
    sf   = seafloor_profile(xz[cols], H, Lx, h₀_frac, 3)   # floor elevation per x
    prof = fill(NaN, length(zz))
    for (k, z) in enumerate(zz)
        wet = findall(c -> z > sf[c], 1:length(cols))       # wet x-columns at this z
        isempty(wet) && continue
        prof[k] = mean(bz[cols[wet], k])
    end
    return prof
end

# z for a target buoyancy from a (possibly non-monotone) profile: pick the
# shallowest crossing of b̄(z) = b_target (profile increases toward surface).
function depth_of_b(prof, b_target)
    zs = Float64[]
    for k in 1:length(zz)-1
        b1, b2 = prof[k], prof[k+1]
        (isnan(b1) || isnan(b2)) && continue
        if (b1 - b_target) * (b2 - b_target) <= 0 && b1 != b2
            f = (b_target - b1) / (b2 - b1)
            push!(zs, zz[k] + f * (zz[k+1] - zz[k]))
        end
    end
    isempty(zs) && return NaN
    return maximum(zs)   # shallowest (closest to surface)
end

# nearest physical landmark for context
hill_crest = Dict("Hill 1" => -H + h₀, "Hill 2" => -H + 0.75h₀, "Hill 3" => -H + 0.5h₀)
function landmark(region, z)
    isnan(z) && return "—"
    cands = [("surface", 0.0), ("zBL", zBL), ("floor", -H)]
    haskey(hill_crest, region) && push!(cands, ("crest", hill_crest[region]))
    name, _ = cands[argmin([abs(z - zc) for (_, zc) in cands])]
    return name
end

# =========================================================
# find signed global extrema per (region, curve), build table
# =========================================================
curve_names = ["ψ_in", "ψ_out", "Δψ", "G_mix"]
rows = Vector{Any}[]
push!(rows, ["region", "curve", "extremum", "b_over_bstar", "value", "z_depth", "landmark"])

println("\n", "="^96)
@printf("%-9s %-7s %-5s %12s %14s %10s %9s\n",
        "region", "curve", "ext", "b/b★", "value", "z", "landmark")
println("-"^96)

for (label, rk, xL, xR) in regions
    iL = nearest_xi(x_psi, xL);  iR = nearest_xi(x_psi, xR)
    ψ_in  = ψ_inner[iL, :]
    ψ_out = ψ_inner[iR, :]
    Δψ    = ψ_out .- ψ_in
    g     = gmix_means[rk]

    prof  = region_profile(xL, xR)
    b_min = minimum(filter(!isnan, prof))          # densest water realized in region
    valid = findall(b -> (b >= b_min) && (b <= bstar), b_g)

    for (cname, curve) in zip(curve_names, (ψ_in, ψ_out, Δψ, g))
        cv = curve[valid];  bb = b_g[valid]
        peak_abs = maximum(abs.(cv))                       # for prominence filter
        for (etype, idx) in (("max", argmax(cv)), ("min", argmin(cv)))
            val = cv[idx]
            # skip signed extrema that are just roundoff: a curve that is
            # effectively one-signed (e.g. the overturning ψ) has a meaningless
            # opposite-sign extremum.  Keep only extrema ≥ 5% of the curve's peak.
            abs(val) < 0.05 * peak_abs && continue
            bval = bb[idx]
            z    = depth_of_b(prof, bval)
            lm   = landmark(label, z)
            @printf("%-9s %-7s %-5s %12.4f %14.6e %10.3f %9s\n",
                    label, cname, etype, bval/bstar, val, z, lm)
            push!(rows, [label, cname, etype, bval/bstar, val, z, lm])
        end
    end
    println("-"^96)
end

writedlm(out_csv, rows, ',')
println("\nsaved → ", out_csv)
