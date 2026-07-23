using TopographicHorizontalConvection   # region masks: gmix_region_masks, precompute_regions
using NCDatasets
using CairoMakie
using NaNStatistics
using Printf

#lets look at one dataset of the Ra1e8 hilly experiment 

data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/"
seg = 10
ds = NCDataset(joinpath(data_dir, "buoyancy_seg$(seg).nc"))
ds1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))

# load in global attribs from seg 1

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
b_range   = (-1.0, 1.0)
n_b_bins  = 101
b_edges   = collect(range(b_range[1], b_range[2], length=n_b_bins))
b_centers = 0.5 .* (b_edges[1:end-1] .+ b_edges[2:end])

# region geometry — SAME masks the other G_mix methods use (src/analysis/regions.jl).
# "boundary_layer" = surface layer above zBL (where the forcing / b≈1 spike lives);
# the plume + basin/hill columns below zBL are the interior. These partition the
# whole wet domain, so summing all regions recovers the whole-domain curve.
ΔA_2d          = dropdims(Δx_center .* Δy_center, dims=3)   # [Nx, Ny] cell footprint
region_masks   = gmix_region_masks(x, z, Lx, Ra)
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
    for n in 1:n_b_bins
        M[n] = sum(@view cg[bg .< b_edges[n]])
    end
    return -diff(M) ./ diff(b_edges) #figure out where i have a sign error, negative sign thrown in for now
end

# ---- loop over every time in the segment, every region ---------------------
times = ds["time"][:]
Nt    = length(times)
Gmix_regions = Dict(name => zeros(n_b_bins - 1, Nt) for name in region_names)

for ti in 1:Nt
    b = Array{Float64}(ds["b"][:, :, :, ti])
    b[.!wet] .= NaN
    CONV_dV = conv_dV_snapshot(b)
    for r in region_precomp
        Gmix_regions[r.name][:, ti] .= gmix_region(b, CONV_dV, r.idxs)
    end
    @printf("ti=%2d  t=%.2f\n", ti, times[ti])
end
close(ds)

# time-mean per region over the segment
Gmix_region_mean = Dict(name => vec(nanmean(Gmix_regions[name], dims=2))
                        for name in region_names)

# ---- plot: one G_mix(b, t) heatmap per region ------------------------------
clim = 0.008
fig  = Figure(size=(300 * length(region_names), 420))
for (i, name) in enumerate(region_names)
    ax = Axis(fig[1, 2i-1], xlabel="time", ylabel="b", title=name)
    hm = heatmap!(ax, times, b_centers, permutedims(Gmix_regions[name]),
                  colormap=:balance, colorrange=(-clim, clim))
    Colorbar(fig[1, 2i], hm)
end

outpng = joinpath(data_dir, "figures", "gmix_CODF_regions_seg$(seg).png")
mkpath(dirname(outpng))
save(outpng, fig)
println("saved figure → $outpng")

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

outpng2 = joinpath(data_dir, "figures", "gmix_CODF_regions_tmean_seg$(seg).png")
save(outpng2, fig2)
println("saved figure → $outpng2")


#i want to compare this method to gmix from version two
ds_gmix = NCDataset(joinpath(data_dir, "Gmix_regions_v2_3hill_RA1e8_seg1to23.nc"))
time_int = ds_gmix["time"][1792:1948] #same time interval as gmix codf
b_v2     = ds_gmix["b"][:]            # v2 buoyancy axis (499 pts; ≠ b_centers)

Gmix_region_mean_v2 = Dict(name => vec(nanmean(ds_gmix["Gmix_$name"][:, 1792:1948], dims=2))
                        for name in region_names)



# ---- plot: time-mean G_mix(b) line per region ------------------------------
# each region gets its own x-scale (magnitudes differ ~100×); b-axis is shared.
# CODF (this method, green) vs v2 sort (red dashed) on their own b-axes.
fig3  = Figure(size=(230 * length(region_names), 520))
axes3 = Axis[]
local lc, l2
for (i, name) in enumerate(region_names)
    ax = Axis(fig3[1, i], xlabel="⟨G_mix⟩", ylabel="b", title=name)
    push!(axes3, ax)
    i > 1 && hideydecorations!(ax; ticks=false, grid=false)
    global lc = lines!(ax, Gmix_region_mean[name],    b_centers, color=:seagreen, linewidth=2)
    global l2 = lines!(ax, Gmix_region_mean_v2[name], b_v2,      color=:red, linestyle=:dash, linewidth=2)
    vlines!(ax, 0.0, color=:gray, linestyle=:dash, linewidth=0.8)
end
linkyaxes!(axes3...)
Legend(fig3[1, length(region_names)+1], [lc, l2], ["CODF (flux-conv)", "v2 sort"])

fig3

