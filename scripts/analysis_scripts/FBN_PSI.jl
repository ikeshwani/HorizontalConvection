using TopographicHorizontalConvection   # region masks + nearest_xi + boundary_layer_depth
using NCDatasets
using CairoMakie
using NaNStatistics
using Statistics
using Printf

#using some of the method from fbn_psi_comparison.jl

# ---- config ----
Ra_tag, Ra_str   = "ra1e8", "RA1e8"
segments         = 1:32                 # must match the CODF file being read
#b1_quantile      = 0.075                # 5-10% volume quantile of downstream basin (B&N Q proxy)

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


# F_BN(b) at bin EDGES: for each edge, -1 times the sum of CONV_dV (the
# volume-integrated convergence of the diffusive buoyancy flux) over cells in
# the region colder than that edge.
function F_BN_region(b, CONV_dV, idxs, b_edges)
    bg = vec(b)[idxs]
    cg = vec(CONV_dV)[idxs]
    return [-sum(@view cg[bg .< be]) for be in b_edges]
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

#CALCULATE F_BN FOR THE LAST SEGMENT!!!!!!!!!! segment = 32

hills        = 1:3
hill_regions = vcat(["hill$n" for n in hills], ["bl_hill$n" for n in hills])

F_BN_sum  = Dict(name => zeros(Float64, n_b_bins) for name in region_names)

ds_last = NCDataset(joinpath(data_dir, "buoyancy_seg32.nc"))

t_start_index = nearest_xi(t0_eq, ds_last["time"][:])
t_end_index = nearest_xi(t1_eq, ds_last["time"][:])

b = ds_last["b"][:, :, :, t_start_index:t_end_index]
t = ds_last["time"][t_start_index:t_end_index]
nt = length(t)
close(ds_last)

# one column per snapshot, per region -- then nanmean over the time dimension,
# same pattern as gmix_CODF.jl's nanmean(Gmix_$(name)[:, time_idx_eq], dims=2)
F_BN_ti = Dict(name => zeros(Float64, n_b_bins, nt) for name in region_names)

for ti in 1:nt
    bi = b[:, :, :, ti]
    bi[.!wet] .= NaN
    conv_dV = conv_dV_snapshot(bi)
    for r in region_precomp
        F_BN_ti[r.name][:, ti] .= F_BN_region(bi, conv_dV, r.idxs, b_edges)
    end
end

F_BN = Dict(name => nanmean(F_BN_ti[name], dims=2)[:] for name in region_names)

# full-depth hill columns = interior + its boundary-layer strip
F_BN_col = Dict(n => F_BN["hill$n"] .+ F_BN["bl_hill$n"] for n in hills)

