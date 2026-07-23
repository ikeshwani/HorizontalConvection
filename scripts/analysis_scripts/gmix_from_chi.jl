# GMIX FROM CHI!!!!!!!!!!!!!
# i reran the gmix calculation from chi (the OG METHOD) : gmix = 0.5 ∂²/∂b² ∫\_{b'<b} χ dV
# but without the gaussian smoothing method 
# it is rerun with the quantile method for variable b bins 
# and time averaged over 100s intervals 
# but I didnt recalculate ψ, G_surf, and dMdt because they don't depend on gmix
# so we'll need to get those from the "Gmix_quantile_regions_CODF_3hill_RA1e8_seg1to26.nc" file 

# one more important difference between the chi calculation and the CODF calculation 
# since the chi calculation has a second derivative wrt to buoyancy
# this means that the values land on the interior b edges rather than the centers

#I'm going to need to load both files and figure out how to deal with interpolating

using TopographicHorizontalConvection   # physics: boundary_layer_depth, nearest_xi
using NCDatasets
using CairoMakie
using Statistics
using Printf
using NaNStatistics

#directory containing Ra1e8 hill data
data_dir       = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/"
plot_dir       = joinpath(data_dir, "figures")
mkpath(plot_dir)

#purely quantile
gmix_chi = joinpath(data_dir, "Gmix_chi_quantile_regions_3hill_RA1e8_seg1to28.nc")
gmix_CODF = joinpath(data_dir, "Gmix_quantile_regions_CODF_3hill_RA1e8_seg1to26.nc") 

ds_CHI = NCDataset(gmix_chi)
ds_CODF = NCDataset(gmix_CODF)

# getting base data from buoyancy segment 1
ds1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
Lx  = Float64(ds1.attrib["Lx"])
Ra  = Float64(ds1.attrib["Ra"])
x = ds1["x_caa"][:]
close(ds1)

#calculating boundary layer depth from TopographicHorizontalConvection:boundary_layer_depth
zBL = boundary_layer_depth(Lx, Ra)

#lets just look at the hill one region first
region_key = "hill3"
x_hill1_left = 0.65
x_hill1_right = 1.35

# lets load buoyancy from both files
b_chi = ds_CHI["b"][:]
b_CODF = ds_CODF["b"][:]

#im choosing time interval end-10 because that corresponds to 680.3-690.3 seconds which is the same time interval I was looking at for the CODF file
gmix_chi_hill1 = ds_CHI["Gmix_int_$region_key"][:, end-10]

#okay here's the corresponding gmix for hill 1 from CODF
gmix_CODF_hill1 = ds_CODF["Gmix_int_$region_key"][:, end-1]

#other things we need to load from the CODF file include: psi_b, transport_in, gBF, and dMdt 
gBF_hill1 = ds_CODF["Gsurf_int_$region_key"][:, end-1]
transport_hill1 = ds_CODF["psi_int_$region_key"][:, end-1]
dMdt_hill1 = ds_CODF["dMdt_$region_key"][:, end-1]
R_hill1_CODF = ds_CODF["R_$region_key"][:, end-1]

#okay now lets get psi_in_left and psi_in_right from psi_b of CODF file
tstart = ds_CODF["t_start"][end-1]
tend = ds_CODF["t_end"][end-1]
t = ds_CODF["time"][:]

index_start = nearest_xi(t, tstart)
index_end = nearest_xi(t, tend)

index_in = nearest_xi(x, x_hill1_left)
index_out = nearest_xi(x, x_hill1_right)

ψ_hill1_int = nanmean(ds_CODF["psi_b"][:, :, index_start:index_end], dims=3)[:, :]
ψ_hill1_left  = - ψ_hill1_int[index_in, :]
ψ_hill1_right = ψ_hill1_int[index_out, :]

fig = Figure(size = (600, 600))
ax = Axis(fig[1,1], xlabel="gmix and transport", ylabel="buoyancy", title="comparing gmix calculated with chi and CODF for $region_key from $tstart to $tend")
lines!(-gmix_chi_hill1, b_chi, label="gmix chi", color="purple")
lines!(gmix_CODF_hill1, b_CODF, label="gmix CODF", color="orange", linestyle=:dash)
Legend(fig[1,2], ax)
fig