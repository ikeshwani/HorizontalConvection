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

#gmix hilly
gmix_file   = joinpath(data_dir, "Gmix_quantile_regions_CODF_3hill_RA1e8_seg1to32.nc")

#gmix control experiment
data_dir_control = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_flat_baseforcing_zerostart/"
plot_dir_control = joinpath(data_dir_control, "figures")
gmix_control = joinpath(data_dir_control, "Gmix_quantile_regions_CODF_Control_RA1e8_seg1to20.nc")

# getting base data from buoyancy segment 1
ds1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
Lx  = Float64(ds1.attrib["Lx"])
Ra  = Float64(ds1.attrib["Ra"])
x = ds1["x_caa"][:]
close(ds1)

#calculating boundary layer depth from TopographicHorizontalConvection:boundary_layer_depth
zBL = boundary_layer_depth(Lx, Ra)

regions = [
    ("hill one",    "hill1",  -1.35, -0.65),
    ("basin one",   "basin1", -0.65, -0.35),
    ("hill two",    "hill2",  -0.35,  0.35),
    ("basin two",   "basin2",  0.35,  0.65),
    ("hill three",  "hill3",   0.65,  1.35),
    ("basin three", "basin3",  1.35,  x[end]),
]

ds = NCDataset(gmix_file)

#Gmix hovmollers 
col_names = ["basin0","hill1","basin1","hill2","basin2","hill3","basin3"]

Gmix_interior_sum = sum(ds["Gmix_$nm"][:, :] for nm in col_names) # this is the sum of all the interior regions
Gmix_bl_sum = sum(ds["Gmix_bl_$nm"][:, :] for nm in col_names) # this is the sum of all the boundary layers that are split into respective interior regions
Gmix_plume = ds["Gmix_plume"][:, :]
GBF = sum(ds["Gsurf_$nm"][:,:] for nm in col_names) #this is the sum of all the G_BF's of each interior region

G_all = Gmix_interior_sum .+ Gmix_bl_sum .+ Gmix_plume .+ GBF # this is Gmix plus GBF
maximum(G_all)
minimum(G_all)'

t = ds["time"][:]
b_bins = ds["b"][:]

heatmap(t, b_bins, G_all', colormap=:balance, colorrange = (-0.006, 0.006))


#gmix vs time 

G_all_t = nanmean(G_all, dims=1)[:]
lines(t, G_all_t)

#gmix vs buoy @ equilibrium