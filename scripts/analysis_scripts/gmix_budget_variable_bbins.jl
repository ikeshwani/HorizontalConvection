# RERUN WITH THE VARIABLE BUOY BINS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! lets see what the vol budget and PDF looks like
# this was a test rerun for the Ra1e8 three hill experiment from segment 22-23
# what we did was combine buoyancy ruler and volume ruler to get a blended bins
# φ(b) = (1 - λ) (b - bmin)/(bmax - bmin) + λ F(b)
#
# Two cases compared side by side, each on ITS OWN b axis (the two files place
# their bin centers at different buoyancies — never compare them elementwise):
#   λ = 0.7  blended axis
#   λ = 1.0  pure quantile (equal-volume bins)

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
gmix_file_q     = joinpath(data_dir, "Gmix_quantile_regions_CODF_3hill_RA1e8_seg1to26.nc")

#purely quantile but control experiment
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



#this function takes in the dataset and the region of interest and creates a plot for the second to last interval
function load_interval_case(ds_g, regiontitle, region_key, region_in, region_out)
    b = Float64.(ds_g["b"][:])

    gmix = ds_g["Gmix_int_$region_key"][:, end-1]
    gBF = ds_g["Gsurf_int_$region_key"][:, end-1]
    transport = ds_g["psi_int_$region_key"][:, end-1]
    dMdt = ds_g["dMdt_$region_key"][:, end-1]
    R = ds_g["R_$region_key"][:, end-1]

    time = ds_g["time"][:]
    tstart = ds_g["t_start"][end-1]
    tend = ds_g["t_end"][end-1]

    index_tstart = nearest_xi(time, tstart)
    index_tend = nearest_xi(time, tend)

    index_in = nearest_xi(x, region_in)
    index_out = nearest_xi(x, region_out)

    ψ = nanmean(ds_g["psi_b"][:, :, index_tstart:index_tend], dims=3)[:, :]
    ψ_in_left  = - ψ[index_in, :]
    ψ_in_right = ψ[index_out, :]

    fig = Figure(size=(600, 600))
    ax = Axis(fig[1,1], xlabel = "transport and gmix", ylabel = "buoyancy", title = "$regiontitle volume budget for hill experiment from $tstart to $tend")
    lines!(gmix .+ gBF, b, label=L"\mathcal{G}_{mix} + \mathcal{G}^{bf}", color=:green, linestyle=:dot)
    lines!(transport, b, color=:purple, linestyle=:dot, label=L"\text{transport in}")
    lines!(dMdt, b, color=:pink, label=L"\frac{dM}{dt}")
    lines!(R, b, color=:navyblue, label = L"\text{residual}", linestyle=:dash)
    lines!(ψ_in_right, b, color=:red, label = L"\psi_{in, left}")
    lines!(ψ_in_left, b, color=:dodgerblue, label = L"\psi_{in, right}")
    Legend(fig[1,2], ax)

    save(joinpath(plot_dir_control, "gmix_budget_$(region_key)_intervalmean.png"), fig)

    return fig, (; b, gmix, gBF, transport, dMdt, R, ψ_in_left, ψ_in_right)
end

#load gmix data from gmix CODF calculation files

ds_quantile = NCDataset(gmix_file_q)
ds_quantile_flat = NCDataset(gmix_control)
    

case_int_quantile_hills = load_interval_case(ds_quantile, regions[6][1], regions[6][2], regions[6][3], regions[6][4])


case_int_quantile_control = load_interval_case(ds_quantile_flat, regions[6][1], regions[6][2], regions[6][3], regions[6][4])

tstart = ds_quantile["t_start"][end-1]
tend = ds_quantile["t_end"][end-1]
t = ds_quantile["time"][:]
b = ds_quantile["b"][:]

index_tstart = nearest_xi(t, tstart)
index_tend = nearest_xi(t, tend)


#Gmix hovmollers 
col_names = ["basin0","hill1","basin1","hill2","basin2","hill3","basin3"]

Gmix_interior_sum = sum(ds_quantile["Gmix_$nm"][:, :] for nm in col_names) # this is the sum of all the interior regions
Gmix_bl_sum = sum(ds_quantile["Gmix_bl_$nm"][:, :] for nm in col_names) # this is the sum of all the boundary layers that are split into respective interior regions
Gmix_plume = ds_quantile["Gmix_plume"][:, :]
GBF = sum(ds_quantile["Gsurf_$nm"][:,:] for nm in col_names) #this is the sum of all the G_BF's of each interior region

G_all = Gmix_interior_sum .+ Gmix_bl_sum .+ Gmix_plume .+ GBF # this is Gmix plus GBF
maximum(G_all)
minimum(G_all)

heatmap(t, b, G_all', colormap=:balance, colorrange = (-0.006, 0.006))



#smoothed interval gmix heatmap --- but it doesnt have plume gmix it DOES have boundary layer though
t_start = ds_quantile["t_start"][:]
Gmix_total_int = sum(ds_quantile["Gmix_int_$nm"][:, :] for nm in col_names)
b[95:100]
maximum(Gmix_total_int'[:, 1:95])
minimum(Gmix_total_int'[:, 1:95])

heatmap(t_start, b[1:95], Gmix_total_int'[:, 1:95], colormap=:balance, colorrange=(-0.006, 0.006))

ds_quantile["Gmix_bl_basin0"]