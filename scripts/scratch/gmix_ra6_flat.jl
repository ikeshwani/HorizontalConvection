# we have yet to do any analysis on gmix for the Ra=1e6 experiments
# I have the FULL equilibrated results and calculations for gmix
# so lets make some volume budget plots, heatmaps, gmix versus time, etc.

using TopographicHorizontalConvection   # physics: boundary_layer_depth, nearest_xi
using NCDatasets
using CairoMakie
using Statistics
using Printf
using NaNStatistics

#directory containing Ra1e8 hill data
data_dir       = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e6_4xstretch_flat_baseforcing_zerostart/"
plot_dir       = joinpath(data_dir, "figures")
mkpath(plot_dir)

gmix_file     = joinpath(data_dir, "Gmix_quantile_regions_CODF_Control_RA1e6_seg1to19.nc")

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
    Ra = ds_g.attrib["Ra"]

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
    ax = Axis(fig[1,1], xlabel = "transport and gmix", ylabel = "buoyancy", title = "$regiontitle volume budget for flat Ra=$Ra experiment from $tstart to $tend")
    lines!(gmix .+ gBF, b, label=L"\mathcal{G}_{mix} + \mathcal{G}^{bf}", color=:green, linestyle=:dot)
    lines!(transport, b, color=:purple, linestyle=:dot, label=L"\text{transport in}")
    lines!(dMdt, b, color=:pink, label=L"\frac{dM}{dt}")
    lines!(R, b, color=:navyblue, label = L"\text{residual}", linestyle=:dash)
    lines!(ψ_in_right, b, color=:red, label = L"\psi_{in, right}")
    lines!(ψ_in_left, b, color=:dodgerblue, label = L"\psi_{in, left}")
    Legend(fig[1,2], ax)

    save(joinpath(plot_dir, "gmix_budget_$(region_key)_intervalmean.png"), fig)

    return fig, (; b, gmix, gBF, transport, dMdt, R, ψ_in_left, ψ_in_right)
end

ds_gmix = NCDataset(gmix_file)

for (i, (title, key, left, right)) in enumerate(regions)
    case_int = load_interval_case(ds_gmix, title, key, left, right)
end

# now we want to make the FULL hovmoller for the Ra6 flat case so ∂M/∂t = G_mix + G_SURF + ψ 
# so we sum Gmix over all the regions, sum Gsurf over all regions, and don't need to sum ψ over all regions because its zero

col_names = ["basin0","hill1","basin1","hill2","basin2","hill3","basin3"]

# these are all the interior hill regions
Gmix_interior_sum = sum(ds_gmix["Gmix_$c"][:, :] for c in col_names)

# these are all the boundary layer regions which i split into the respective interior regions
Gmix_bl_sum = sum(ds_gmix["Gmix_bl_$c"][:, :] for c in col_names)

# we need to add the plume to this 
Gmix_plume = ds_gmix["Gmix_plume"][:, :]

# now we need to add gsurf to this
GBF = sum(ds_gmix["Gsurf_$c"][:,:] for c in col_names)

# now add them all up
G_all = Gmix_interior_sum .+ Gmix_bl_sum .+ Gmix_plume .+ GBF

t = ds_gmix["time"][:]
b = ds_gmix["b"][:]

println("extrema(G_all) = ", extrema(G_all))

# robust symmetric range: clip to a high percentile of |G_all| rather than the
# literal max, so a handful of outlier cells don't wash out the rest of the field
clim = nanquantile(abs.(vec(G_all)), 0.98)
@printf("98th percentile of |G_all| = %.4e\n", clim)

maximum(GBF)
minimum(GBF)

fig = Figure(size=(800,700))
ax = Axis(fig[1, 1], xlabel=L"\text{time (s)}", ylabel = L"\text{buoyancy (m s}^{-1})", title=L"\text{Domain Total } \mathcal{G}_{\text{MIX}} \, + \mathcal{G}_{BF} \text{ vs time and buoyancy for Flat Control Ra = } 10^6")
hm = heatmap!(ax, t, b, G_all', colorrange=(-clim, clim), colormap=:balance)
Colorbar(fig[1,2], hm, label=L"\mathcal{G}_{mix} + \mathcal{G}^{bf}")
fig

save(joinpath(plot_dir, "G_all_hovmoller.png"), fig)
