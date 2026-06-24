# energy_fluxes.jl
#
# Plot the reversible buoyancy flux ϕ_z and the energy-supply flux ϕ_i as
# functions of time.
#
# Thin script: vertical_b_flux / phi_i physics live in
# TopographicHorizontalConvection; this file loads the dataset, calls them,
# and plots.
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/energy_fluxes.jl

using TopographicHorizontalConvection   # physics: vertical_b_flux, phi_i
using NCDatasets
using CairoMakie

# ---- config ----
data_file = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/cheb_5x_stretch/b_base/Ra1e7/512_64/buoyancy.nc"
plot_dir  = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU_test/cheby_grid_verification/"
mkpath(plot_dir)

ds = NCDataset(data_file, "r")

ϕ_i_ra7 = phi_i(ds)
ϕ_z_ra7 = vertical_b_flux(ds)
time    = ds["time"][:]

fig = Figure()
ax = Axis(fig[1,1], xlabel="time", ylabel="dissipation param",
          title = "ϕ_z and ϕ_i as function of time for Ra=1e7 5x grid stretch")

scatter!(ax, time, ϕ_i_ra7, label = "ϕ_i")
scatter!(ax, time, ϕ_z_ra7, label = "ϕ_z")

fig[1,2] = axislegend()

save(joinpath(plot_dir, "phi_i_z_testra7_5x.png"), fig)
