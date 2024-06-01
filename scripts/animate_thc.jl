using TopographicHorizontalConvection: HorizontalConvectionSimulation
using Printf
using NCDatasets
using CairoMakie
using Oceananigans
using Oceananigans.Fields
using Oceananigans.AbstractOperations: volume

saved_output_filename = NCDataset(string("/pub/hfdrake/code/HorizontalConvection/output/turbulent_h0.6_buoyancy.nc"));

b_timeseries = saved_output_filename["b"][4+1:end-4, 4+1:end-4, 4+1:end-4, :]
time = saved_output_filename["time"]

t_final = time[end]

x = saved_output_filename["xC"][4+1:end-4]
y = saved_output_filename["yC"][4+1:end-4]
z = saved_output_filename["zC"][4+1:end-4]

@info "Making an animation from saved data..."

n = Observable(1)

title = @lift @sprintf("buoyancy [m/s²] at t = %.2f", times[$n])

bₙ = @lift interior(b_timeseries, :, 1, :, $n)

H = 1
Lx = 4H

axis_kwargs = (xlabel = L"x / H",
               ylabel = L"z / H",
               limits = ((-4, 4), (-1, 0)),
               aspect = Lx / H,
               titlesize = 20)

fig = Figure(size=(800, 600))

ax_B = Axis(fig[1, 1]; title = title, kwargs...)

B_lims = (-0.9143529691691485, 0.9143529691691485)

hm_B = heatmap!(ax_B, x, z, bₙ; colorrange = B_lims, colormap = :balance)
Colorbar(fig[1, 2], hm_B)

frames = 1:length(time)

record(fig, "../animations/thc.mp4", frames, framerate=8) do i
    n[] = i
end
