using TopographicHorizontalConvection: HorizontalConvectionSimulation
using Printf
using NCDatasets
using CairoMakie
using Oceananigans
using Oceananigans.Fields
using Oceananigans.AbstractOperations: volume

saved_output_filename = NCDataset(string("/Users/hfdrake/code/HorizontalConvection/output/tanhforcing/256_32/buoyancy.nc"));

b_timeseries = saved_output_filename["b"][:,:,:] #2d 
time = saved_output_filename["time"]

t_final = time[end]

x = saved_output_filename["x_caa"][:]
#y = saved_output_filename["yC"][:] # no y-dimension for 2d
z = saved_output_filename["z_aac"][:]

@info "Making an animation from saved data..."

n = Observable(1)

title = @lift @sprintf("buoyancy [m/s²] at t = %.2f", time[$n])

bₙ = @lift b_timeseries[:, :, $n]

H = saved_output_filename.attrib["H"]
Lx = saved_output_filename.attrib["Lx"]

axis_kwargs = (xlabel = L"x / H",
               ylabel = L"z / H",
               limits = ((-4, 4), (-1, 0)),
               aspect = Lx / H,
               titlesize = 20)

fig = Figure(size=(800, 600))

ax_B = Axis(fig[1, 1]; title = title, axis_kwargs...)

B_lims = (-maximum(b_timeseries), maximum(b_timeseries))

hm_B = heatmap!(ax_B, x, z, bₙ; colorrange = B_lims, colormap = :balance)
Colorbar(fig[1, 2], hm_B)

frames = 1:length(time)

record(fig, "/Users/hfdrake/code/HorizontalConvection/animations/tanhforcingshortperiod.mp4", frames, framerate=8) do i
    n[] = i
end
