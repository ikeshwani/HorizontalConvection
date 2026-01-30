using TopographicHorizontalConvection: HorizontalConvectionSimulation
using Printf
using NCDatasets
using CairoMakie
using Oceananigans
using Oceananigans.Fields
using Oceananigans.AbstractOperations: volume

saved_output_filename = NCDataset(string("/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/b_seasonal/Ra1e6/320_40/buoyancy.nc"));

b_timeseries = saved_output_filename["b"][:,:,:,:] #3d 
time = saved_output_filename["time"]

t_final = time[end]

x = saved_output_filename["x_caa"][:]
y = saved_output_filename["y_aca"][:]
z = saved_output_filename["z_aac"][:]

Ra = saved_output_filename.attrib["Ra"]

@info "Making an animation from saved data..."

n = Observable(1)
H = saved_output_filename.attrib["H"]
Lx = saved_output_filename.attrib["Lx"]
Ly = saved_output_filename.attrib["Ly"]
Ny = saved_output_filename.attrib["Ny"]

title = @lift @sprintf("buoyancy [m/s²] of experiment with Ra = %.2e at t = %.2f", Ra, time[$n])

mid_Ny = div(Ny, 2)

bₙ = @lift b_timeseries[:, mid_Ny, :, $n]

axis_kwargs = (xlabel = L"x / H",
               ylabel = L"z / H",
               limits = ((-Lx/2, Lx/2), (-H, 0)),
               aspect = Lx / H,
               titlesize = 20)

fig = Figure(size=(800, 600))

ax_B = Axis(fig[1, 1]; title = title, axis_kwargs...)

B_lims = (-1.0, 1.0)

hm_B = heatmap!(ax_B, x, z, bₙ; colorrange = B_lims, colormap = :balance)
Colorbar(fig[1, 2], hm_B)

frames = 1:length(time)

record(fig, "/work/hdd/bfxn/ikeshwani/HorizontalConvection/animations/GPU/b_seasonal.mp4", frames, framerate=8) do i
    n[] = i
end
