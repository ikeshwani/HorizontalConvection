using CairoMakie
using Oceananigans
using Oceananigans.Fields
using Oceananigans.AbstractOperations: volume
using Statistics

saved_output_filename = "turbulent_convection_hills.jld2"

s_timeseries = FieldTimeSeries(saved_output_filename, "s")
b_timeseries = FieldTimeSeries(saved_output_filename, "b")
ζ_timeseries = FieldTimeSeries(saved_output_filename, "ζ")

times = b_timeseries.times

#before replacing 0 buoyancy values with NaNs we have

#Create plot for buoyancy averaged over all space versus time

b_int = interior(b_timeseries)

b_mean = mean(b_int, dims =(1,2,3))[1,1,1,:]

fig_buoy1 = Figure(resolution = (600, 600))
ax_buoy1 = Axis(fig_buoy1[1,1], xlabel = "Time", ylabel= "Buoyancy (b(t))", title = "Buoyancy Averaged over 3 Dimensions versus Time")
lines!(ax_buoy1, times, b_mean)
ax_heat1 = Axis(fig_buoy1[2,1], xlabel = "Time", ylabel = "Depth (z)", title = "Heatmap of Buoyancy Averaged Over x & y (b(z)) versus Time")


#Create heatmap for bouyancy averaged over x&y (only dependent on z) vs time

b_mean_x = mean(b_int, dims = (1, 2))[1,1,:,:]

#flip the matrix so that we have a 21x64 matrix
b_mean_x_perm = permutedims(b_mean_x, (2,1))

#create a heatmap times on the x-axis, depth on y-axis
heatmap!(ax_heat1, times, zc, b_mean_x_perm)
fig_buoy1
#Now recreate these graphs, but replace the zeros in b_int with NaNs before taking the mean

b_int = replace(b_int, NaN => 0.0)

b_nans = replace(b_int, 0.0 => NaN)

#Now use NaNStatistics to take the mean of b_nans
using NaNStatistics

b_nans_mean = nanmean(b_nans, dims = (1,2,3))[1,1,1,:]

fig_buoy2 = Figure(resolution=(600,600))
ax_buoy2 = Axis(fig_buoy2[1,1], xlabel="Times", ylabel="Buoyancy (b(t))", title = "Corrected Buoyancy Averaged Over 3D Versus Time")
lines!(ax_buoy2, times, b_nans_mean)
ax_heat2 = Axis(fig_buoy2[2,1], xlabel="Times", ylabel="Buoyancy b(z)", title = "Corrected Buoyancy Averaged over xy Versus Time")


#now create heatmap following the steps before

b_nans_meanx = nanmean(b_nans, dims=(1,2))[1,1,:,:]

b_nans_meanx_perm = permutedims(b_nans_meanx, (2,1))

heatmap!(ax_heat2, times, zc, b_nans_meanx_perm)
fig_buoy2
#Now that we have run an experiment with the two advection schemes 
# turbulent and diffusive we can analyze those to plot the Nusselt Number

#first open the file with our χ data

b_timeseries_diffusive = FieldTimeSeries("diffusive_convection_hills.jld2", "b")
times_diff = b_timeseries_diffusive.times
χ_timeseries_diffusive = FieldTimeSeries("diffusive_convection_hills.jld2", "χ")

b_timeseries_turbulent = FieldTimeSeries("turbulent_convection_hills.jld2", "b")
times_turb = b_timeseries_turbulent.times
χ_timeseries_turbulent = FieldTimeSeries("turbulent_convection_hills.jld2", "χ")

χₙ_diffusive = @lift interior(χ_timeseries_diffusive[$n], :, 1, :)
χₙ_turbulent = @lift interior(χ_timeseries_turbulent[$n], :, 1, :)

χ_diffusive_int = zeros(size(times_diff))
χ_turbulent_int = zeros(size(times_turb))

for i = 1:length(times_diff)
    χ_diffusive_int_snapshot = Field(Integral(χ_timeseries_diffusive[i]))
    compute!(χ_diffusive_int_snapshot)
    χ_diffusive_int[i] = χ_diffusive_int_snapshot[1,1,1]

    χ_turbulent_int_snapshot = Field(Integral(χ_timeseries_turbulent[i]))
    compute!(χ_turbulent_int_snapshot)
    χ_turbulent_int[i] = χ_turbulent_int_snapshot[1,1,1]
end

Nu = χ_turbulent_int ./ χ_diffusive_int

#lets plot the buoyancy dissipation over time for each scheme

fig = Figure(resolution = (500,800))
ax_diff = Axis(fig[1,1], xlabel= L"t \, (b_* / L_x) ^{1/2}", ylabel = "χ Diffusive", title="Diffusive Buoyancy Dissipation vs Time")
lines!(ax_diff, times_diff, χ_diffusive_int)

ax_turb = Axis(fig[2, 1], xlabel= L"t \, (b_* / L_x) ^{1/2}", ylabel = "χ Turbulent", title = "Turbulent Buoyancy Dissipation vs Time")
lines!(ax_turb, times_turb, χ_turbulent_int)

ax_Nu = Axis(fig[3, 1], xlabel= L"t \, (b_* / L_x) ^{1/2}", ylabel = L"Nu $= \frac{\langle χ_{Turb} \rangle }{\langle χ_{Diff} \rangle}$", title = "Nusselt number vs Time", limits=((0, nothing), (0, nothing)))
lines!(ax_Nu, times_turb, Nu)

current_figure()
fig


