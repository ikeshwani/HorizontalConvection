using CairoMakie
using Oceananigans
using Oceananigans.Fields
using Oceananigans.AbstractOperations: volume
using Statistics

saved_output_filename = "horizontal_convection.jld2"

s_timeseries = FieldTimeSeries(saved_output_filename, "s")
b_timeseries = FieldTimeSeries(saved_output_filename, "b")
ζ_timeseries = FieldTimeSeries(saved_output_filename, "ζ")

times = b_timeseries.times

#before replacing 0 buoyancy values with NaNs we have

#Create plot for buoyancy averaged over all space versus time

b_int = interior(b_timeseries)

b_mean = mean(b_int, dims =(1,2,3))[1,1,1,:]

lines(times, b_mean)

#Create heatmap for bouyancy averaged over x&y (only dependent on z) vs time

b_mean_x = mean(b_int, dims = (1, 2))[1,1,:,:]

#flip the matrix so that we have a 21x64 matrix
b_mean_x_perm = permutedims(b_mean_x, (2,1))

#create a heatmap times on the x-axis, depth on y-axis
heatmap(times, zc, b_mean_x_perm)

#Now recreate these graphs, but replace the zeros in b_int with NaNs before taking the mean

b_int = replace(b_int, NaN => 0.0)

b_nans = replace(b_int, 0.0 => NaN)

#Now use NaNStatistics to take the mean of b_nans
using NaNStatistics

b_nans_mean = nanmean(b_nans, dims = (1,2,3))[1,1,1,:]

lines(times, b_nans_mean)

#now create heatmap following the steps before

b_nans_meanx = nanmean(b_nans, dims=(1,2))[1,1,:,:]

b_nans_meanx_perm = permutedims(b_nans_meanx, (2,1))

heatmap(times, zc, b_nans_meanx_perm)

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


for i = 1:length(times_diff)
    χ_diff_diffusive = Field(Integral(χ_timeseries_diffusive[i]))
    compute!(χ_diff_diffusive)

    χ_diff_turbulent = Field(Integral(χ_timeseries_turbulent[i]))
    compute!(χ_diff_turbulent)
    
end


#lets plot the buoyancy dissipation over time for each scheme


fig = Figure(resolution = (600,400))
ax_diff = Axis(fig[1,1], xlabel= L"t \, (b_* / L_x) ^{1/2}", ylabel = "χ Diffusive", title="Diffusive Buoyancy Dissipation vs Time")
lines!(ax_diff, times_diff, χ_diff_diffusive)

ax_turb = Axis(fig[2, 1], xlabel= L"t \, (b_* / L_x) ^{1/2}", ylabel = "χ Turbulent", title = "Turbulent Buoyancy Dissipation vs Time")
lines!(ax_turb, times_turb, χ_diff_turbulent)

current_figure()
fig


