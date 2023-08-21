using CairoMakie
using Oceananigans
using Oceananigans.Fields
using Oceananigans.AbstractOperations: volume

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