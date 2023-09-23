using Printf
using CairoMakie
using Oceananigans
using Oceananigans.Fields
using Oceananigans.AbstractOperations: volume

saved_output_filename = "../output/turbulent_convection_hills.jld2"

## Open the file with our data
s_timeseries = FieldTimeSeries(saved_output_filename, "s")
b_timeseries = FieldTimeSeries(saved_output_filename, "b")
ζ_timeseries = FieldTimeSeries(saved_output_filename, "ζ")
χ_timeseries = FieldTimeSeries(saved_output_filename, "χ")

times = b_timeseries.times

## Coordinate arrays
xc, yc, zc = nodes(b_timeseries[1])
xζ, yζ, zζ = nodes(ζ_timeseries[1])
nothing # hide

# Now we're ready to animate using Makie.

@info "Making an animation from saved data..."

n = Observable(1)

title = @lift @sprintf("t=%1.2f", times[$n])

sₙ = @lift interior(s_timeseries[$n], :, 1, :)
ζₙ = @lift interior(ζ_timeseries[$n], :, 1, :)
bₙ = @lift interior(b_timeseries[$n], :, 1, :)
χₙ = @lift interior(χ_timeseries[$n], :, 1, :)

slim = 0.5
blim = 0.5
ζlim = 10.
χlim = 0.002

H = 1.
Lx = 4H

axis_kwargs = (xlabel = L"x / H",
               ylabel = L"z / H",
               limits = ((-Lx/2, Lx/2), (-H, 0)),
               aspect = Lx / H,
               titlesize = 20)

fig = Figure(resolution = (900, 1100))

ax_s = Axis(fig[2, 1];
            title = L"speed, $(u^2+w^2)^{1/2} / (L_x b_*) ^{1/2}", axis_kwargs...)

ax_b = Axis(fig[3, 1];
            title = L"buoyancy, $b / b_*$", axis_kwargs...)

ax_ζ = Axis(fig[4, 1];
            title = L"vorticity, $(∂u/∂z - ∂w/∂x) \, (L_x / b_*)^{1/2}$", axis_kwargs...)

ax_χ = Axis(fig[5, 1];
            title = L"buoyancy dissipation, $κ |\mathbf{\nabla}b|^2 \, (L_x / {b_*}^5)^{1/2}$", axis_kwargs...)

fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

hm_s = heatmap!(ax_s, xc, zc, sₙ;
                colorrange = (0, slim),
                colormap = :speed)
Colorbar(fig[2, 2], hm_s)

hm_b = heatmap!(ax_b, xc, zc, bₙ;
                colorrange = (-blim, blim),
                colormap = :thermal)
Colorbar(fig[3, 2], hm_b)

hm_ζ = heatmap!(ax_ζ, xζ, zζ, ζₙ;
                colorrange = (-ζlim, ζlim),
                colormap = :balance)
Colorbar(fig[4, 2], hm_ζ)

hm_χ = heatmap!(ax_χ, xc, zc, χₙ;
                colorrange = (0, χlim),
                colormap = :dense)
Colorbar(fig[5, 2], hm_χ)

# And, finally, we record a movie.

frames = 1:length(times)

record(fig, "../animations/hilly_horizontal_convection.mp4", frames, framerate=8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end

