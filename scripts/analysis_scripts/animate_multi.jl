using Printf
using NCDatasets
using CairoMakie
using NaNStatistics

function make_animation(input_path, output_path)
    println("Processing: $input_path")
    
    ds = NCDataset(input_path)
    b = ds["b"]
    time = ds["time"]
    x = ds["x_caa"][:]
    y = ds["y_aca"][:]
    z = ds["z_aac"][:]
    Ra = ds.attrib["Ra"]
    H  = ds.attrib["H"]
    Lx = ds.attrib["Lx"]
    Ly = ds.attrib["Ly"]
    Ny = ds.attrib["Ny"]

    n = Observable(1)
    mid_Ny = div(Ny, 2)

    b_ref = Array(b[:, mid_Ny, :, 2]) #im using any time index that is not the initial in case theres no cold start
    wet = b_ref .!= 0.0  # bool array : true = fluid, false = hills # size = Nx, Nz
    wet_masked = Float64.(copy(wet))
    wet_masked[wet] .= NaN

    title = @lift @sprintf("buoyancy [m/s²] Ra = %.2e, t = %.2f", Ra, time[$n])
    bₙ    = @lift Array(b[:, mid_Ny, :, $n])

    axis_kwargs = (xlabel    = L"x / H",
                   ylabel    = L"z / H",
                   limits    = ((-Lx/2, Lx/2), (-H, 0)),
                   aspect    = Lx / H,
                   titlesize = 20)

    fig   = Figure(size=(800, 600))
    ax_B  = Axis(fig[1, 1]; title=title, axis_kwargs...)
    B_lims = (-1.0, 1.0)
    hm_B  = heatmap!(ax_B, x, z, bₙ; colorrange=B_lims, colormap=:balance)
    hm_hill = heatmap!(ax_B, x, z, wet_masked[:, :], colormap=:turbid)
    Colorbar(fig[1, 2], hm_B)

    frames = 1:length(time)
    record(fig, output_path, frames, framerate=20) do i
        n[] = i
    end

    close(ds)
    println("Saved: $output_path")
end

# define all nine simulations as (input, output) pairs
base_in  = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/"
base_out = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/animations/GPU/"

simulations = [
    # ("cheb_2x_stretch/b_base/Ra1e7/512_64/buoyancy.nc",  "cheb_2x_stretch/b_base/512x64/Ra1e7.mp4"),
    # ("cheb_2x_stretch/b_base/Ra1e8/512_64/buoyancy.nc",  "cheb_2x_stretch/b_base/512x64/Ra1e8.mp4"),
    # ("cheb_2x_stretch/b_base/Ra1e9/512_64/buoyancy.nc",  "cheb_2x_stretch/b_base/512x64/Ra1e9.mp4"),
    ("cheb_2x_stretch/b_base/Ra1e10/512_64/buoyancy.nc", "cheb_2x_stretch/b_base/512x64/Ra1e10.mp4"),
    # ("cheb_5x_stretch/b_base/Ra1e7/512_64/buoyancy.nc", "cheb_5x_stretch/b_base/512x64/Ra1e7.mp4"),
    # ("cheb_5x_stretch/b_base/Ra1e8/512_64/buoyancy.nc", "cheb_5x_stretch/b_base/512x64/Ra1e8.mp4"),
    # ("cheb_5x_stretch/b_base/Ra1e9/512_64/buoyancy.nc", "cheb_5x_stretch/b_base/512x64/Ra1e9.mp4"),
    ("cheb_5x_stretch/b_base/Ra1e10/512_64/buoyancy.nc", "cheb_5x_stretch/b_base/512x64/Ra1e10.mp4"),
    # ("cheb_8x_stretch/b_base/Ra1e7/512_64/buoyancy.nc", "cheb_8x_stretch/b_base/512x64/Ra1e7.mp4"),
    # ("cheb_8x_stretch/b_base/Ra1e8/512_64/buoyancy.nc", "cheb_8x_stretch/b_base/512x64/Ra1e8.mp4"),
    # ("cheb_8x_stretch/b_base/Ra1e9/512_64/buoyancy.nc", "cheb_8x_stretch/b_base/512x64/Ra1e9.mp4"),
    ("cheb_8x_stretch/b_base/Ra1e10/512_64/buoyancy.nc", "cheb_8x_stretch/b_base/512x64/Ra1e10.mp4")
    # ... add remaining sims
]

# for (rel_in, fname_out) in simulations
#     input_path  = joinpath(base_in, rel_in)
#     output_path = joinpath(base_out, fname_out)
#     mkpath(dirname(output_path))
#     make_animation(input_path, output_path)
# end

function compute_nusselt(ds, ds_v)

    """
    using the def for nusselt number : Nu = <wb>/(κ b★ / H)
    """
    # load attributes
    κ    = ds.attrib["κ"]
    bstar = ds.attrib["b★"]   # adjust key name to match yours
    H    = ds.attrib["H"]
    Lx   = ds.attrib["Lx"]
    Ly   = ds.attrib["Ly"]
    Nx, Ny, Nz = ds.attrib["Nx"], ds.attrib["Ny"], ds.attrib["Nz"]
    time = ds["time"][:]
    Nt   = length(time)

    # grid spacings for volume averaging
    Δx = reshape(ds["Δx_caa"][:], Nx, 1, 1)
    Δy = reshape(ds["Δy_aca"][:], 1, Ny, 1)
    Δz = reshape(ds["Δz_aac"][:], 1, 1, Nz)
    ΔV = Δx .* Δy .* Δz

    # load fields lazily
    b = ds["b"]
    w = ds_v["w"]   # face-centered in z, size Nz+1

    b_ref = Array(b[:, :, :, 2])
    wet = b_ref .!= 0.0

    wet_masked = Float32.(copy(wet))
    wet_masked[wet] .= NaN

    wet_vol = nansum(
        wet .* ΔV,
        dims = (1, 2, 3)
    )[1, 1, 1]

    Nu = zeros(Float64, Nt)

    for n in 1:Nt
        b_n = Array(b[:, :, :, n])           # (Nx, Ny, Nz)
        w_n = Array(w[:, :, :, n])           # (Nx, Ny, Nz+1)

        # interpolate w to cell centers
        w_interp = 0.5 .* (w_n[:, :, 1:end-1] .+ w_n[:, :, 2:end])  # (Nx, Ny, Nz)

        # mask land
        wet = b_n .!= 0.0
        b_n[.!wet]      .= 0.0
        w_interp[.!wet] .= 0.0

        # volume averaged wb
        wb_mean = sum(w_interp .* b_n .* ΔV) / wet_vol

        # diffusive scale
        diffusive_flux = κ * bstar / H

        Nu[n] = wb_mean / diffusive_flux
    end

    return time, Nu
end

# run over all 12 simulations and plot
base_in = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/"

simulations = [
    ("cheb_2x_stretch/b_base/Ra1e7/512_64/buoyancy.nc",  "2x",  1e7),
    ("cheb_2x_stretch/b_base/Ra1e8/512_64/buoyancy.nc",  "2x",  1e8),
    ("cheb_2x_stretch/b_base/Ra1e9/512_64/buoyancy.nc",  "2x",  1e9),
    ("cheb_2x_stretch/b_base/Ra1e10/512_64/buoyancy.nc", "2x",  1e10),
    ("cheb_5x_stretch/b_base/Ra1e7/512_64/buoyancy.nc",  "5x",  1e7),
    ("cheb_5x_stretch/b_base/Ra1e8/512_64/buoyancy.nc",  "5x",  1e8),
    ("cheb_5x_stretch/b_base/Ra1e9/512_64/buoyancy.nc",  "5x",  1e9),
    ("cheb_5x_stretch/b_base/Ra1e10/512_64/buoyancy.nc", "5x",  1e10),
    ("cheb_8x_stretch/b_base/Ra1e7/512_64/buoyancy.nc",  "8x",  1e7),
    ("cheb_8x_stretch/b_base/Ra1e8/512_64/buoyancy.nc",  "8x",  1e8),
    ("cheb_8x_stretch/b_base/Ra1e9/512_64/buoyancy.nc",  "8x",  1e9),
    ("cheb_8x_stretch/b_base/Ra1e10/512_64/buoyancy.nc", "8x",  1e10),
]

# colors  = Dict("2x" => :blue, "5x" => :orange, "8x" => :red)
# Ra_vals = [1e7, 1e8, 1e9, 1e10]

# fig = Figure(size=(1200, 800))

# for (i, Ra_target) in enumerate(Ra_vals)
#     ax = Axis(fig[div(i-1,2)+1, mod(i-1,2)+1],
#               xlabel = "time",
#               ylabel = "Nu",
#               title  = @sprintf("Ra = %.0e", Ra_target))

#     for (path, stretch, Ra) in simulations
#         Ra != Ra_target && continue

#         #separate velocity and buoyancy paths
#         b_path = joinpath(base_in, path)
#         vel_path = replace(b_path, "buoyancy.nc" => "velocities.nc")

#         ds = NCDataset(b_path)
#         ds_v = NCDataset(vel_path)
#         time, Nu = compute_nusselt(ds, ds_v)

#         lines!(ax, time, Nu, label="$(stretch) stretch", color=colors[stretch])
#         close(ds)
#     end

#     axislegend(ax)
# end

# save("/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU_test/nusselt_comparison.png", fig)


base_in = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/"

# Ra_vals      = [1e7, 1e8, 1e9, 1e10]
# stretches    = ["2x", "5x", "8x"]
# stretch_vals = [2, 5, 8]  # numeric x-axis values
# Ra_labels = Dict(1e7 => "Ra1e7", 1e8 => "Ra1e8", 1e9 => "Ra1e9", 1e10 => "Ra1e10")

# # store results in a dict keyed by (stretch, Ra)
# results = Dict()

# # plot
# colors  = Dict("2x" => :blue, "5x" => :orange, "8x" => :red)
# t_target = 9.0

# fig = Figure(size=(1000, 1200))

# for (i, Ra) in enumerate(Ra_vals)
#     ax_left = Axis(fig[i, 1], 
#                 xlabel = " b [m/s²]", 
#                 ylabel = " z / H", 
#                 title = @sprintf("Ra = %.0e, x = -3", Ra))

#     ax_right = Axis(fig[i, 2], 
#                 xlabel = "b [m/s²]", 
#                 ylabel = " z / H", 
#                 title = @sprintf("Ra = %.0e, x = 3", Ra))

#     for stretch in stretches 
#         b_path = joinpath(base_in, "cheb_$(stretch)_stretch/b_base/$(Ra_labels[Ra])/512_64/buoyancy.nc")
#         ds = NCDataset(b_path)

#         time = ds["time"][:]
#         x = ds["x_caa"][:]
#         z = ds["z_aac"][:]
#         Ny = ds.attrib["Ny"]
#         Lx = ds.attrib["Lx"]
#         mid_Ny = div(Ny, 2)

#         #calculate the boundary layer height, h
#         h = -Lx * Ra^(-1/5)

#         i_left = argmin(abs.(x .+ 3.0))
#         i_right = argmin(abs.(x .- 3.0))
#         k_top = argmin(abs.(z .+ 0.3))

#         n = argmin(abs.(time .- t_target))

#         b_left = Array(ds["b"][i_left, mid_Ny, :, n])
#         b_right = Array(ds["b"][i_right, mid_Ny, :, n])

#         b_left[b_left .== 0] .= NaN
#         b_right[b_right .== 0] .= NaN

#         lines!(ax_left, b_left[k_top:end], z[k_top:end], color=colors[stretch], label = "$(stretch) stretch")
#         hlines!(ax_left, h, color=:green, linestyle=:dash, label = "boundary layer height from theory")

#         lines!(ax_right, b_right[k_top:end], z[k_top:end], color=colors[stretch], label = "$(stretch) strech")
#         hlines!(ax_right, h, color=:green, linestyle=:dash, label = "boundary layer height from theory")
        

#         close(ds)
#     end

#     if i == 1
#         Legend(fig[i, 3], ax_left, framevisible=false)
#     end
# end

# Label(fig[0, 1], "left boundary (x = -3)", fontsize=16, font=:bold)
# Label(fig[0,2], "right boundary (x = +3)", fontsize=16, font=:bold)

# save("/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU_test/vert_buoy_profiles_zoomed_withBL.png", fig)

############################

######## nusselt number plot from diffusive versus advective data ###########

#first we have a function to compute the integrated dissipation

function global_volume_integral(ds, var)
    x, y, z = ds["x_caa"][:] , ds["y_aca"][:], ds["z_aac"][:]
    Nx, Ny, Nz = ds.attrib["Nx"] , ds.attrib["Ny"], ds.attrib["Nz"]
    time = ds["time"][:]
    Δx = reshape(ds["Δx_caa"][:], Nx,1,1)
    Δy = reshape(ds["Δy_aca"][:], 1, Ny, 1)
    Δz = reshape(ds["Δz_aac"][:], 1,1,Nz)
    ΔA = Δx .* Δy #three dimensional
    ΔV = ΔA.*Δz
    var_array = zeros(size(time,1))
    for n in 1:size(time, 1)
        var_t = ds[var][:, :, :, n] #3D
        wet = var_t.!=0.
        var_t[.!wet] .= NaN 
        var_array[n] = nansum(
            var_t .*
            ΔV,
            dims=(1,2,3)
        )[1,1,1]
    end  
    return var_array
end

function compute_NU_from_diff(diff_ds, adv_ds)
    time_diff, χ_diff = global_volume_integral(diff_ds, "chi")
    time_adv, χ_adv   = global_volume_integral(adv_ds, "chi")

    # i know that my time arrays won't match up, they have all been run for different times unfortunately
    n = min(length(time_diff), length(time_adv))

    Nu = χ_adv[1:n] ./ χ_diff[1:n]

    return Nu
end


# run over all 12 simulations and plot
base_in = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/"

simulations = [
    ("cheb_2x_stretch/b_base/Ra1e7/512_64/buoyancy.nc",  "2x",  1e7),
    ("cheb_2x_stretch/b_base/Ra1e8/512_64/buoyancy.nc",  "2x",  1e8),
    ("cheb_2x_stretch/b_base/Ra1e9/512_64/buoyancy.nc",  "2x",  1e9),
    # ("cheb_2x_stretch/b_base/Ra1e10/512_64/buoyancy.nc", "2x",  1e10),
    ("cheb_5x_stretch/b_base/Ra1e7/512_64/buoyancy.nc",  "5x",  1e7),
    ("cheb_5x_stretch/b_base/Ra1e8/512_64/buoyancy.nc",  "5x",  1e8),
    ("cheb_5x_stretch/b_base/Ra1e9/512_64/buoyancy.nc",  "5x",  1e9),
    # ("cheb_5x_stretch/b_base/Ra1e10/512_64/buoyancy.nc", "5x",  1e10),
    ("cheb_8x_stretch/b_base/Ra1e7/512_64/buoyancy.nc",  "8x",  1e7),
    ("cheb_8x_stretch/b_base/Ra1e8/512_64/buoyancy.nc",  "8x",  1e8),
    ("cheb_8x_stretch/b_base/Ra1e9/512_64/buoyancy.nc",  "8x",  1e9),
    # ("cheb_8x_stretch/b_base/Ra1e10/512_64/buoyancy.nc", "8x",  1e10),
]

diff_sims = [
    ("cheb_2x_stretch/diffusive/b_base/Ra1e7/512_64/buoyancy.nc",  "2x",  1e7),
    ("cheb_2x_stretch/diffusive/b_base/Ra1e8/512_64/buoyancy.nc",  "2x",  1e8),
    ("cheb_2x_stretch/diffusive/b_base/Ra1e9/512_64/buoyancy.nc",  "2x",  1e9),
    # ("cheb_2x_stretch/b_base/Ra1e10/512_64/buoyancy.nc", "2x",  1e10),
    ("cheb_5x_stretch/diffusive/b_base/Ra1e7/512_64/buoyancy.nc",  "5x",  1e7),
    ("cheb_5x_stretch/diffusive/b_base/Ra1e8/512_64/buoyancy.nc",  "5x",  1e8),
    ("cheb_5x_stretch/diffusive/b_base/Ra1e9/512_64/buoyancy.nc",  "5x",  1e9),
    # ("cheb_5x_stretch/b_base/Ra1e10/512_64/buoyancy.nc", "5x",  1e10),
    ("cheb_8x_stretch/diffusive/b_base/Ra1e7/512_64/buoyancy.nc",  "8x",  1e7),
    ("cheb_8x_stretch/diffusive/b_base/Ra1e8/512_64/buoyancy.nc",  "8x",  1e8),
    ("cheb_8x_stretch/diffusive/b_base/Ra1e9/512_64/buoyancy.nc",  "8x",  1e9),
    # ("cheb_8x_stretch/b_base/Ra1e10/512_64/buoyancy.nc", "8x",  1e10),
]

colors  = Dict("2x" => :blue, "5x" => :orange, "8x" => :red)
Ra_vals = [1e7, 1e8, 1e9]


# Compute Nusselt numbers for all combinations
for (adv_sim, diff_sim) in zip(simulations, diff_sims)
    adv_path, stretch, Ra = adv_sim
    diff_path, _, _ = diff_sim
    
    println("Processing Ra = $Ra, stretch = $stretch")
    
    # Open datasets
    adv_ds = NCDataset(joinpath(base_in, adv_path))
    diff_ds = NCDataset(joinpath(base_in, diff_path))
    
    # Compute Nusselt number
    time, Nu = compute_NU_from_diff(diff_ds, adv_ds)
    
    # Store results
    key = (Ra = Ra, stretch = stretch)
    results[key] = (time = time, Nu = Nu)
    
    # Close datasets
    close(adv_ds)
    close(diff_ds)
    
    println("  Final Nu: $(Nu[end])")
    println("  Mean Nu (2nd half): $(mean(Nu[end÷2:end]))")
end

# Create figure with 3 panels (one per Ra)
fig = Figure(size = (1600, 500))

for (i, Ra) in enumerate(Ra_vals)
    ax = Axis(fig[1, i],
        xlabel = "Time",
        ylabel = i == 1 ? "Nusselt Number" : "",
        title = "Ra = $(Ra)"
    )
    
    # Plot each stretch for this Ra
    for stretch in ["2x", "5x", "8x"]
        key = (Ra = Ra, stretch = stretch)
        
        if haskey(results, key)
            result = results[key]
            
            lines!(ax, result.time, result.Nu,
                color = colors[stretch],
                linewidth = 2,
                label = stretch
            )
            
            # Add mean line for second half
            Nu_mean = mean(result.Nu[end÷2:end])
            hlines!(ax, [Nu_mean],
                color = (colors[stretch], 0.4),
                linestyle = :dash,
                linewidth = 1.5
            )
        end
    end
    
    # Only show legend on last panel
    if i == 3
        axislegend(ax, position = :rb)
    end
end

save("nusselt_comparison_from_diffusive_sims.png", fig)
display(fig)

# Print summary table
println("\n" * "="^60)
println("NUSSELT NUMBER SUMMARY")
println("="^60)
println(@sprintf("%-10s %-10s %-15s %-15s", "Ra", "Stretch", "Final Nu", "Mean Nu (2nd half)"))
println("-"^60)

for Ra in Ra_vals
    for stretch in ["2x", "5x", "8x"]
        key = (Ra = Ra, stretch = stretch)
        if haskey(results, key)
            result = results[key]
            Nu_final = result.Nu[end]
            Nu_mean = mean(result.Nu[end÷2:end])
            println(@sprintf("%-10.0e %-10s %-15.3f %-15.3f", Ra, stretch, Nu_final, Nu_mean))
        end
    end
end