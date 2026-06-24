# plot_dissipation.jl
#
# Plotting routines for dissipation diagnostics (χ, ε) and the overturning
# streamfunction.  These were previously tangled inside src/analysis.jl; the
# physics now lives in TopographicHorizontalConvection (dissipation_analysis,
# get_ψ), and this script is the thin plotting layer that calls it.
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/plot_dissipation.jl

using TopographicHorizontalConvection   # physics: dissipation_analysis, get_ψ
using NCDatasets
using CairoMakie
using NaNStatistics

# overlay the normalized volume-averaged dissipation (diss/constraint) vs τ_eq-scaled time
function plot_normalized(ax, ds, var)
    constraint, diss_avg, diss_mean = dissipation_analysis(ds, var)
    Ra = ds.attrib["Ra"]
    τ_eq = sqrt(Ra)
    time = ds["time"][:]
    time_norm = time ./ τ_eq  # Normalize time

    plot!(ax, time_norm, diss_avg/constraint, label = "Ra = $(Ra)")
end

# overlay the raw volume-averaged dissipation vs τ_eq-scaled time
function plot_avg(ax, ds, var)
    constraint, diss_avg, diss_mean = dissipation_analysis(ds, var)
    Ra = ds.attrib["Ra"]
    τ_eq = sqrt(Ra)
    time = ds["time"][:]
    time_norm = time ./ τ_eq  # Normalize time

    plot!(ax, time_norm, diss_avg, label = "Ra = $(Ra)")
end

# grid of equilibrium-averaged log(diss/theory) heatmaps with buoyancy contours
function diss_norm_eq_subplots(datasets, datasets_buoy, var, CR; cols=2, main_title="")
    n_datasets = length(datasets)
    rows = ceil(Int, n_datasets / cols)

    fig = Figure(size=(500*cols + 100, 360*rows))  # Extra width for single colorbar

    # Add main title if provided
    if !isempty(main_title)
        Label(fig[0, :], main_title, fontsize=25, tellwidth=false)
    end

    heatmaps = []  # Store heatmap objects
    hill_maps = [] #store heatmap for hills

    for (i, ds) in enumerate(datasets)
        row = ceil(Int, i / cols)
        col = ((i - 1) % cols) + 1

        # Get data for this dataset
        diss_var = ds[var][:,:,:]
        Ra = ds.attrib["Ra"]
        time = ds["time"][:]
        x = ds["x_caa"][:]
        z = ds["z_aac"][:]
        equilibrium_start = round(Int, 0.9 * size(time, 1))
        #now this array contains values only for the equilibrium period
        var_eq = diss_var[:, :, equilibrium_start:end]

        #adding a wet mask so the hills are a different color in the heatmap
        wet = var_eq.!= 0
        wet_masked = Float64.(copy(wet))
        wet_masked[wet] .= NaN # Set wet areas to NaN for masking

        #we want to take the average of the data over the equilibrium time
        var_eq_avg = nanmean(var_eq, dims=3)
        var_eq_avg_2d = dropdims(var_eq_avg; dims=3)



        var_theory = dissipation_analysis(ds, var)[1]
        var_norm = var_eq_avg_2d ./ var_theory

        #lets try plotting the log of the heatmap
        log_var_norm = log10.(var_norm)

        # ----------- now for the buoyancy contours -------------------
        buoy_ds = datasets_buoy[i]
        b = buoy_ds["b"][:,:,:]
        #okay soooo the fucked issue is that the time_interval that the data sets were outputting at were different for oceanostics data and buoy data
        #i fixed this for the new sims, but for the prelim work im still using some older data (and im not tryna rerun all my sims)
        #so i need to make sure i recalculate the equilibrium time for buoyancy
        #ill be able to remove these steps in the future assuming I use newly run data

        time_b = buoy_ds["time"][:]
        eq_start_b = round(Int, 0.9 * size(time_b, 1))
        b_eq = b[:, :, eq_start_b:end]
        wet_b = b_eq.!= 0
        b_eq[.!wet_b] .= NaN # Set wet areas to NaN so it doesnt show up in contours

        #similarly now we want to take the average of the buoyancy data over the equilibrium time
        b_eq_avg = nanmean(b_eq, dims=3)
        b_eq_avg_2d = dropdims(b_eq_avg; dims=3)

        Nx = ds.attrib["Nx"]
        Nz = ds.attrib["Nz"]

        # ------------ now its time to plot -------------- letsgooooooooo
        ax = Axis(fig[row, col],
                xlabel=L"\hat{x}", ylabel=L"\hat{z}",
                title="Resolution = $Nx, $Nz")

        ax.titlesize = 39
        ax.xlabelsize = 33
        ax.ylabelsize = 33
        ax.xticklabelsize = 25
        ax.yticklabelsize = 25

        # -------- heatmaps for the dissipation variables --------------
        if var == "χ"
            color = :delta
        elseif var == "ε"
            color = :curl
        end

        print(maximum(log_var_norm))
        print(minimum(log_var_norm))

        hm = heatmap!(ax, x, z, log_var_norm; colorrange=CR, colormap=color)

        # ---------- adding hills as a different color -----------------
        hm_hill = heatmap!(ax, x, z, wet_masked[:,:,1], colormap=:turbid)

        # ---------- adding buoyancy contours over the heatmap ----------
        contour!(ax, x, z, b_eq_avg_2d, linewidth=1.0, color=:red, levels=LinRange(-1, 1, 15))


        push!(heatmaps, hm)
        push!(hill_maps, hm_hill)
    end

    # Add single colorbar on the right side
    if var == "χ"
        cblabel = L"\frac{χ}{χ_{theoretical}}"
    elseif var == "ε"
        cblabel = L"\frac{ε}{ε_{theoretical}}"
    end

    Colorbar(fig[:, end+1], heatmaps[1], label=cblabel, labelsize=38, ticklabelsize=25)


    return fig
end


# grid of equilibrium-averaged streamfunction heatmaps with buoyancy contours
function streamfunction_subplots(buoy_datasets, vel_datasets; cols=2, main_title="")
    n_datasets = length(buoy_datasets)
    rows = ceil(Int, n_datasets / cols)

    fig = Figure(size=(500*cols + 150, 300*rows))

    # Add main title if provided
    if !isempty(main_title)
        Label(fig[0, :], main_title, fontsize=20, tellwidth=false)
    end

    heatmaps = []  # Store heatmap objects for shared colorbar

    for (i, (ds_b, ds_v)) in enumerate(zip(buoy_datasets, vel_datasets))
        row = ceil(Int, i / cols)
        col = ((i - 1) % cols) + 1

        # Get spatial coordinates and time
        x = ds_v["x_caa"][:]
        z = ds_v["z_aac"][:]
        time = ds_v["time"]
        eq_start = round(Int, 0.9 * size(time, 1))

        # Process buoyancy data
        b_eq = ds_b["b"][:, :, eq_start:end]
        wet = b_eq .!= 0
        b_eq[.!wet] .= NaN
        b_avg = nanmean(b_eq, dims=3)
        b_avg_2d = dropdims(b_avg, dims=(3))

        #create a mask for the wet points
        wet_mask = Float64.(copy(wet))
        wet_mask[wet] .= NaN  # Set wet points to NaN

        # Process streamfunction data (physics from TopographicHorizontalConvection)
        ψ = get_ψ(ds_v)
        ψ_eq = ψ[:, :, eq_start:end]
        ψ_avg = nanmean(ψ_eq, dims=3)
        ψ_avg_2d = dropdims(ψ_avg, dims=(3))

        # Get Ra for title
        Ra = ds_b.attrib["Ra"]

        # Create subplot
        ax = Axis(fig[row, col],
                xlabel=L"\hat{x}", ylabel=L"\hat{z}",
                title="Ra = $Ra")

        ax.titlesize = 39
        ax.xlabelsize = 33
        ax.ylabelsize = 33
        ax.xticklabelsize = 25
        ax.yticklabelsize = 25

        # Create heatmap and contour
        hm = heatmap!(ax, x, z, -ψ_avg_2d,
                     colormap=(:balance, 0.9),
                     colorrange=((-nanmaximum(ψ_avg_2d)), (nanmaximum(ψ_avg_2d))))
        #plot hills as a different color
        heatmap!(ax, x, z, wet_mask[:,:,1], colormap=:turbid)
        contour!(ax, x, z, b_avg_2d, labels=true, levels=10, linewidth=2)

        push!(heatmaps, hm)
    end

    # Add single colorbar on the right side
    Colorbar(fig[:, end+1], heatmaps[1], label=L"\text{Nondimensional Streamfunction } \hat{\psi}", labelsize=38, ticklabelsize=25)

    return fig
end
