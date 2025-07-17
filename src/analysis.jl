using Oceananigans
using NCDatasets
using Printf
using CairoMakie
using Oceananigans.Fields
using Oceananigans.AbstractOperations: volume
using NaNStatistics


#global volume integral function to generalize find χ and find ϵ

function global_volume_integral(ds, var)
    x = ds["xC"][4+1:end-4]; Nx = length(x);
    z = ds["zC"][4+1:end-4]; Nz = length(z);
    time = ds["time"][:];
    Δx = reshape(diff(ds["xF"])[4+1:end-4], Nx,1,1);
    Δz = reshape(diff(ds["zF"])[4+1:end-4], 1,1,Nz);
    ΔA = Δx; #flat in y -- 2 dimensional
    ΔV = ΔA.*Δz;
    var_array = zeros(size(time,1));
    for n in 1:size(time, 1)
        var_t = ds[var][4+1:end-4, :, 4+1:end-4, n]
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

#function to find the volume averaged variable (generalized for χ or ε)
function global_volume_avg(ds, var)
    x = ds["xC"][4+1:end-4]; Nx = length(x);
    z = ds["zC"][4+1:end-4]; Nz = length(z);
    time = ds["time"][:];
    Δx = reshape(diff(ds["xF"])[4+1:end-4], Nx,1,1);
    Δz = reshape(diff(ds["zF"])[4+1:end-4], 1,1,Nz);
    ΔA = Δx; #flat in y -- 2 dimensional
    ΔV = ΔA.*Δz;
    var_array = zeros(size(time,1));
    for n in 1:size(time, 1)
        var_t = ds[var][4+1:end-4, :, 4+1:end-4, n]
        wet = ds[var][4+1:end-4, :, 4+1:end-4, 3].!=0.
        var_t[.!wet] .= NaN

         #volume integral over "wet" variable
        integral = nansum(
            var_t .*
            ΔV,
            dims=(1,2,3)
        )[1,1,1]

        #volume of wet ocean 
        wet_volume = nansum(
            wet .*
            ΔV,
            dims=(1,2,3)
        )[1,1,1]

        #now we find the volume averaged variable
        var_array[n] = integral ./ wet_volume
    end
    return var_array
end

#function to find the bottom buoyancy average

function buoyancy_bottom_avg(ds)
    b_bottom = ds["b"][4+1:end-4,1,4+1,:]
    bottom_avg = zeros(size(b_bottom[1,:]));
    for n in 1:size(b_bottom,2)
        bb = b_bottom[:,n]
        wet = bb.!=0.
        bb[.!wet] .= NaN

        bottom_avg[n] = nanmean(bb, dims=1)[1,1]
    end
    return bottom_avg
end


#function to find the streamfunction ψ

function get_ψ(ds)

    function integrate_udy(ds)
        x = ds["xC"][4+1:end-4]; Nx = length(x);
        H = 1.0;
        y = ds["yC"][:]; Ny = length(y); Ly = H/4;
        Δy = Ly/Ny;
        z = ds["zC"][4+1:end-4]; Nz = length(z); 
        Δz = reshape(diff(ds["zF"])[4+1:end-4], 1,1,Nz);
        t = ds["time"][:];
    
        #the first step is calculating the integral of u over dy
        ∫udy = zeros(Nx, 1, Nz, size(t,1))
        for n in 1:size(t,1)
            ut = ds["u"][5+1:end-4, :, 4+1:end-4, n];
            wet = ut.!=0.
            ut[.!wet] .= NaN
            ∫udy[:,1,:,n] = nansum(
                ut .*
                Δy, 
                dims=(2))
        end
        return ∫udy
    end 
    x = ds["xC"][4+1:end-4]; Nx = length(x);
    z = ds["zC"][4+1:end-4]; Nz = length(z);
    Δz = reshape(diff(ds["zF"])[4+1:end-4], 1,1,Nz,1);
    t = ds["time"][:];
    ψ = zeros(Nx, Nz, size(t,1))  
    for i in 1:Nz
        ∫udy = integrate_udy(ds)[:, 1:1, 1:i, :]
        wet = ∫udy.!=0.
        ∫udy[.!wet] .= NaN
        Ψ_tmp = nansum(
            ∫udy .* 
            Δz[:,:,1:i,1],
            dims=(3)
        )[:,1,1,:]
        ψ[:,i,:] = Ψ_tmp
    end
    return ψ
end


function dissipation_analysis(ds, var)
#     this function can be used for ε or chi
#     we return the constraint theorized by paperella and young (2003) (ε) and winters and young (2009) (χ)
#     then we calculate the volume averaged dissipation variable
#     and take the mean of the last 10% of the time (equilibrium period)


    Ra = ds.attrib["Ra"]
    Pr = ds.attrib["Pr"]
    ν = ds.attrib["ν"]
    κ = ds.attrib["κ"]
    b★ = ds.attrib["b★"]
    H = ds.attrib["H"]
    Lx = ds.attrib["Lx"]

    if var == "ε"
        #find the theoretical constraint on ε
        constraint = κ .* H^(-1) .* 2 .* b★
    elseif var == "χ"
        #find the theoretical constraint on χ
        constraint = 4.57 * (κ^(1/3) * (2 * b★)^(7/3)) / (Pr^(1/3) * H)
    end

    #find the global volume averaged simulation ε
    #this function takes the global volume integral of ε
    #and divides it by the volume of the wet ocean (so works for both hills & no hills)
    vol_avg = global_volume_avg(ds, var)

    #find the mean of ε_avg --- ONLY over the equilibrium period
    #assuming the last 10% of the time is the equilibrium period
    equilibrium_start = round(Int, 0.9 * size(vol_avg, 1))
    vol_avg_eq = vol_avg[equilibrium_start:end]
    time_mean = nanmean(vol_avg_eq)

    return constraint, vol_avg, time_mean

end


function plot_normalized(ax,ds,var)
    constraint, diss_avg, diss_mean = dissipation_analysis(ds, var)
    Ra = ds.attrib["Ra"]
    τ_eq = sqrt(Ra)
    time = ds["time"][:]
    time_norm = time ./ τ_eq  # Normalize time

    plot!(ax, time_norm, diss_avg/constraint, label = "Ra = $(Ra)")
end

function plot_avg(ax,ds,var)
    constraint, diss_avg, diss_mean = dissipation_analysis(ds, var)
    Ra = ds.attrib["Ra"]
    τ_eq = sqrt(Ra)
    time = ds["time"][:]
    time_norm = time ./ τ_eq  # Normalize time

    plot!(ax, time_norm, diss_avg, label = "Ra = $(Ra)")
end


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
        diss_var = ds[var][4+1:end-4, 1, 4+1:end-4, :]
        Ra = ds.attrib["Ra"]
        time = ds["time"][:]
        x = ds["xC"][4+1:end-4]
        z = ds["zC"][4+1:end-4]
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

        # ----------- now for the buoyancy contours -------------------
        buoy_ds = datasets_buoy[i]
        b = buoy_ds["b"][4+1:end-4, 1, 4+1:end-4, :]
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

        # ------------ now its time to plot -------------- letsgooooooooo
        ax = Axis(fig[row, col], 
                xlabel="x", ylabel="z", 
                title="Ra = $Ra")
        # -------- heatmaps for the dissipation variables --------------
        hm = heatmap!(ax, x, z, var_norm, colorrange=CR, colormap=:dense)

        # ---------- adding hills as a different color -----------------
        hm_hill = heatmap!(ax, x, z, wet_masked[:,:,1], colormap=:deep)
        
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

    Colorbar(fig[:, end+1], heatmaps[1], label=cblabel)

    return fig
end



# Function to plot the streamfunction and buoyancy contours 

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
        x = ds_v["xC"][4+1:end-4]
        z = ds_v["zC"][4+1:end-4]
        time = ds_v["time"]
        eq_start = round(Int, 0.9 * size(time, 1))
        
        # Process buoyancy data
        b_eq = ds_b["b"][4+1:end-4, 1, 4+1:end-4, eq_start:end]
        wet = b_eq .!= 0
        b_eq[.!wet] .= NaN
        b_avg = nanmean(b_eq, dims=3)
        b_avg_2d = dropdims(b_avg, dims=(3))

        #create a mask for the wet points
        wet_mask = Float64.(copy(wet))
        wet_mask[wet] .= NaN  # Set wet points to NaN
        
        # Process streamfunction data
        ψ = get_ψ(ds_v)
        ψ_eq = ψ[:, :, eq_start:end]
        ψ_avg = nanmean(ψ_eq, dims=3)
        ψ_avg_2d = dropdims(ψ_avg, dims=(3))
        
        # Get Ra for title
        Ra = ds_b.attrib["Ra"]
        
        # Create subplot
        ax = Axis(fig[row, col], 
                xlabel="x", ylabel="z", 
                title="Ra = $Ra")
        
        # Create heatmap and contour
        hm = heatmap!(ax, x, z, -ψ_avg_2d, 
                     colormap=(:balance, 0.9), 
                     colorrange=((-nanmaximum(ψ_avg_2d)), (nanmaximum(ψ_avg_2d))))
        #plot hills as a different color
        heatmap!(ax, x, z, wet_mask[:,:,1], colormap=:deep)
        contour!(ax, x, z, b_avg_2d, labels=true, levels=10, linewidth=2)
        
        push!(heatmaps, hm)
    end
    
    # Add single colorbar on the right side
    Colorbar(fig[:, end+1], heatmaps[1], label="Streamfunction (ψ)")
    
    return fig
end