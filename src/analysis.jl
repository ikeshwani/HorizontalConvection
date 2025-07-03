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
        wet = var_t.!=0.
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
        var_array[n] = integral / wet_volume
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


function ε_analysis(ds)
    Ra = ds.attrib["Ra"]
    Pr = ds.attrib["Pr"]
    ν = ds.attrib["ν"]
    κ = ds.attrib["κ"]
    b★ = ds.attrib["b★"]
    H = ds.attrib["H"]
    Lx = ds.attrib["Lx"]

    #find the theoretical constraint on ε
    ε_theory = κ .* H^(-1) .* 2 .* b★

    #find the global volume averaged simulation ε
    #this function takes the global volume integral of ε
    #and divides it by the volume of the wet ocean (so works for both hills & no hills)
    ε_avg = global_volume_avg(ds, "ε")

    #find the mean of ε_avg --- ONLY over the equilibrium period
    #assuming the last 10% of the time is the equilibrium period
    equilibrium_start = round(Int, 0.9 * size(ε_avg, 1))
    ε_avg_eq = ε_avg[equilibrium_start:end]
    ε_mean = nanmean(ε_avg_eq)

    return ε_theory, ε_avg, ε_mean
    
end

function plot_ε_normalized(ax,ds)
    ε_theory, ε_avg, ε_mean = ε_analysis(ds)
    Ra = ds.attrib["Ra"]
    τ_eq = sqrt(Ra)
    time = ds["time"][:]
    time_norm = time ./ τ_eq  # Normalize time

    plot!(ax, time_norm, ε_avg/ε_theory, label = "Ra = $(Ra)")
end

function plot_ε_avg(ax,ds)
    ε_theory, ε_avg, ε_mean = ε_analysis(ds)
    Ra = ds.attrib["Ra"]
    τ_eq = sqrt(Ra)
    time = ds["time"][:]
    time_norm = time ./ τ_eq  # Normalize time

    plot!(ax, time_norm, ε_avg, label = "Ra = $(Ra)")
end