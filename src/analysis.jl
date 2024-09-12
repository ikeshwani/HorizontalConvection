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
        var_t = ds[var][4+1:end-4, 1, 4+1:end-4, n]
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

#function to find the buoyancy average at the bottom or top

function buoyancy_level_avg(ds, level)
    if level == "bottom"
        b_level = ds["b"][4+1:end-4,1,4+1,:]
    elseif level == "top"
        b_level = ds["b"][4+1:end-4, 1, end-4, :]
    end
    level_avg = zeros(size(b_level[1,:]));
    for n in 1:size(b_level,2)
        bL = b_level[:,n]
        wet = bL.!=0.
        bL[.!wet] .= NaN
        level_avg[n] = nanmean(bL, dims=1)[1,1]
    end
    return level_avg     
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