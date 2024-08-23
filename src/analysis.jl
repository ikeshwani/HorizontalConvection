using Oceananigans
using NCDatasets
using Printf
using CairoMakie
using Oceananigans.Fields
using Oceananigans.AbstractOperations: volume
using NaNStatistics


#function to find  χ same method used for turbulent and diffusive
function find_χ(ds)
    x = ds["xC"][4+1:end-4]; Nx = length(x);
    z = ds["zC"][4+1:end-4]; Nz = length(z);
    time = ds["time"][:];
    Δx = reshape(diff(ds["xF"])[4+1:end-4], Nx,1,1);
    Δz = reshape(diff(ds["zF"])[4+1:end-4], 1,1,Nz);
    ΔA = Δx; #flat in y -- 2 dimensional
    ΔV = ΔA.*Δz;
    χ = zeros(size(time,1));
    for n in 1:size(time, 1)
        χt = ds["chi"][4+1:end-4, 1, 4+1:end-4, n]
        wet = χt.!=0.
        χt[.!wet] .= NaN
        χ[n] = nansum(
            χt .*
            ΔV,
            dims=(1,2,3)
        )[1,1,1]
    end
    return χ
    
    
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

#function to find ε 

function find_ε(ds)
    x = ds["xC"][4+1:end-4]; Nx = length(x);
    z = ds["zC"][4+1:end-4]; Nz = length(z);
    time = ds["time"][:];
    Δx = reshape(diff(ds["xF"])[4+1:end-4], Nx,1,1);
    Δz = reshape(diff(ds["zF"])[4+1:end-4], 1,1,Nz);
    ΔA = Δx; #flat in y -- 2 dimensional
    ΔV = ΔA.*Δz;
    ε = zeros(size(time,1));
    for n in 1:size(time, 1)
       εt = ds["ε"][4+1:end-4, 1, 4+1:end-4, n]
        wet =εt.!=0.
       εt[.!wet] .= NaN
        ε[n] = nanmean(εt, dims=(1,1))[1,1]
    end
    return ε
    
    
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