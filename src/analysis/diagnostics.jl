# diagnostics.jl
#
# Volume integral / volume average diagnostics for 2D NetCDF post-processing,
# plus the dissipation constraint analysis (Paparella & Young 2002 for ε,
# Winters & Young 2009 for χ).
#
# Physics only — no plotting. Moved out of the former src/analysis.jl.

using NCDatasets
using NaNStatistics

export global_volume_integral, global_volume_avg, buoyancy_bottom_avg, dissipation_analysis

#global volume integral function to generalize find χ and find ϵ
function global_volume_integral(ds, var)
    x, z = ds["x_caa"][:] , ds["z_aac"][:]
    Nx, Nz = ds.attrib["Nx"] , ds.attrib["Nz"]
    time = ds["time"][:]
    Δx = reshape(ds["Δx_caa"][:], Nx,1,1)
    Δz = reshape(ds["Δz_aac"][:], 1,1,Nz)
    ΔA = Δx #flat in y -- 2 dimensional
    ΔV = ΔA.*Δz
    var_array = zeros(size(time,1))
    for n in 1:size(time, 1)
        var_t = ds[var][:, :, n] #2D only x and z dimensions
        wet = var_t.!=0.
        var_t[.!wet] .= NaN
        var_array[n] = nansum(
            var_t .*
            ΔV,
            dims=(1,2)
        )[1,1,1]
    end
    return var_array
end

#function to find the volume averaged variable (generalized for χ or ε)
function global_volume_avg(ds, var)
    Nx, Nz = ds.attrib["Nx"], ds.attrib["Nz"]
    x, z = ds["x_caa"][:], ds["z_aac"][:]
    time = ds["time"][:]
    Δx = reshape(ds["Δx_caa"][:], Nx,1,1);
    Δz = reshape(ds["Δz_aac"][:], 1,1,Nz);
    ΔA = Δx; #flat in y -- 2 dimensional
    ΔV = ΔA.*Δz;
    var_array = zeros(size(time,1));
    for n in 1:size(time, 1)
        var_t = ds[var][:, :, n]
        wet = ds[var][:, :, 3].!=0.
        var_t[.!wet] .= NaN

         #volume integral over "wet" variable
        integral = nansum(
            var_t .*
            ΔV,
            dims=(1,2)
        )[1,1,1]

        #volume of wet ocean
        wet_volume = nansum(
            wet .*
            ΔV,
            dims=(1,2)
        )[1,1,1]

        #now we find the volume averaged variable
        var_array[n] = integral ./ wet_volume
    end
    return var_array
end

#function to find the bottom buoyancy average
function buoyancy_bottom_avg(ds)
    b_bottom = ds["b"][:,1,:]
    bottom_avg = zeros(size(b_bottom[1,:]))
    for n in 1:size(b_bottom,2)
        bb = b_bottom[:,n]
        wet = bb.!=0.
        bb[.!wet] .= NaN

        bottom_avg[n] = nanmean(bb, dims=1)[1,1]
    end
    return bottom_avg
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
