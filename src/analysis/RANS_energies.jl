using Oceananigans
using Printf
using NaNStatistics
using NCDatasets

# the goal of this notebook is to write functions that can 
# 1. reynolds average any variable
# 2. find the fluctuating component of any variable

#this will allow us to calculate the mean and fluctuating component of energy fluxes

function ReynoldsAverageCalc(ds, var)
    """
    this function serves to calculate the reynolds average defined using a moving box average
        the inputs are the data file in NCD form and the variable that is being averaged 
        the variable must be inputted in the function as a string, the way that it is denoted in the NCD output file

        ** adjustment made to allow function to accept variable either as a string key or a pre-loaded array

        the moving box average takes the average of the variable over the y dimension and over the plume timescale
        the resulting variable has the size (Nx, Nz, Nt) 
    """
    #import variable and simulation size, time, shape attributes

    time = ds["time"][:]
    #b = ds["b"]
    Nx, Ny, Nz, Nt = ds.attrib["Nx"], ds.attrib["Ny"], ds.attrib["Nz"], length(time)

    #handle both string and array input
    if var isa String
        var_data = ds[var]
    else
        var_data = var # already an array
    end

    #create a wet mask that we can apply to the data (so we take volume average only over the liquid in simulation)

    var_ref = Array(var_data[:, :, :, 2]) #ref buoyancy can be at any time except the initial
    
    # define wet array as bool array : true = fluid, false = hills 
    # size is (Nx, Ny, Nz)
    wet = var_ref .!= 0.0 
    wet_masked = Float64.(copy(wet))
    wet_masked[wet] .= NaN

    @inline function mask_land!(A, wet)
        A[.!wet] .= NaN
        return A
    end

    #definitions for the moving box average

    t_idx = argmin(abs.(time .-18))
    Δt = time[t_idx] - time[1]
    Nt_window = round(Int, Δt)

    println("time at $t_idx is : ", time[t_idx])
    println("length of window : ", Nt_window)

    #pre-allocate output arrays over all time steps
    var_bar = zeros(Float32, Nx, Nz, Nt) 
    var_prime = zeros(Float32, Nx, Ny, Nz, Nt)

    for n in 1:Nt

        #define the time window over which we want to take the time mean
        t_start = max(1, n - Nt_window +1)
        t_inds = t_start:n

        #load variable block over which we take the time mean
        var_block = Array(var_data[1:Nx, 1:Ny, 1:Nz, t_inds])

        #apply wet mask land
        for k in axes(var_block, 4)
            mask_land!(var_block[:, :, :, k], wet)
        end

        #then we take the mean over all y and over the time window Nt_window
        var_bar_n = dropdims(nanmean(nanmean(var_block, dims=2), dims=4); dims=(2,4))
        var_bar[:, :, n] = var_bar_n #store at timestep n

        # we can extract the fluctuating component of variable from it

        varₙ = Array(var_data[1:Nx, 1:Ny, 1:Nz, n])
        mask_land!(varₙ, wet)

        var_prime[:, :, :, n] = varₙ .- reshape(var_bar_n, Nx, 1, Nz)
    end
       
        return var_bar, var_prime
end

# function MKE_TKE_Calc(ds)
#     b = ds["b"]
#     u = ds_v["u"]
#     v = ds_v["v"]
#     w = ds_v["w"]
#     ke = ds
# end

function ϕ_z_calc(ds_b, ds_v)
    """
    this function calculates the mean and turbulent components of the vertical buoyancy flux, ϕz
        the function takes in the buoyancy and velocity datafiles
        and outputs ϕz mean and fluctuating arrays as a function of time
    """

    Nx, Ny, Nz = ds_b.attrib["Nx"], ds_b.attrib["Ny"], ds_b.attrib["Nz"]
    time = ds_b["time"][:]
    Nt = length(time)

    #preallocate arrays
    ϕz_mean = zeros(Nt)
    ϕz_turb = zeros(Nt)

    Δx, Δy, Δz = reshape(ds_b["Δx_caa"][:], Nx, 1, 1), reshape(ds_b["Δy_aca"][:], 1, Ny, 1), reshape(ds_b["Δz_aac"][:], 1, 1, Nz)

    ΔV = Δx .* Δy .* Δz

    #create a wet mask
    b = ds_b["b"]
    b_ref = Array(b[:, :, :, 2])
    wet = b_ref .!= 0.0

    wet_masked = Float32.(copy(wet))
    wet_masked[wet] .= NaN

    wet_vol = nansum(
        wet .* ΔV,
        dims = (1, 2, 3)
    )[1, 1, 1]

    # to calculate the vertical buoyancy flux we need the mean and fluct components of w and b which we can get from calling the RANS function
    b_bar, b_prime = ReynoldsAverageCalc(ds_b, "b")


    # interpolate w to cell centers in z (to match the @ (Center, Center, Center) online calculation)
    w_raw = ds_v["w"][1:Nx, 1:Ny, :, :]
    w_interp = 0.5 .* (w_raw[:, :, 1:end-1, :] .+ w_raw[:, :, 2:end, :])

    w_bar, w_prime = ReynoldsAverageCalc(ds_b, w_interp)

    # also note b_bar and w_bar have size (Nx, Nz, Nt)
    # while b_prime and w_prime have size (Nx, Ny, Nz, Nt)

    # mean flux 
    bw_bar = b_bar .* w_bar                                        # size (Nx, Nz, Nt)
    bw_turb_bar = dropdims(nanmean(b_prime .* w_prime, dims=2), dims=2) # avg over Ny : size (Nx, Nz, Nt)

    for n in 1:Nt
        # bw_bar and bw_turb_bar have already been masked --> this was done in the ReynoldsAverageCalc function
        
        ϕz_mean[n] = nansum(
                bw_bar[:, :, n] .* ΔV[:, 1, :], 
                dims = (1, 2)
        )[1, 1, 1] ./ wet_vol

        ϕz_turb[n] = nansum(
                bw_turb_bar[:, :, n] .* ΔV[:, 1, :],
                dims = (1, 2)
        )[1, 1, 1] ./ wet_vol

    end
    
    return ϕz_mean, ϕz_turb

end