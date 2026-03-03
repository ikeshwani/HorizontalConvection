using CairoMakie
using NCDatasets
using Printf
using Oceananigans
using Statistics
using NaNStatistics

# this script is to calculate the relevant energy fluxes in the HC problem

# reversible buoyancy flux :  ϕ_z = ∫ ρgw dV = ∫ - bw dV 

# rate of conversion from internal to potential energy : ϕ_i = -κgA(ρ_top - ρ_bottom)

# rate of change of background potential due to diapycnal mixing : ϕ_d = κg ∫ \frac{dz⋆}{dρ} |∇ρ|² dV

function vertical_b_flux(ds_b)
    Nx = ds_b.attrib["Nx"]
    Ny = ds_b.attrib["Ny"]
    Nz = ds_b.attrib["Nz"]

    Δx = reshape(ds_b["Δx_caa"][:], Nx, 1, 1)
    Δy = reshape(ds_b["Δy_aca"][:], 1, Ny, 1)
    Δz = reshape(ds_b["Δz_acc"][:], 1, 1, Nz)

    ΔA = Δx .* Δy
    ΔV = ΔA .* Δz

    # b = ds_b["b"][:, :, :, :]
    # w = ds_v["w"][:, :, :, :]
    time = ds_b["time"][:]

    ∫ϕ_z = ds_b["∫ϕz"][:, :, :, :]

    ϕ_z = zeros(size(time,1));
    for n in 1:size(time, 1)
        ∫ϕz_t = ∫ϕ_z[:, :, :, n]
        wet = ∫ϕz_t.!=0
        ∫ϕz_t[.!wet] .= NaN
        ϕ_z[n] = nansum(
            ∫ϕz_t .*
            ΔV, 
            dims=(3)
            )[1,1,1]
    end
    
    return ϕ_z
end

#this should be good

#next is ϕ_i = κA (b̄_top - b̄_bottom)

#first we need a function to calculate an average over xy of b(z=top/bottom)

function buoyancy_level_avg(ds, level)
    if level == "bottom"
        b_level = ds["b"][:, :, 1, :] #3d buoyancy array at z=-H
    elseif level == "top"
        b_level = ds["b"][:, :, end, :] #3d buoyancy array at z=0 = surface
    else
        return "invalid level : $level, try top or bottom"
    end

    time = ds["time"][:]
    level_avg = zeros(size(time, 1))
    for n in 1:size(time, 1)
        bb = b_level[:, :, n]
        wet = bb.!=0.
        bb[.!wet] .= NaN

        level_avg[n] = mean(filter(!isnan, bb))
    end
    return level_avg
end


function phi_i(ds_b)
    "this function calculates the flux ϕ_i as a function of time 
    ϕ_i represents the rate of energy supply from heat inputs at the surface of the domain 
    it governs the transfer of internal energy to available potential energy"

    κ = ds_b.attrib["κ"]
    Lx = ds_b.attrib["Lx"]
    Ly = ds_b.attrib["Ly"]
    A = Lx * Ly

    b̄_top = buoyancy_level_avg(ds_b, "top")
    b̄_bottom = buoyancy_level_avg(ds_b, "bottom")

    ϕ_i = κ * A .* (b̄_top .- b̄_bottom)
    
    return ϕ_i
end

ds = NCDataset(
    "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/cheb_5x_stretch/b_base/Ra1e7/512_64/buoyancy.nc",
    "r"
)

ϕ_i_ra7 = phi_i(ds)
ϕ_z_ra7 = vertical_b_flux(ds)
time = ds["time"][:]

fig = Figure()
ax = Axis(fig[1,1], xlabel="time", ylabel="dissipation param", title = "ϕ_z and ϕ_i as function of time for Ra=1e7 5x grid stretch")

scatter!(ax, ϕ_i_ra7, time, label = "ϕ_i")
scatter!(ax, ϕ_z_ra7, time, label = "ϕ_z")

fig[1,2] = axislegend()

plot_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU_test/cheby_grid_verification/"

save(joinpath(plot_dir, "phi_i_z_testra7_5x.png"), fig)

