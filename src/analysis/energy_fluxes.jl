# energy_fluxes.jl
#
# Energy-flux diagnostics for the horizontal convection problem:
#   - vertical_b_flux  : reversible buoyancy flux ϕ_z = ∫ -bw dV (from stored ∫ϕz)
#   - buoyancy_level_avg: horizontal average of b at the top or bottom level
#   - phi_i            : rate of energy supply ϕ_i = κ A (b̄_top - b̄_bottom)
#
# Physics only — no plotting.

using NCDatasets
using NaNStatistics

export vertical_b_flux, buoyancy_level_avg, phi_i

# reversible buoyancy flux  ϕ_z = ∫ ρgw dV = ∫ -bw dV
# (the integrand ∫ϕz is written out by the simulation; here we volume-integrate it)
function vertical_b_flux(ds_b)
    Nx = ds_b.attrib["Nx"]
    Ny = ds_b.attrib["Ny"]
    Nz = ds_b.attrib["Nz"]

    Δx = reshape(ds_b["Δx_caa"][:], Nx, 1, 1)
    Δy = reshape(ds_b["Δy_aca"][:], 1, Ny, 1)
    Δz = reshape(ds_b["Δz_aac"][:], 1, 1, Nz)

    ΔA = Δx .* Δy
    ΔV = ΔA .* Δz

    time = ds_b["time"][:]

    ∫ϕ_z = ds_b["∫ϕz"][:, :, :, :]

    ϕ_z = zeros(size(time, 1))
    for n in 1:size(time, 1)
        ∫ϕz_t = ∫ϕ_z[:, :, :, n]
        wet = ∫ϕz_t .!= 0
        ∫ϕz_t[.!wet] .= NaN
        ϕ_z[n] = nansum(
            ∫ϕz_t .*
            ΔV,
            dims=(3)
        )[1, 1, 1]
    end

    return ϕ_z
end

# horizontal (xy) average of buoyancy at the top (z=0) or bottom (z=-H) level
function buoyancy_level_avg(ds, level)
    if level == "bottom"
        b_level = ds["b"][:, :, 1, :]   # 3d buoyancy array at z=-H
    elseif level == "top"
        b_level = ds["b"][:, :, end, :] # 3d buoyancy array at z=0 = surface
    else
        return "invalid level : $level, try top or bottom"
    end

    time = ds["time"][:]
    level_avg = zeros(size(time, 1))
    for n in 1:size(time, 1)
        bb = b_level[:, :, n]
        wet = bb .!= 0.
        bb[.!wet] .= NaN

        level_avg[n] = nanmean(bb)
    end
    return level_avg
end

# ϕ_i = κ A (b̄_top - b̄_bottom):
# rate of energy supply from the surface buoyancy forcing, governing the transfer
# of internal energy to available potential energy.
function phi_i(ds_b)
    κ = ds_b.attrib["κ"]
    Lx = ds_b.attrib["Lx"]
    Ly = ds_b.attrib["Ly"]
    A = Lx * Ly

    b̄_top = buoyancy_level_avg(ds_b, "top")
    b̄_bottom = buoyancy_level_avg(ds_b, "bottom")

    ϕ_i = κ * A .* (b̄_top .- b̄_bottom)

    return ϕ_i
end
