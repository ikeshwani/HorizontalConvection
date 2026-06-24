# energetics.jl
#
# Energy-reservoir diagnostics for the horizontal convection problem:
#   Potential energy reservoirs (from BPEcalc.jl):
#     - calc_PE  : volume-averaged total potential energy ⟨PE⟩ = -∫ b z dV / V_wet
#     - calc_BPE : volume-averaged background PE (adiabatically sorted state)
#     - calc_APE : available PE = PE - BPE
#   Kinetic energy reservoirs (from kinetic_energetics.jl):
#     - mask_land!            : NaN-out immersed (dry) cells in place
#     - calc_kinetic_energies : volume-averaged ⟨MKE⟩, ⟨TKE⟩, ⟨KE⟩ + xz densities
#
# Physics only — no plotting.
#
# Note: calc_PE / calc_BPE are non-mutating (they copy the buoyancy snapshot
# internally) and treat both b==0 and NaN cells as dry, so the result no longer
# depends on the order in which they are called on the same snapshot.

using NCDatasets
using NaNStatistics
using Interpolations: linear_interpolation, Line   # named import: avoids Interpolations'
                                                   # Periodic/Flat clashing with Oceananigans

export calc_PE, calc_BPE, calc_APE, mask_land!, calc_kinetic_energies

# NaN-out immersed (dry) cells of A in place using the boolean wet mask.
@inline function mask_land!(A, wet)
    A[.!wet] .= NaN
    return A
end

# wet mask from a buoyancy snapshot: dry cells are immersed (b==0) or already NaN.
@inline _wet_mask(b) = (b .!= 0) .& .!isnan.(b)

# Volume-averaged total potential energy:  ⟨PE⟩ = -∫ b z dV / V_wet.
function calc_PE(bₙ, z_3d, ΔV, wet_Vol)
    b = copy(bₙ)
    mask_land!(b, _wet_mask(b))
    return nansum(-b .* z_3d .* ΔV, dims=(1,2,3))[1,1,1] ./ wet_Vol
end

# Volume-averaged background potential energy ⟨BPE⟩: the PE of the buoyancy field
# after adiabatic resorting into a stable stratification.  Each fluid parcel is
# assigned a reference height z⋆ by filling the (wet) domain from the bottom up in
# order of increasing buoyancy; A(z⋆) is the wet horizontal area at each level.
function calc_BPE(ds, bₙ, wet_Vol)
    b = copy(bₙ)
    wet = _wet_mask(b)
    mask_land!(b, wet)

    z = ds["z_aac"][:]

    Nx = ds.attrib["Nx"]
    Ny = ds.attrib["Ny"]
    Nz = ds.attrib["Nz"]

    Δx = reshape(ds["Δx_caa"][:], Nx, 1, 1)
    Δy = reshape(ds["Δy_aca"][:], 1, Ny, 1)
    Δz = reshape(ds["Δz_aac"][:], 1, 1, Nz)

    ΔA = Δx .* Δy
    ΔV = ΔA .* Δz

    b_flat = reshape(b, (Nx * Ny * Nz))

    sort_idx = sortperm(b_flat)
    b_sorted = b_flat[sort_idx]

    ΔV_flat = reshape(ΔV, (Nx * Ny * Nz))
    ΔV_flat = ΔV_flat[sort_idx]

    A_wet = dropdims(sum(wet .* ΔA, dims=(1,2)), dims=(1,2))
    A_interpolation = linear_interpolation(z, A_wet, extrapolation_bc=Line())

    z_bot = ds["z_aaf"][1]

    zstar_flat = zeros(size(b_sorted))
    zstar_flat[1] = z_bot

    for k in 2:length(b_sorted)
        A = A_interpolation(zstar_flat[k-1])
        if !isnan(b_sorted[k])
            zstar_flat[k] = zstar_flat[k-1] + ΔV_flat[k] / A
        else
            zstar_flat[k] = NaN
        end
    end

    unsort_idx = sortperm(sort_idx)
    zstar = reshape(zstar_flat[unsort_idx], size(b))

    return nansum(-b .* zstar .* ΔV, dims=(1,2,3))[1,1,1] ./ wet_Vol
end

# Available potential energy: ⟨APE⟩ = ⟨PE⟩ - ⟨BPE⟩.  Accepts scalars or arrays.
calc_APE(PE, BPE) = PE .- BPE

# Volume-averaged kinetic energy reservoirs over all time steps.
#
# For each step n the velocity field is decomposed into a mean (averaged over y
# and over a trailing time window of Nt_window steps) and a fluctuation:
#   ⟨MKE⟩(t) = vol-avg of 0.5 (ū² + v̄² + w̄²)
#   ⟨TKE⟩(t) = vol-avg of 0.5 ( (u'²)‾ + (v'²)‾ + (w'²)‾ )   (y-averaged squares)
#   ⟨KE⟩(t)  = vol-avg of the Oceanostics `ke` (midplane slice), for comparison.
#
# `u`, `v`, `w` are the (lazy) velocity dataset variables [Nx,Ny,Nz,Nt];
# `ke` is the Oceanostics ke variable [Nx,1,Nz,Nt]; `wet` is the [Nx,Ny,Nz] mask.
# Returns (MKE_t, TKE_t, KE_t, MKE_xz, TKE_xz).
function calc_kinetic_energies(u, v, w, ke, wet, ΔV, wet_Vol, Nx, Ny, Nz, Nt; Nt_window)
    MKE_t = zeros(Nt)
    TKE_t = zeros(Nt)
    KE_t  = zeros(Nt)

    MKE_xz = zeros(Nx, Nz, Nt)
    TKE_xz = zeros(Nx, Nz, Nt)

    for n in 1:Nt
        # Oceanostics KE (midplane slice) for comparison with MKE + TKE
        keₙ = Array(ke[:, 1, :, n])
        mask_land!(keₙ, wet[:, 1, :])

        KE_t[n] = nansum(keₙ .* ΔV[:, 1, :], dims=(1,2))[1,1,1] ./ wet_Vol

        # trailing time window for the time mean
        t_start = max(1, n - Nt_window + 1)
        t_inds  = t_start:n

        u_block = Array(u[1:Nx, 1:Ny, 1:Nz, t_inds])
        v_block = Array(v[1:Nx, 1:Ny, 1:Nz, t_inds])
        w_block = Array(w[1:Nx, 1:Ny, 1:Nz, t_inds])

        for k in axes(u_block, 4)
            mask_land!(u_block[:,:,:,k], wet)
            mask_land!(v_block[:,:,:,k], wet)
            mask_land!(w_block[:,:,:,k], wet)
        end

        # mean over y and over the time window → ū, v̄, w̄ functions of (x, z)
        u_bar = dropdims(nanmean(nanmean(u_block, dims=2), dims=4); dims=(2,4))
        v_bar = dropdims(nanmean(nanmean(v_block, dims=2), dims=4); dims=(2,4))
        w_bar = dropdims(nanmean(nanmean(w_block, dims=2), dims=4); dims=(2,4))

        # MKE density and its volume average
        MKE_xz[:,:,n] .= 0.5 .* (u_bar.^2 .+ v_bar.^2 .+ w_bar.^2)
        MKE_t[n] = nansum(MKE_xz[:, :, n] .* ΔV[:,1,:], dims=(1,2))[1,1,1] ./ wet_Vol

        # instantaneous velocities at time n
        uₙ = Array(u[1:Nx, 1:Ny, 1:Nz, n])
        vₙ = Array(v[1:Nx, 1:Ny, 1:Nz, n])
        wₙ = Array(w[1:Nx, 1:Ny, 1:Nz, n])

        mask_land!(uₙ, wet)
        mask_land!(vₙ, wet)
        mask_land!(wₙ, wet)

        # fluctuations about the mean
        u_prime = uₙ .- reshape(u_bar, Nx, 1, Nz)
        v_prime = vₙ .- reshape(v_bar, Nx, 1, Nz)
        w_prime = wₙ .- reshape(w_bar, Nx, 1, Nz)

        # square then y-average → (u'²)‾ etc.
        u_prime_square_bar = dropdims(nanmean(u_prime.^2, dims=2); dims=2)
        v_prime_square_bar = dropdims(nanmean(v_prime.^2, dims=2); dims=2)
        w_prime_square_bar = dropdims(nanmean(w_prime.^2, dims=2); dims=2)

        TKE_xz[:, :, n] .= 0.5 .* (u_prime_square_bar .+ v_prime_square_bar .+ w_prime_square_bar)
        TKE_t[n] = nansum(TKE_xz[:, :, n] .* ΔV[:,1,:], dims=(1,2))[1,1,1] ./ wet_Vol
    end

    return MKE_t, TKE_t, KE_t, MKE_xz, TKE_xz
end
