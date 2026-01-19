using CairoMakie
using NaNStatistics
using NCDatasets
using Observables
using Printf
using Interpolations

ds = NCDataset(
    "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/bss_eq_Ra1e8/512_64/buoyancy.nc",
    "r"
)

b = ds["b"] #lazy load or else sheesh will crash

x = ds["x_caa"][:]
y = ds["y_aca"][:]
z = ds["z_aac"][:]
time = ds["time"][:]

Ra = ds.attrib["Ra"]
H  = ds.attrib["H"]
Lx = ds.attrib["Lx"]
Ly = ds.attrib["Ly"]

Nx = ds.attrib["Nx"]
Ny = ds.attrib["Ny"]
Nz = ds.attrib["Nz"]
Nt = length(time)

println("Grid : Nx=$Nx, Ny=$Ny, Nz=$Nz, Nt=$Nt")

Δx = reshape(ds["Δx_caa"][:], Nx, 1, 1)
Δy = reshape(ds["Δy_aca"][:], 1, Ny, 1)
Δz = reshape(ds["Δz_aac"][:], 1, 1, Nz)

ΔA = Δx .* Δy
ΔV = ΔA .* Δz

#create the 3D z array for PE calculations 
z_3d = repeat(reshape(z, 1, 1, Nz), Nx, Ny, 1)

#calculate the wet volume once 
b_ref = Array(b[:, :, :, 2])
wet = b_ref .!= 0.0 #bool array --> true = wet, false = hills

wet_Vol = nansum(wet .* ΔV, dims=(1,2,3))[1,1,1]

@info "Wet Volume: $wet_Vol"

function calc_PE(bₙ, z_3d, ΔV, wet_Vol)
    """
    calculate the total volume averaged ⟨PE⟩ = - ∫ b * z dV / V_wet
    """

    wet = bₙ .!=0
    bₙ[.!wet] .= NaN
    
    PE = nansum(-bₙ .* z_3d .* ΔV, dims=(1,2,3))[1,1,1] ./ wet_Vol

    return PE
end

function calc_BPE(ds, bₙ, wet_Vol)
    """
    calculate the volume-averaged background potential energy
    """

    #we load in bₙ which is a 3D array at one time step (Nx, Ny, NZ)

    wet = bₙ .!= 0
    bₙ[.!wet] .= NaN

    z = ds["z_aac"][:]

    Nx = ds.attrib["Nx"]
    Ny = ds.attrib["Ny"]
    Nz = ds.attrib["Nz"]

    Δx = reshape(ds["Δx_caa"][:], Nx, 1, 1)
    Δy = reshape(ds["Δy_aca"][:], 1, Ny, 1)
    Δz = reshape(ds["Δz_aac"][:], 1, 1, Nz)

    ΔA = Δx .* Δy
    ΔV = ΔA .* Δz

    z_3d = repeat(reshape(z, 1, 1, Nz), Nx, Ny, 1)

    b_flat = reshape(bₙ, (Nx * Ny * Nz))
    wet_flat = reshape(wet, (Nx * Ny * Nz))
    z_flat = reshape(z_3d, (Nx * Ny * Nz))

    sort_idx = sortperm(b_flat)
    b_sorted = b_flat[sort_idx]
    z_sorted = z_flat[sort_idx]
    
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
    zstar = reshape(zstar_flat[unsort_idx], size(bₙ))

    BPE = nansum(-bₙ .* zstar .* ΔV, dims=(1,2,3))[1,1,1] ./ wet_Vol

    return BPE
end

#now we want to calculate BPE for all time steps 
#lets start with just 50
Nt_test = 50

PE = zeros(Nt)
BPE = zeros(Nt)
APE = zeros(Nt)

for n in 1:Nt
    bₙ = Array(b[:, :, :, n])
    PE[n] = calc_PE(bₙ, z_3d, ΔV, wet_Vol)
    BPE[n] = calc_BPE(ds, bₙ, wet_Vol)
    APE[n] = PE[n] - BPE[n]

    if n % 10 == 0
        @info "Processed $n/Nt timesteps"
    end
end

println("size of BPE is: ", size(BPE))

@info "⟨PE⟩ range : $(minimum(PE)) to $(maximum(PE))"
@info "⟨BPE⟩ range : $(minimum(BPE)) to $(maximum(BPE))"
@info "⟨APE⟩ range : $(minimum(APE)) to $(maximum(APE))"

#save data to netcdf so I can plot with the KE stuff

output_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/bss_eq_Ra1e8/512_64/"
output_file = joinpath(output_dir, "energetics.nc")

@info "saving energetics to $output_file"

NCDataset(output_file, "c") do ds_out
    #define dimensions 
    defDim(ds_out, "time", Nt)

    #define variables
    defVar(ds_out, "time", Float64, ("time",))
    defVar(ds_out, "PE", Float64, ("time",))
    defVar(ds_out, "BPE", Float64, ("time",))
    defVar(ds_out, "APE", Float64, ("time",))

    #write data
    ds_out["time"][:] = time[1:Nt]
    ds_out["PE"][:] = PE
    ds_out["BPE"][:] = BPE
    ds_out["APE"][:] = APE

    #add attributes 
    ds_out["time"].attrib["units"] = "seconds"
    ds_out["time"].attrib["long_name"] = "time"

    ds_out["PE"].attrib["units"] = "m²/s²"
    ds_out["PE"].attrib["long_name"] = "Volume-Averaged Total Potential Energy"
    ds_out["PE"].attrib["description"] = "⟨PE⟩ = - ∫b * z dV / V_wet"

    ds_out["BPE"].attrib["units"] = "m²/s²"
    ds_out["BPE"].attrib["long_name"] = "Volume-Averaged Background Potential Energy"
    ds_out["BPE"].attrib["description"] = "⟨BPE⟩ = PE of adiabatically sorted buoyancy field / V_wet"
    
    ds_out["APE"].attrib["units"] = "m²/s²"
    ds_out["APE"].attrib["long_name"] = "Volume-Averaged Available Potential Energy"
    ds_out["APE"].attrib["description"] = "⟨APE⟩ = ⟨PE⟩ - ⟨BPE⟩"

    #copy simulation metadata

    ds_out.attrib["Ra"] = Ra
    ds_out.attrib["H"] = H
    ds_out.attrib["Lx"] = Lx
    ds_out.attrib["Ly"] = Ly
    ds_out.attrib["Nx"] = Nx
    ds_out.attrib["Ny"] = Ny
    ds_out.attrib["Nz"] = Nz
    ds_out.attrib["wet_Vol"] = wet_Vol
end

@info "Energetics Saved Successfully"

plot_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU_test/energyplots/"

fig = Figure(size=(600,600))
ax = Axis(fig[1,1], xlabel="time", ylabel="BPE")

lines!(ax, time[1:Nt], PE, label="⟨PE⟩", linewidth=2, color=:orange)
lines!(ax, time[1:Nt], BPE, label="⟨BPE⟩", linewidth=2, color=:blue)
lines!(ax, time[1:Nt], APE, label="⟨APE⟩", linewidth=2, color=:green)


save(joinpath(plot_dir, "Ra1e8_PE_APE_BPE.png"), fig)
@info " saved Ra1e8_BPE_plot"

close(ds)