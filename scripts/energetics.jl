using Oceananigans
using CairoMakie
using Printf
using Observables
using NaNStatistics
using NCDatasets

ds = NCDataset(
    "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/bss_eq_Ra1e8/512_64/buoyancy.nc",
    "r"
)

ds_v = NCDataset(
    "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/bss_eq_Ra1e8/512_64/velocities.nc", 
    "r"
)

ds_o = NCDataset(
    "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/bss_eq_Ra1e8/512_64/oceanostics.nc", 
    "r"
)

plot_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU_test/energyplots/"
mkpath(plot_dir)

b = ds["b"]                # DO NOT slice here
u = ds_v["u"]
v = ds_v["v"]
w = ds_v["w"]
ke = ds_o["ke"]
time = ds["time"][:]

x = ds["x_caa"][:]
y = ds["y_aca"][:]
z = ds["z_aac"][:]

#Metadata

Ra = ds.attrib["Ra"]
H  = ds.attrib["H"]
Lx = ds.attrib["Lx"]
Ly = ds.attrib["Ly"]

Nx = ds.attrib["Nx"]
Ny = ds.attrib["Ny"]
Nz = ds.attrib["Nz"]

Nt = length(time)

Δx = reshape(ds["Δx_caa"][:], Nx, 1, 1)
Δy = reshape(ds["Δy_aca"][:], 1, Ny, 1)
Δz = reshape(ds["Δz_aac"][:], 1, 1, Nz)

ΔV = Δx .* Δy .* Δz

#lets create a wet mask once that we can apply to the data 

b_ref = Array(b[:, :, :, 2]) #im using any time index that is not the initial in case theres no cold start
wet = b_ref .!= 0.0  # bool array : true = fluid, false = hills # size = Nx, Ny, Nz

wet_masked = Float64.(copy(wet))
wet_masked[wet] .= NaN

wet_Vol = nansum(
    wet .* ΔV, 
    dims = (1,2,3)
)[1,1,1]


@inline function mask_land!(A, wet)
    A[.!wet] .= NaN
    return A
end

# in order to calculate the mean velocity we want to average over the y dimension and over the plume timescale

#define MKE and TKE vs time arrays and xz arrays 

MKE_t = zeros(Nt)
TKE_t = zeros(Nt)
KE_t = zeros(Nt)

MKE_xz = zeros(Nx, Nz, Nt)
TKE_xz = zeros(Nx, Nz, Nt)

u_prime = zeros(Float32, Nx, Ny, Nz)
v_prime = zeros(Float32, Nx, Ny, Nz)
w_prime = zeros(Float32, Nx, Ny, Nz)

#definitions for moving box average 

Δt = time[5] - time[1]
Nt_window = round(Int, Δt)

println("time 1 is", time[1])
println("time 5 is", time[5])

println(Nt_window)

#start with doing it over only 50 steps
for n in 1:Nt

    #define the oceanostics KE to compare with MKE + TKE
    keₙ = Array(ke[:, 1, :, n]) #ke only has dimensions Nx, Nz, Nt it is snapshotted at Ny/2
    mask_land!(keₙ, wet[:,1,:])

    KE_t[n] = nansum(
        keₙ .* ΔV[:, 1, :], 
        dims = (1,2)
    )[1,1,1] ./ wet_Vol

    #define time window over which we want to take time mean
    t_start = max(1, n - Nt_window +1)
    t_inds = t_start:n

    #load velocity block over which we take time mean
    u_block = Array(u[1:Nx, 1:Ny, 1:Nz, t_inds])
    v_block = Array(v[1:Nx, 1:Ny, 1:Nz, t_inds])
    w_block = Array(w[1:Nx, 1:Ny, 1:Nz, t_inds])

    #apply wet mask_land
    for k in axes(u_block, 4)
        mask_land!(u_block[:,:,:,k], wet)
        mask_land!(v_block[:,:,:,k], wet)
        mask_land!(w_block[:,:,:,k], wet)
    end

    #take the mean over all y and time Nt_window
    u_bar = dropdims(nanmean(nanmean(u_block, dims=2), dims=4); dims=(2,4))
    v_bar = dropdims(nanmean(nanmean(v_block, dims=2), dims=4); dims=(2,4))
    w_bar = dropdims(nanmean(nanmean(w_block, dims=2), dims=4); dims=(2,4))

    #now bar velocities are functions of x, z, and t

    #we can solve for MKE density :  MKE(x,z,t)
    MKE_xz[:,:,n] .= 0.5 .* (u_bar.^2 .+ v_bar.^2 .+ w_bar.^2)

    #then we take the volume average which gives us MKE(t)
    #volume averaged MKE
    MKE_t[n] = nansum(
        MKE_xz[:, :, n] .* ΔV[:,1,:],
        dims=(1,2)
    )[1,1,1] ./ wet_Vol

    #instantaneous velocities at time n
    
    uₙ = Array(u[1:Nx, 1:Ny, 1:Nz, n])
    vₙ = Array(v[1:Nx, 1:Ny, 1:Nz, n])
    wₙ = Array(w[1:Nx, 1:Ny, 1:Nz, n])

    # apply our wet mask over the hills
    mask_land!(uₙ, wet)
    mask_land!(vₙ, wet)
    mask_land!(wₙ, wet)

    # now for TKE : first we have to define the fluctuating velocities

    u_prime = uₙ .- reshape(u_bar, Nx, 1, Nz)
    v_prime = vₙ .- reshape(v_bar, Nx, 1, Nz)
    w_prime = wₙ .- reshape(w_bar, Nx, 1, Nz)

    #for TKE we have to square the primed velocity THEN take the average in this case the y avg

    u_prime_square_bar = dropdims(nanmean(u_prime.^2, dims=2); dims=2)
    v_prime_square_bar = dropdims(nanmean(v_prime.^2, dims=2); dims=2)
    w_prime_square_bar = dropdims(nanmean(w_prime.^2, dims=2); dims=2)

    TKE_xz[:, :, n] .= 0.5 .* (u_prime_square_bar .+ v_prime_square_bar .+ w_prime_square_bar)

    TKE_t[n] = nansum(
        TKE_xz[:, :, n] .* ΔV[:,1,:], 
        dims= (1,2)
    )[1,1,1] ./  wet_Vol
end


fig = Figure(size=(800,800))
ax = Axis(fig[1,1], 
            xlabel = "Time (seconds)", 
            ylabel = "⟨E⟩ [m²/s²]", 
            title = @sprintf("Volume-Averaged Mean and Turbulent Kinetic Energies versus Time for Ra = %.2e", Ra))

lines!(ax, time[1:Nt], MKE_t, linewidth=2, color=:purple, label="⟨MKE⟩")
lines!(ax, time[1:Nt], TKE_t, linewidth=2, color=:blue, label="⟨TKE⟩")
lines!(ax, time[1:Nt], MKE_t .+ TKE_t, linewidth=2, linestyle=:dash, color=:red, label="⟨MKE⟩ + ⟨TKE⟩")
lines!(ax, time[1:Nt], KE_t, linewidth=2, linestyle=:dash, color=:black, label="⟨KE⟩")
Legend(fig[1,2], ax)

save(joinpath(plot_dir, "Ra1e8_KE_plot_timewindow.png"), fig)
@info " saved Ra1e8_KE_plot"


output_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/bss_eq_Ra1e8/512_64/"
output_file = joinpath(output_dir, "kinetic_energetics_smallerwindow.nc")

@info "saving energetics to $output_file"

NCDataset(output_file, "c") do ds_out
    #define dimensions 
    defDim(ds_out, "time", Nt)
    defDim(ds_out, "x", Nx)
    defDim(ds_out, "z", Nz)

    #define variables
    defVar(ds_out, "time", Float64, ("time",))
    defVar(ds_out, "x", Float64, ("x",))
    defVar(ds_out, "z", Float64, ("z",))
    defVar(ds_out, "KE", Float64, ("time",))
    defVar(ds_out, "MKE_Density", Float64, ("x", "z", "time"))
    defVar(ds_out, "TKE_Density", Float64, ("x", "z", "time"))
    defVar(ds_out, "MKE", Float64, ("time",))
    defVar(ds_out, "TKE", Float64, ("time",))

    #write data
    ds_out["time"][:] = time[1:Nt]
    ds_out["x"][:] = x
    ds_out["z"][:] = z
    ds_out["KE"][:] = KE_t
    ds_out["MKE_Density"][:, :, :] = MKE_xz
    ds_out["TKE_Density"][:, :, :] = TKE_xz
    ds_out["MKE"][:] = MKE_t
    ds_out["TKE"][:] = TKE_t

    #add attributes 
    ds_out["time"].attrib["units"] = "seconds"
    ds_out["time"].attrib["long_name"] = "time"

    ds_out["x"].attrib["units"] = "m"
    ds_out["x"].attrib["long_name"] = "x coordinate"
    
    ds_out["z"].attrib["units"] = "m"
    ds_out["z"].attrib["long_name"] = "z coordinate"

    ds_out["KE"].attrib["units"] = "m²/s²"
    ds_out["KE"].attrib["long_name"] = "Volume-Averaged Total Kinetic Energy From Oceanostics Output"
    ds_out["KE"].attrib["description"] = "⟨KE⟩ = ∫ke dV / V_wet"

    ds_out["MKE_Density"].attrib["units"] = "m²/s²"
    ds_out["MKE_Density"].attrib["long_name"] = "Mean Kinetic Energy Density"
    ds_out["MKE_Density"].attrib["description"] = "MKE(x,z,t) = 0.5 * (ū² + v̄² + w̄²)"
    
    ds_out["TKE_Density"].attrib["units"] = "m²/s²"
    ds_out["TKE_Density"].attrib["long_name"] = "Turbulent Kinetic Energy Density"
    ds_out["TKE_Density"].attrib["description"] = "TKE(x,z,t) = 0.5 * ((u'²)_bar (v'²)_bar (w'²)_bar)"
    
    ds_out["MKE"].attrib["units"] = "m²/s²"
    ds_out["MKE"].attrib["long_name"] = "Volume-Averaged Mean Kinetic Energy"
    ds_out["MKE"].attrib["description"] = "⟨MKE⟩ = ∫MKE dV / V_wet"

    ds_out["TKE"].attrib["units"] = "m²/s²"
    ds_out["TKE"].attrib["long_name"] = "Volume-Averaged Turbulent Kinetic Energy"
    ds_out["TKE"].attrib["description"] = "⟨TKE⟩ = ∫TKE dV / V_wet"

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


@info " creating animation of ⟨MKE⟩ and ⟨TKE⟩..."

fig2 = Figure(size=(800,1200))

frame = Observable(1)

logMKE = replace(log10.(MKE_xz .+ eps()), NaN => -10.0, Inf => -10.0, -Inf => -10.0)
logTKE = replace(log10.(TKE_xz .+ eps()), NaN => -10.0, Inf => -10.0, -Inf => -10.0)

MKE_obs = @lift logMKE[:, :, $frame]
TKE_obs = @lift logTKE[:, :, $frame]

MKE_lims = (-8, -3)
TKE_lims = (-7, -2)

title_mke = @lift @sprintf(
    "Log 10 of Volume Averaged Mean Kinetic Energy, Ra = %.2e, t = %.2f",
    Ra, time[$frame]
)

title_tke = @lift @sprintf(
    "Log 10 of Volume Averaged Turbulent Kinetic Energy, Ra = %.2e, t = %.2f", 
    Ra, time[$frame]
)

ax1 = Axis(fig2[1,1], 
            xlabel = "x", 
            ylabel = "z", 
            title = title_mke)

ax2 = Axis(
    fig2[2,1], 
    xlabel="x", 
    ylabel = "z", 
    title = title_tke
)

hm1 = heatmap!(
    ax1, 
    x, z, MKE_obs;
    colormap=:deep, 
    colorrange = MKE_lims
)

hm_hill = heatmap!(
    ax1, 
    x, z, wet_masked[:,1,:], 
    colormap=:turbid
)

Colorbar(fig2[1,2], hm1)

hm2 = heatmap!(
    ax2, 
    x, z, TKE_obs;
    colormap=:matter,
    colorrange = TKE_lims
)

hm_hill = heatmap!(
    ax2, 
    x, z, wet_masked[:,1,:], 
    colormap=:turbid
)

Colorbar(fig2[2,2], hm2)

stride = 50
frames = 1:stride:Nt   # or 1:5:Nt to subsample

output_file = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/animations/GPU/Ra1e8_energetics_smallerwindow.mp4"

record(fig2, output_file, frames; framerate = 8) do i
    frame[] = i
end

close(ds)
close(ds_o)
close(ds_v)