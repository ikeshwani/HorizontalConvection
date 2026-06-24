# kinetic_energetics.jl
#
# Compute volume-averaged kinetic-energy reservoirs ⟨MKE⟩, ⟨TKE⟩, ⟨KE⟩ vs time
# (plus MKE/TKE densities), save to NetCDF, plot, and animate.
#
# Thin script: the MKE/TKE/KE physics lives in calc_kinetic_energies in
# TopographicHorizontalConvection; this file loads data, calls it, saves, plots.
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/kinetic_energetics.jl

using TopographicHorizontalConvection   # physics: calc_kinetic_energies
using NCDatasets
using NaNStatistics
using CairoMakie
using Observables
using Printf

# ---- config ----
run_dir    = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/cheb_8x_stretch/b_base/Ra1e7/512_64/"
plot_dir   = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU_test/energyplots/"
anim_file  = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/animations/GPU/Ra1e7_8xgrid_MKE_TKE.mp4"
mkpath(plot_dir)

ds   = NCDataset(joinpath(run_dir, "buoyancy.nc"),   "r")
ds_v = NCDataset(joinpath(run_dir, "velocities.nc"), "r")
ds_o = NCDataset(joinpath(run_dir, "oceanostics.nc"), "r")

b  = ds["b"]    # DO NOT slice here (lazy)
u  = ds_v["u"]
v  = ds_v["v"]
w  = ds_v["w"]
ke = ds_o["ke"]
time = ds["time"][:]

x = ds["x_caa"][:]
y = ds["y_aca"][:]
z = ds["z_aac"][:]

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

# wet mask (from a non-initial snapshot in case of cold start)
b_ref = Array(b[:, :, :, 2])
wet   = b_ref .!= 0.0           # true = fluid, false = hills
wet_masked = Float64.(copy(wet))
wet_masked[wet] .= NaN          # for plotting hills as a separate color

wet_Vol = nansum(wet .* ΔV, dims=(1,2,3))[1,1,1]

# trailing time-window length (in steps) for the Reynolds mean
Δt = time[90] - time[1]
Nt_window = round(Int, Δt)
println("length of time ", length(time), ", last time: ", time[end])
println("Nt_window = ", Nt_window)

# ---- compute KE reservoirs (physics from src/) ----
MKE_t, TKE_t, KE_t, MKE_xz, TKE_xz =
    calc_kinetic_energies(u, v, w, ke, wet, ΔV, wet_Vol, Nx, Ny, Nz, Nt; Nt_window=Nt_window)

# ---- line plot ----
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

save(joinpath(plot_dir, "Ra1e7_8xgridstretch_KE_plot.png"), fig)
@info "saved Ra1e7 8x grid stretch KE plot"

# ---- save ----
output_file = joinpath(run_dir, "KE.nc")
@info "saving energetics to $output_file"

NCDataset(output_file, "c") do ds_out
    defDim(ds_out, "time", Nt)
    defDim(ds_out, "x", Nx)
    defDim(ds_out, "z", Nz)

    defVar(ds_out, "time", Float64, ("time",))
    defVar(ds_out, "x", Float64, ("x",))
    defVar(ds_out, "z", Float64, ("z",))
    defVar(ds_out, "KE", Float64, ("time",))
    defVar(ds_out, "MKE_Density", Float64, ("x", "z", "time"))
    defVar(ds_out, "TKE_Density", Float64, ("x", "z", "time"))
    defVar(ds_out, "MKE", Float64, ("time",))
    defVar(ds_out, "TKE", Float64, ("time",))

    ds_out["time"][:] = time[1:Nt]
    ds_out["x"][:] = x
    ds_out["z"][:] = z
    ds_out["KE"][:] = KE_t
    ds_out["MKE_Density"][:, :, :] = MKE_xz
    ds_out["TKE_Density"][:, :, :] = TKE_xz
    ds_out["MKE"][:] = MKE_t
    ds_out["TKE"][:] = TKE_t

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
    ds_out["MKE_Density"].attrib["description"] = "MKE(x,z,t) = 0.5 * (ū² + v̄² + w̄²)"

    ds_out["TKE_Density"].attrib["units"] = "m²/s²"
    ds_out["TKE_Density"].attrib["long_name"] = "Turbulent Kinetic Energy Density"
    ds_out["TKE_Density"].attrib["description"] = "TKE(x,z,t) = 0.5 * ((u'²)_bar (v'²)_bar (w'²)_bar)"

    ds_out["MKE"].attrib["units"] = "m²/s²"
    ds_out["MKE"].attrib["long_name"] = "Volume-Averaged Mean Kinetic Energy"
    ds_out["MKE"].attrib["description"] = "⟨MKE⟩ = ∫MKE dV / V_wet"

    ds_out["TKE"].attrib["units"] = "m²/s²"
    ds_out["TKE"].attrib["long_name"] = "Volume-Averaged Turbulent Kinetic Energy"
    ds_out["TKE"].attrib["description"] = "⟨TKE⟩ = ∫TKE dV / V_wet"

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

# ---- animation of ⟨MKE⟩ and ⟨TKE⟩ densities ----
@info "creating animation of ⟨MKE⟩ and ⟨TKE⟩..."
mkpath(dirname(anim_file))

frame = Observable(1)

logMKE = replace(log10.(MKE_xz .+ eps()), NaN => -10.0, Inf => -10.0, -Inf => -10.0)
logTKE = replace(log10.(TKE_xz .+ eps()), NaN => -10.0, Inf => -10.0, -Inf => -10.0)

MKE_obs = @lift logMKE[:, :, $frame]
TKE_obs = @lift logTKE[:, :, $frame]

MKE_lims = (-8, -3)
TKE_lims = (-7, -2)

title_mke = @lift @sprintf("Log 10 of Volume Averaged Mean Kinetic Energy, Ra = %.2e, t = %.2f", Ra, time[$frame])
title_tke = @lift @sprintf("Log 10 of Volume Averaged Turbulent Kinetic Energy, Ra = %.2e, t = %.2f", Ra, time[$frame])

fig2 = Figure(size=(800,1200))
ax1 = Axis(fig2[1,1], xlabel="x", ylabel="z", title=title_mke)
ax2 = Axis(fig2[2,1], xlabel="x", ylabel="z", title=title_tke)

hm1 = heatmap!(ax1, x, z, MKE_obs; colormap=:deep, colorrange=MKE_lims)
heatmap!(ax1, x, z, wet_masked[:,1,:], colormap=:turbid)
Colorbar(fig2[1,2], hm1)

hm2 = heatmap!(ax2, x, z, TKE_obs; colormap=:matter, colorrange=TKE_lims)
heatmap!(ax2, x, z, wet_masked[:,1,:], colormap=:turbid)
Colorbar(fig2[2,2], hm2)

stride = 50
frames = 1:stride:Nt

record(fig2, anim_file, frames; framerate = 8) do i
    frame[] = i
end
@info "saved animation → $anim_file"

close(ds)
close(ds_o)
close(ds_v)
