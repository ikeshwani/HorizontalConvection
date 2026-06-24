# BPEcalc.jl
#
# Compute volume-averaged potential-energy reservoirs ⟨PE⟩, ⟨BPE⟩, ⟨APE⟩ vs time,
# save them to NetCDF, and plot.
#
# Thin script: calc_PE / calc_BPE / calc_APE physics live in
# TopographicHorizontalConvection; this file loads data, calls them, saves, plots.
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/BPEcalc.jl

using TopographicHorizontalConvection   # physics: calc_PE, calc_BPE, calc_APE
using NCDatasets
using NaNStatistics
using CairoMakie

# ---- config ----
data_file  = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/cheb_5x_stretch/b_base/Ra1e7/512_64/buoyancy.nc"
output_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/cheb_5x_stretch/b_base/Ra1e7/512_64/"
plot_dir   = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU_test/energyplots/"
mkpath(plot_dir)

ds = NCDataset(data_file, "r")

b = ds["b"]   # lazy load — slicing the whole thing would blow up memory

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

ΔV = Δx .* Δy .* Δz

# 3D z array for PE calculations
z_3d = repeat(reshape(z, 1, 1, Nz), Nx, Ny, 1)

# wet volume (computed once from a non-initial snapshot in case of cold start)
b_ref = Array(b[:, :, :, 2])
wet   = b_ref .!= 0.0
wet_Vol = nansum(wet .* ΔV, dims=(1,2,3))[1,1,1]
@info "Wet Volume: $wet_Vol"

# ---- compute reservoirs (physics from src/) ----
PE  = zeros(Nt)
BPE = zeros(Nt)
APE = zeros(Nt)

for n in 1:Nt
    bₙ = Array(b[:, :, :, n])
    PE[n]  = calc_PE(bₙ, z_3d, ΔV, wet_Vol)
    BPE[n] = calc_BPE(ds, bₙ, wet_Vol)
    APE[n] = calc_APE(PE[n], BPE[n])

    n % 10 == 0 && @info "Processed $n/$Nt timesteps"
end

@info "⟨PE⟩ range : $(minimum(PE)) to $(maximum(PE))"
@info "⟨BPE⟩ range : $(minimum(BPE)) to $(maximum(BPE))"
@info "⟨APE⟩ range : $(minimum(APE)) to $(maximum(APE))"

# ---- save ----
output_file = joinpath(output_dir, "PE.nc")
@info "saving energetics to $output_file"

NCDataset(output_file, "c") do ds_out
    defDim(ds_out, "time", Nt)

    defVar(ds_out, "time", Float64, ("time",))
    defVar(ds_out, "PE", Float64, ("time",))
    defVar(ds_out, "BPE", Float64, ("time",))
    defVar(ds_out, "APE", Float64, ("time",))

    ds_out["time"][:] = time[1:Nt]
    ds_out["PE"][:] = PE
    ds_out["BPE"][:] = BPE
    ds_out["APE"][:] = APE

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

# ---- plot ----
fig = Figure(size=(600,600))
ax = Axis(fig[1,1], xlabel="time", ylabel="energy [m²/s²]")
lines!(ax, time[1:Nt], PE,  label="⟨PE⟩",  linewidth=2, color=:orange)
lines!(ax, time[1:Nt], BPE, label="⟨BPE⟩", linewidth=2, color=:blue)
lines!(ax, time[1:Nt], APE, label="⟨APE⟩", linewidth=2, color=:green)
axislegend(ax)
save(joinpath(plot_dir, "Ra1e7_5xgrid_PE.png"), fig)
@info "saved Ra1e7 5x grid stretch PE plot"

close(ds)
