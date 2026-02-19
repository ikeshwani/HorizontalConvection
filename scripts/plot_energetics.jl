using CairoMakie
using Printf
using Observables
using NaNStatistics
using NCDatasets

#import the KE and PE energetics files that we saved

ds_KE = NCDataset(
    "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/cheb_5x_stretch/b_base/Ra1e7/512_64/KE.nc",
    "r"
)

ds_PE = NCDataset(
    "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/cheb_5x_stretch/b_base/Ra1e7/512_64/PE.nc",
    "r"
)

plot_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU_test/energyplots/"

@info "Loading data from energetics files"

PE = ds_PE["PE"][:]
APE = ds_PE["APE"][:]
BPE = ds_PE["BPE"][:]
KE = ds_KE["KE"][:]
MKE = ds_KE["MKE"][:]
TKE = ds_KE["TKE"][:]

time = ds_KE["time"][:]
Ra = ds_KE.attrib["Ra"]

fig = Figure(size=(800,800))
ax = Axis(fig[1,1], 
            xlabel = "Time (seconds)", 
            ylabel = "⟨E⟩ [m²/s²]", 
            title = @sprintf("Volume-Averaged Energies vs Time for Ra = %.2e", Ra))

lines!(ax, time, MKE .+ TKE, linewidth=2, linestyle=:dash, color=:darkred, label="⟨MKE⟩ + ⟨TKE⟩")
lines!(ax, time, KE, linewidth=2, linestyle=:dash, color=:salmon, label="⟨KE⟩ from Oceanostics Output")
lines!(ax, time, MKE, linewidth=2, color=:purple4, label="⟨MKE⟩")
lines!(ax, time, TKE, linewidth=2, color=:magenta, label="⟨TKE⟩")
lines!(ax, time, PE, linewidth=2, color=:darkgreen, label="⟨PE⟩")
lines!(ax, time, APE, linewidth=2, color=:navy, label="⟨APE⟩")
lines!(ax, time, BPE, linewidth=2, color=:deepskyblue3, label="⟨BPE⟩")

Legend(fig[1,2], ax)

save(joinpath(plot_dir, "Ra1e7_5xgridstretch_energies.png"), fig)
@info " saved Ra1e7 5x grid stretched energy plot"