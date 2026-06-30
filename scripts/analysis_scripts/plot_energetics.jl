# plot_energetics.jl
#
# Overlay the potential- and kinetic-energy reservoirs saved by BPEcalc.jl
# (PE.nc) and kinetic_energetics.jl (KE.nc) for one GRC run.
#
# NOTE: PE.nc and KE.nc must share the same time axis — all series are plotted
# against the KE.nc time vector. Run BPEcalc.jl and kinetic_energetics.jl on the
# same `experiment` first so both files exist in the run folder.
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/plot_energetics.jl

using CairoMakie
using Printf
using Observables
using NaNStatistics
using NCDatasets

# ---- config ----
experiment = "control"          # "control" (flat bottom) or "hill" (3-hill GRC)

if experiment == "control"
    data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_flat_baseforcing_zerostart/"
    tag      = "Control"
elseif experiment == "hill"
    data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/"
    tag      = "3hill"
else
    error("unknown experiment: $experiment (use \"control\" or \"hill\")")
end
plot_dir = joinpath(data_dir, "figures")   # figures live inside the run folder
mkpath(plot_dir)

# import the KE and PE energetics files that we saved
ds_KE = NCDataset(joinpath(data_dir, "KE.nc"), "r")
ds_PE = NCDataset(joinpath(data_dir, "PE.nc"), "r")

@info "Loading data from energetics files"

PE  = ds_PE["PE"][:]
APE = ds_PE["APE"][:]
BPE = ds_PE["BPE"][:]
KE  = ds_KE["KE"][:]
MKE = ds_KE["MKE"][:]
TKE = ds_KE["TKE"][:]

time = ds_KE["time"][:]
Ra = ds_KE.attrib["Ra"]

fig = Figure(size=(800,800))
ax = Axis(fig[1,1],
            xlabel = "Time (seconds)",
            ylabel = "⟨E⟩ [m²/s²]",
            title = @sprintf("%s — Volume-Averaged Energies vs Time, Ra = %.2e", tag, Ra))

lines!(ax, time, MKE .+ TKE, linewidth=2, linestyle=:dash, color=:darkred, label="⟨MKE⟩ + ⟨TKE⟩")
lines!(ax, time, KE, linewidth=2, linestyle=:dash, color=:salmon, label="⟨KE⟩ from Oceanostics Output")
lines!(ax, time, MKE, linewidth=2, color=:purple4, label="⟨MKE⟩")
lines!(ax, time, TKE, linewidth=2, color=:magenta, label="⟨TKE⟩")
lines!(ax, time, PE, linewidth=2, color=:darkgreen, label="⟨PE⟩")
lines!(ax, time, APE, linewidth=2, color=:navy, label="⟨APE⟩")
lines!(ax, time, BPE, linewidth=2, color=:deepskyblue3, label="⟨BPE⟩")

Legend(fig[1,2], ax)

save(joinpath(plot_dir, "energies_$(tag).png"), fig)
@info "saved energy plot → $(joinpath(plot_dir, "energies_$(tag).png"))"

close(ds_KE)
close(ds_PE)
