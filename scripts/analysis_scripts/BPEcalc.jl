# BPEcalc.jl
#
# Compute volume-averaged potential-energy reservoirs ⟨PE⟩, ⟨BPE⟩, ⟨APE⟩ vs time,
# save them to NetCDF, and plot.
#
# Thin script: calc_PE / calc_BPE / calc_APE physics live in
# TopographicHorizontalConvection; this file loads data, calls them, saves, plots.
#
# GRC runs are written in segments (buoyancy_seg<N>.nc), so this loads them in
# order, drops overlapping time steps between consecutive segments, and computes
# the reservoirs one snapshot at a time (memory-light — never holds the whole 4D
# buoyancy field in memory).
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/BPEcalc.jl

using TopographicHorizontalConvection   # physics: calc_PE, calc_BPE, calc_APE
using NCDatasets
using NaNStatistics
using CairoMakie

# ---- config ----
experiment = "control"          # "control" (flat bottom) or "hill" (3-hill GRC)

if experiment == "control"
    data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_flat_baseforcing_zerostart/"
    segments = 1:15
    tag      = "Control"
elseif experiment == "hill"
    data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/"
    segments = 1:22
    tag      = "3hill"
else
    error("unknown experiment: $experiment (use \"control\" or \"hill\")")
end
plot_dir = joinpath(data_dir, "figures")   # figures live inside the run folder
mkpath(plot_dir)

# ---- grid info from seg1 (kept open: calc_BPE reads grid metadata from it) ----
ds_grid = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"), "r")

x = ds_grid["x_caa"][:]
y = ds_grid["y_aca"][:]
z = ds_grid["z_aac"][:]

Ra = ds_grid.attrib["Ra"]
H  = ds_grid.attrib["H"]
Lx = ds_grid.attrib["Lx"]
Ly = ds_grid.attrib["Ly"]

Nx = ds_grid.attrib["Nx"]
Ny = ds_grid.attrib["Ny"]
Nz = ds_grid.attrib["Nz"]

Δx = reshape(ds_grid["Δx_caa"][:], Nx, 1, 1)
Δy = reshape(ds_grid["Δy_aca"][:], 1, Ny, 1)
Δz = reshape(ds_grid["Δz_aac"][:], 1, 1, Nz)
ΔV = Δx .* Δy .* Δz

# 3D z array for PE calculations
z_3d = repeat(reshape(z, 1, 1, Nz), Nx, Ny, 1)

# wet volume (from a non-initial snapshot in case of cold start)
b_ref   = Array(ds_grid["b"][:, :, :, 2])
wet     = b_ref .!= 0.0
wet_Vol = nansum(wet .* ΔV, dims=(1,2,3))[1,1,1]
@info "Grid : Nx=$Nx, Ny=$Ny, Nz=$Nz   Wet Volume: $wet_Vol"

# ---- compute reservoirs over segments, skipping overlapping time steps ----
PE   = Float64[]
BPE  = Float64[]
APE  = Float64[]
time = Float64[]

let t_last = -Inf
    for s in segments
        ds = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"), "r")
        t_seg = ds["time"][:]
        valid = findall(t_seg .> t_last)

        if isempty(valid)
            @info "  seg $s: all $(length(t_seg)) steps are duplicates — skipping"
            close(ds)
            continue
        end
        n_skip = valid[1] - 1
        n_skip > 0 && @info "  seg $s: skipping first $n_skip overlapping step(s)"

        for k in valid[1]:length(t_seg)
            bₙ  = Array(ds["b"][:, :, :, k])
            peₙ = calc_PE(bₙ, z_3d, ΔV, wet_Vol)
            bpeₙ = calc_BPE(ds_grid, bₙ, wet_Vol)   # ds_grid → grid metadata (identical across segments)
            push!(PE,  peₙ)
            push!(BPE, bpeₙ)
            push!(APE, calc_APE(peₙ, bpeₙ))
            push!(time, t_seg[k])
        end

        t_last = t_seg[end]
        close(ds)
        @info "  seg $s: processed through t = $(round(t_last, digits=2))  (total $(length(time)) steps)"
    end
end

Nt = length(time)
@info "⟨PE⟩ range : $(minimum(PE)) to $(maximum(PE))"
@info "⟨BPE⟩ range : $(minimum(BPE)) to $(maximum(BPE))"
@info "⟨APE⟩ range : $(minimum(APE)) to $(maximum(APE))"

# ---- save ----
output_file = joinpath(data_dir, "PE.nc")
@info "saving energetics to $output_file"

NCDataset(output_file, "c") do ds_out
    defDim(ds_out, "time", Nt)

    defVar(ds_out, "time", Float64, ("time",))
    defVar(ds_out, "PE", Float64, ("time",))
    defVar(ds_out, "BPE", Float64, ("time",))
    defVar(ds_out, "APE", Float64, ("time",))

    ds_out["time"][:] = time
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
ax = Axis(fig[1,1], xlabel="time", ylabel="energy [m²/s²]",
          title="$(tag)  Ra = $(Ra) — potential energy reservoirs")
lines!(ax, time, PE,  label="⟨PE⟩",  linewidth=2, color=:orange)
lines!(ax, time, BPE, label="⟨BPE⟩", linewidth=2, color=:blue)
lines!(ax, time, APE, label="⟨APE⟩", linewidth=2, color=:green)
axislegend(ax)
save(joinpath(plot_dir, "PE_$(tag).png"), fig)
@info "saved PE plot → $(joinpath(plot_dir, "PE_$(tag).png"))"

close(ds_grid)
