using Printf
using NCDatasets
using CairoMakie
using Observables

basepath = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/Control/RA1e8/4x_stretch/512_128/"
output_file = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/animations/GPU/GRC/Control/first167sec.mp4"

mkpath(dirname(output_file))

datasets = [NCDataset(joinpath(basepath, "buoyancy_seg$(i).nc")) for i in 1:8]

ds1 = datasets[1]
Nx = ds1.attrib["Nx"]
Ny = ds1.attrib["Ny"]
Nz = ds1.attrib["Nz"]
Lx = ds1.attrib["Lx"]
Ly = ds1.attrib["Ly"]
H  = ds1.attrib["H"]
Ra = ds1.attrib["Ra"]
x  = ds1["x_caa"][:]
y  = ds1["y_aca"][:]
z  = ds1["z_aac"][:]

# Build global frame index removing overlapping timesteps between segments
frame_index = Tuple{Int,Int}[]
t_global    = Float64[]

let t_end_prev = -Inf
    for seg in 1:8
        t_seg     = datasets[seg]["time"][:]
        new_start = findfirst(t -> t > t_end_prev, t_seg)
        new_start === nothing && continue
        for k in new_start:length(t_seg)
            push!(frame_index, (seg, k))
            push!(t_global, t_seg[k])
        end
        t_end_prev = t_global[end]
    end
end

Nt = length(t_global)
@info "Total frames: $Nt,  t ∈ [$(t_global[1]), $(t_global[end])]"

yidx = Ny ÷ 2   # mid-y slice for x-z plane
xidx = 1         # leftmost x slice for y-z plane (cold plume entry)

# Wet mask from seg1 time index 2 — avoids cold-start zeros misidentifying dry cells
b_ref_xz = Array(datasets[1]["b"][:, yidx, :, 2])
b_ref_yz = Array(datasets[1]["b"][xidx, :, :, 2])

hill_xz        = fill(NaN, Nx, Nz)
hill_xz[b_ref_xz .== 0.0] .= 1.0

hill_yz        = fill(NaN, Ny, Nz)
hill_yz[b_ref_yz .== 0.0] .= 1.0

n = Observable(1)

title_xz = @lift @sprintf("buoyancy x-z plane  Ra = %.2e  t = %.2f", Ra, t_global[$n])
title_yz = @lift @sprintf("buoyancy y-z plane  Ra = %.2e  t = %.2f", Ra, t_global[$n])

b_xzₙ = @lift begin
    seg, k = frame_index[$n]
    Array(datasets[seg]["b"][:, yidx, :, k])
end

b_yzₙ = @lift begin
    seg, k = frame_index[$n]
    Array(datasets[seg]["b"][xidx, :, :, k])
end

B_lims = (-0.5, 0.5)

fig = Figure(size = (800, 1200))

ax_xz = Axis(fig[1, 1];
    title     = title_xz,
    xlabel    = L"x",
    ylabel    = L"z",
    limits    = ((-Lx/2, Lx/2), (-H, 0)),
    aspect    = Lx / H,
    titlesize = 20
)

ax_yz = Axis(fig[2, 1];
    title     = title_yz,
    xlabel    = L"y",
    ylabel    = L"z",
    limits    = ((-Ly/2, Ly/2), (-H, 0)),
    aspect    = Ly / H,
    titlesize = 20
)

hm1 = heatmap!(ax_xz, x, z, b_xzₙ; colormap = :balance, colorrange = B_lims)
heatmap!(ax_xz, x, z, hill_xz; colormap = :turbid)
Colorbar(fig[1, 2], hm1)

hm2 = heatmap!(ax_yz, y, z, b_yzₙ; colormap = :balance, colorrange = B_lims)
heatmap!(ax_yz, y, z, hill_yz; colormap = :turbid)
Colorbar(fig[2, 2], hm2)

record(fig, output_file, 1:Nt; framerate = 24) do i
    n[] = i
end

foreach(close, datasets)
@info "Animation saved to $output_file"
