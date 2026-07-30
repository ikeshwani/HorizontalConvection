using Printf
using NCDatasets
using CairoMakie
using Observables
using TOML

"""
    animate_3d_buoyancy(run_dir; kwargs...)

Build a genuinely 3D buoyancy animation for one self-contained simulation run
folder produced by the new output layout, e.g.
`output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/`.

# Rendering design (see report / commit message for the full rationale)
CairoMakie (this project's CPU/software Makie backend) does **not** support
Makie's 3D `contour`/`volume` isosurface plots — they silently render nothing
(`Volume{...} is not supported by cairo right now`). So instead of true
isosurfaces this draws:

  - the seafloor as a solid 3D surface mesh built directly from each run's
    `bottom_height(x,y)` variable (the true, y-dependent hill geometry,
    channel notch included — not an analytic reconstruction), and
  - the buoyancy field as a small stack of colored cross-sectional planes at
    `n_slices` evenly spaced y-positions (a "slice-stack" / light-sheet style
    3D rendering), each masked transparent over dry (immersed) cells so the
    topography shows through.

This is fully renderable by CairoMakie (it's just colored quad meshes, no
volumetric texture), shows real x-z structure at several depths across y
(revealing the y-dependent channel notch), and is dramatically cheaper than
true marching-cubes isosurfaces would be.

Per-frame cost is dominated by CairoMakie's rasterization of the slice
surfaces, not by data loading, so both spatial downsampling (`xstride`,
`zstride`) and the number of slices (`n_slices`) directly trade visual detail
for render time. Frames are recreated via `delete!`/`surface!` each callback
rather than mutating an `Observable`-linked `color` attribute in place —
empirically ~3x faster here (~2.5-3s/frame vs ~9.5s/frame at matched
settings), apparently because CairoMakie still re-rasterizes every primitive
every frame regardless of the update mechanism, and Observable-driven color
updates carry extra overhead on top of that in this Makie version.

As with the 2D script, only tiny per-frame slices are ever pulled from the
open `NCDataset`s — no segment's full `b` array is loaded into memory.

# Arguments
- `run_dir::AbstractString`: path to the self-contained run folder.

# Keyword arguments
- `output_dir = joinpath(run_dir, "figures")`: where to write the `.mp4`.
- `filename   = "buoyancy_3d_animation.mp4"`: output file name.
- `nsegments  = nothing`: number of leading segments to use; `nothing` = all
  (after the active-segment guard below). Setting this disables the guard.
- `drop_active = true`: drop the highest-numbered segment if its file was
  modified within `active_window` seconds (still being written). Ignored
  when `nsegments` is given explicitly.
- `active_window = 900`: seconds; see `drop_active`.
- `t_max      = nothing`: stop the animation at this nondimensional time.
- `frame_stride = 6`: keep only every `frame_stride`-th deduplicated
  timestep. At full cadence (0.1 time units, ~8700 raw frames across
  segments 1:31) a true 3D render is far too slow to be worth the extra
  frames; subsampling keeps total render time and movie length reasonable.
  Measured render cost is ~3.2 s/frame (see report), so stride=6 over the
  ~8748 deduplicated frames in segments 1:31 gives 1458 frames (~61 s of
  movie at 24 fps) in roughly 75-80 minutes — run via `submit_animate_3d.sh`
  on the `cpu` partition, not interactively on a login node.
- `B_lims     = (-0.5, 0.5)`: buoyancy colour limits (matches the 2D script).
- `framerate  = 24`: frames per second.
- `n_slices   = 3`: number of colored y-cross-sections of the buoyancy field.
- `xstride, zstride = 2, 2`: spatial downsampling of the buoyancy slices
  (and of `bottom_height` in x) for rendering speed. y is never downsampled
  (`Ny` is already only 32).
- `figsize    = (1000, 650)`: figure size in pixels.
- `elevation, azimuth = 0.32, -0.75`: `Axis3` camera angles (radians).
- `aspect     = (4, 1, 1.4)`: `Axis3` relative aspect. This is a *display*
  aspect, not the true physical one (`Lx:Ly:H = 16:1:4`) — Ly is exaggerated
  roughly 4x here purely so the channel/y-structure is visible at all; the
  real domain is very thin in y (`Ly = 0.25` vs `Lx = 4`).
- `numhill    = nothing`: currently unused (topography is always drawn from
  `bottom_height`), kept for interface symmetry with `make_animation`.

Returns the path to the written animation.
"""
function animate_3d_buoyancy(run_dir::AbstractString;
                        output_dir    = joinpath(run_dir, "figures"),
                        filename      = "buoyancy_3d_animation.mp4",
                        nsegments     = nothing,
                        drop_active   = true,
                        active_window = 900,
                        t_max         = nothing,
                        frame_stride  = 6,
                        B_lims        = (-0.5, 0.5),
                        framerate     = 24,
                        n_slices      = 3,
                        xstride       = 2,
                        zstride       = 2,
                        figsize       = (1000, 650),
                        elevation     = 0.32,
                        azimuth       = -0.75,
                        aspect        = (4, 1, 1.4),
                        numhill       = nothing)

    isdir(run_dir) || error("run_dir does not exist: $run_dir")

    # --- discover and naturally sort buoyancy segment files ---
    seg_re   = r"^buoyancy_seg(\d+)\.nc$"
    seg_nums = sort([parse(Int, match(seg_re, f).captures[1])
                     for f in readdir(run_dir) if occursin(seg_re, f)])
    isempty(seg_nums) && error("no buoyancy_seg*.nc files found in $run_dir")

    if nsegments !== nothing
        seg_nums = seg_nums[1:min(nsegments, end)]
    elseif drop_active && length(seg_nums) > 1
        last_file = joinpath(run_dir, "buoyancy_seg$(seg_nums[end]).nc")
        age = time() - mtime(last_file)
        if age < active_window
            @warn "Dropping seg$(seg_nums[end]): modified $(round(Int, age))s ago (< $(active_window)s) — looks like an active run. Pass nsegments to override."
            seg_nums = seg_nums[1:end-1]
        end
    end

    output_file = joinpath(output_dir, filename)
    mkpath(dirname(output_file))

    datasets = [NCDataset(joinpath(run_dir, "buoyancy_seg$(s).nc")) for s in seg_nums]

    try
        ds1 = datasets[1]
        Nx = ds1.attrib["Nx"]
        Ny = ds1.attrib["Ny"]
        Nz = ds1.attrib["Nz"]
        Lx = ds1.attrib["Lx"]
        Ly = ds1.attrib["Ly"]
        H  = ds1.attrib["H"]
        Ra = ds1.attrib["Ra"]

        x_full = Float64.(ds1["x_caa"][:])
        y      = Float64.(ds1["y_aca"][:])
        z_full = Float64.(ds1["z_aac"][:])

        xi = 1:xstride:Nx
        zi = 1:zstride:Nz
        x  = x_full[xi]
        z  = z_full[zi]
        Nxs, Nzs = length(x), length(z)

        bottom = Float64.(ds1["bottom_height"][xi, :])

        # --- global frame index, dropping overlapping timesteps between segments ---
        frame_index = Tuple{Int,Int}[]
        t_global    = Float64[]
        let t_end_prev = -Inf
            for seg in eachindex(datasets)
                t_seg = datasets[seg]["time"][:]
                isempty(t_seg) && continue   # e.g. a zero-length segment file
                new_start = findfirst(t -> t > t_end_prev, t_seg)
                new_start === nothing && continue
                for k in new_start:length(t_seg)
                    push!(frame_index, (seg, k))
                    push!(t_global, t_seg[k])
                end
                t_end_prev = t_global[end]
            end
        end

        if t_max !== nothing
            keep = findall(t -> t <= t_max, t_global)
            frame_index = frame_index[keep]
            t_global    = t_global[keep]
        end

        # --- subsample ---
        keep = 1:frame_stride:length(t_global)
        frame_index = frame_index[keep]
        t_global    = t_global[keep]

        Nt = length(t_global)
        Nt == 0 && error("no frames selected (check t_max=$t_max / frame_stride=$frame_stride)")
        @info "Animating $(basename(run_dir)) in 3D: $Nt frames (stride=$frame_stride), t ∈ [$(round(t_global[1], digits=2)), $(round(t_global[end], digits=2))]"

        # --- wet mask from t-index 2 (t-index 1 is the degenerate IC) ---
        b_ref = Array(datasets[1]["b"][xi, :, zi, 2])
        wet   = b_ref .!= 0.0

        yidxs = round.(Int, range(1, Ny, length=n_slices))

        Xg = repeat(x, 1, Nzs)
        Zg = repeat(reshape(z, 1, Nzs), Nxs, 1)

        fig = Figure(size = figsize)
        n   = Observable(1)
        title = @lift @sprintf("buoyancy (3D)  Ra = %.2e  t = %.2f", Ra, t_global[$n])

        ax = Axis3(fig[1, 1];
            title = title, xlabel = L"x", ylabel = L"y", zlabel = L"z",
            aspect = aspect, elevation = elevation, azimuth = azimuth,
            limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2), (-H, 0)), titlesize = 20)

        # topography is static across the whole animation — draw it once
        surface!(ax, x, y, bottom; colormap = :turbid, colorrange = (-H, 0), shading = NoShading)

        # buoyancy cross-section slices are recreated every frame (see docstring
        # for why this beats mutating an Observable-linked `color` attribute)
        slice_plots = Any[]
        function draw_slices!(seg, k)
            for p in slice_plots
                delete!(ax, p)
            end
            empty!(slice_plots)
            b = Array(datasets[seg]["b"][xi, :, zi, k])
            for yidx in yidxs
                slab = b[:, yidx, :]
                slab[.!wet[:, yidx, :]] .= NaN
                Yg = fill(y[yidx], Nxs, Nzs)
                p = surface!(ax, Xg, Yg, Zg; color = slab, colormap = :balance,
                             colorrange = B_lims, shading = NoShading, transparency = false)
                push!(slice_plots, p)
            end
        end

        seg1, k1 = frame_index[1]
        draw_slices!(seg1, k1)

        Colorbar(fig[1, 2]; colormap = :balance, colorrange = B_lims, label = "b")

        record(fig, output_file, 1:Nt; framerate = framerate) do i
            seg, k = frame_index[i]
            draw_slices!(seg, k)
            n[] = i
        end

        @info "Animation saved to $output_file"
        return output_file
    finally
        foreach(close, datasets)
    end
end

# ---------------------------------------------------------------------------
# Driver: edit run_dir / kwargs below, then run from scripts/. This is a slow,
# CPU-heavy render (see docstring/report) — for the full segment range, submit
# via `submit_animate_3d.sh` (SLURM) rather than running on a login node:
#   julia --project=../ analysis_scripts/animate_3d_buoyancy.jl
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    const GRC = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC"
    run_dir = joinpath(GRC, "ra1e8_4xstretch_threehill_baseforcing_zerostart")
    animate_3d_buoyancy(run_dir; nsegments = 31, frame_stride = 6)
end