using Printf
using NCDatasets
using CairoMakie
using Observables
using TOML

"""
    make_animation(run_dir; kwargs...)

Build a buoyancy animation (x-z and y-z midplane slices) for one self-contained
simulation run folder produced by the new output layout, e.g.
`output/GPU/GRC/ra1e8_4xstretch_flat_baseforcing_zerostart/`.

The run folder is expected to contain `buoyancy_seg<N>.nc` segment files (and,
optionally, a `config.toml`). Segments are auto-detected and naturally sorted,
overlapping timesteps between consecutive segments are removed, and the hill
overlay is drawn only when the run actually has topography (`numhill > 0`,
read from `config.toml`, or inferred from the data when no config is present).

# Arguments
- `run_dir::AbstractString`: path to the self-contained run folder (the data dir).

# Keyword arguments
- `output_dir = joinpath(run_dir, "figures")`: where to write the `.mp4`.
- `filename   = "buoyancy_animation.mp4"`: output file name.
- `nsegments  = nothing`: number of leading segments to use; `nothing` = all
  (after the active-segment guard below). Setting this disables the guard.
- `drop_active = true`: drop the highest-numbered segment if its file was
  modified within `active_window` seconds — i.e. a simulation is still writing
  to it. Reading a NetCDF file mid-append can error or yield a partial frame,
  so by default we skip it. Ignored when `nsegments` is given explicitly.
- `active_window = 900`: seconds; a segment touched more recently than this is
  treated as "still being written" by `drop_active`.
- `t_max      = nothing`: stop the animation at this nondimensional time; `nothing` = full run.
- `B_lims     = (-0.5, 0.5)`: buoyancy colour limits.
- `framerate  = 24`: frames per second.
- `numhill    = nothing`: override topography detection; `nothing` = read config/data.

Returns the path to the written animation.
"""
function make_animation(run_dir::AbstractString;
                        output_dir    = joinpath(run_dir, "figures"),
                        filename      = "buoyancy_animation.mp4",
                        nsegments     = nothing,
                        drop_active   = true,
                        active_window = 900,
                        t_max         = nothing,
                        B_lims        = (-0.5, 0.5),
                        framerate     = 24,
                        numhill       = nothing)

    isdir(run_dir) || error("run_dir does not exist: $run_dir")

    # --- discover and naturally sort buoyancy segment files ---
    seg_re   = r"^buoyancy_seg(\d+)\.nc$"
    seg_nums = sort([parse(Int, match(seg_re, f).captures[1])
                     for f in readdir(run_dir) if occursin(seg_re, f)])
    isempty(seg_nums) && error("no buoyancy_seg*.nc files found in $run_dir")

    if nsegments !== nothing
        # explicit cap: trust the caller, no active-segment guard
        seg_nums = seg_nums[1:min(nsegments, end)]
    elseif drop_active && length(seg_nums) > 1
        # guard: if the last segment is still being appended (recent mtime),
        # a simulation is likely running — drop it to avoid reading mid-write.
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
        x  = ds1["x_caa"][:]
        y  = ds1["y_aca"][:]
        z  = ds1["z_aac"][:]

        # --- topography: prefer config.toml, else infer from any dry cells ---
        if numhill === nothing
            cfg = joinpath(run_dir, "config.toml")
            if isfile(cfg)
                numhill = get(get(TOML.parsefile(cfg), "topography", Dict()), "numhill", nothing)
            end
        end

        # --- global frame index, dropping overlapping timesteps between segments ---
        frame_index = Tuple{Int,Int}[]
        t_global    = Float64[]
        let t_end_prev = -Inf
            for seg in eachindex(datasets)
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

        # --- optional time truncation ---
        if t_max !== nothing
            keep = findall(t -> t <= t_max, t_global)
            frame_index = frame_index[keep]
            t_global    = t_global[keep]
        end

        Nt = length(t_global)
        Nt == 0 && error("no frames selected (check t_max=$t_max)")
        @info "Animating $(basename(run_dir)): $Nt frames, t ∈ [$(round(t_global[1], digits=2)), $(round(t_global[end], digits=2))]"

        yidx = Ny ÷ 2   # mid-y slice for x-z plane
        xidx = 1        # leftmost x slice for y-z plane (cold plume entry)

        # --- hill mask from seg1 time index 2 (avoids cold-start zeros) ---
        # Only meaningful when the run has topography; flat runs draw no overlay.
        draw_hill = numhill === nothing ? true : numhill > 0
        b_ref_xz = Array(datasets[1]["b"][:, yidx, :, 2])
        b_ref_yz = Array(datasets[1]["b"][xidx, :, :, 2])

        hill_xz = fill(NaN, Nx, Nz)
        hill_yz = fill(NaN, Ny, Nz)
        if draw_hill
            hill_xz[b_ref_xz .== 0.0] .= 1.0
            hill_yz[b_ref_yz .== 0.0] .= 1.0
        end

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

        fig = Figure(size = (800, 1200))

        ax_xz = Axis(fig[1, 1];
            title = title_xz, xlabel = L"x", ylabel = L"z",
            limits = ((-Lx/2, Lx/2), (-H, 0)), aspect = Lx / H, titlesize = 20)

        ax_yz = Axis(fig[2, 1];
            title = title_yz, xlabel = L"y", ylabel = L"z",
            limits = ((-Ly/2, Ly/2), (-H, 0)), aspect = Ly / H, titlesize = 20)

        hm1 = heatmap!(ax_xz, x, z, b_xzₙ; colormap = :balance, colorrange = B_lims)
        draw_hill && heatmap!(ax_xz, x, z, hill_xz; colormap = :turbid)
        Colorbar(fig[1, 2], hm1)

        hm2 = heatmap!(ax_yz, y, z, b_yzₙ; colormap = :balance, colorrange = B_lims)
        draw_hill && heatmap!(ax_yz, y, z, hill_yz; colormap = :turbid)
        Colorbar(fig[2, 2], hm2)

        record(fig, output_file, 1:Nt; framerate = framerate) do i
            n[] = i
        end

        @info "Animation saved to $output_file"
        return output_file
    finally
        foreach(close, datasets)
    end
end

# ---------------------------------------------------------------------------
# Driver: edit/uncomment the runs you want to animate, then run from scripts/:
#   julia --project=../ analysis_scripts/make_animation.jl
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    const GRC = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC"
    runs = [
        # "ra1e6_4xstretch_threehill_baseforcing_zerostart",
        # "ra1e6_4xstretch_flat_baseforcing_zerostart",
        "ra1e8_4xstretch_threehill_baseforcing_zerostart"
    ]
    for run in runs
        make_animation(joinpath(GRC, run))
    end
end
