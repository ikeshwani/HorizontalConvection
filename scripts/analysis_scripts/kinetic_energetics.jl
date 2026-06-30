# kinetic_energetics.jl
#
# Compute volume-averaged kinetic-energy reservoirs ⟨MKE⟩, ⟨TKE⟩, ⟨KE⟩ vs time
# (plus MKE/TKE densities) for one GRC run, save to NetCDF, plot, and animate.
# ⟨KE⟩ is computed from the velocities (0.5(u²+v²+w²)) so it is consistent with
# ⟨MKE⟩ + ⟨TKE⟩ on the same (velocity) time axis. The Oceanostics `ke` diagnostic
# (written on a coarser time grid) is compared separately as a consistency check.
#
# Thin script: the MKE/TKE/KE physics lives in calc_kinetic_energies in
# TopographicHorizontalConvection; this file loads data, calls it, saves, plots.
#
# GRC output is segmented and large, so velocity time steps are served by a lazy
# multi-segment view (SegTimeVar) that reads only the requested slices from disk —
# the full 4D velocity field is never held in memory.
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/kinetic_energetics.jl

using TopographicHorizontalConvection   # physics: calc_kinetic_energies, mask_land!
using NCDatasets
using NaNStatistics
using CairoMakie
using Observables
using Printf

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
plot_dir  = joinpath(data_dir, "figures")   # figures live inside the run folder
anim_file = joinpath(plot_dir, "MKE_TKE_$(tag).mp4")
mkpath(plot_dir)

mean_window_time = 1.0          # trailing time-window length (nondim time units) for the Reynolds mean

# ---- lazy multi-segment time view ------------------------------------------------
# Indexes like a [Nx,Ny,Nz,Nt] array but maps the global time index to the right
# segment file + local step and reads only that slice from disk (the full velocity
# field is far too large to hold in memory).
struct SegTimeVar
    datasets::Vector{NCDataset}
    varname::String
    seg::Vector{Int}   # global time index -> position in `datasets`
    loc::Vector{Int}   # global time index -> local time index within that dataset
end

# single time step: pick the file + local index this global step maps to
Base.getindex(s::SegTimeVar, xr, yr, zr, t::Integer) =
    s.datasets[s.seg[t]][s.varname][xr, yr, zr, s.loc[t]]

# several time steps: read each one and stack along time (dim 4)
Base.getindex(s::SegTimeVar, xr, yr, zr, ts::AbstractVector{<:Integer}) =
    cat((s[xr, yr, zr, t] for t in ts)...; dims=4)

# ---- grid + wet mask from buoyancy seg1 ----
ds_b1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"), "r")
x = ds_b1["x_caa"][:]
y = ds_b1["y_aca"][:]
z = ds_b1["z_aac"][:]

Ra = ds_b1.attrib["Ra"]
H  = ds_b1.attrib["H"]
Lx = ds_b1.attrib["Lx"]
Ly = ds_b1.attrib["Ly"]
Nx = ds_b1.attrib["Nx"]
Ny = ds_b1.attrib["Ny"]
Nz = ds_b1.attrib["Nz"]

Δx = reshape(ds_b1["Δx_caa"][:], Nx, 1, 1)
Δy = reshape(ds_b1["Δy_aca"][:], 1, Ny, 1)
Δz = reshape(ds_b1["Δz_aac"][:], 1, 1, Nz)
ΔV = Δx .* Δy .* Δz

b_ref   = Array(ds_b1["b"][:, :, :, 2])   # non-initial snapshot (cold start has b=0 at t=0)
wet     = b_ref .!= 0.0
wet_Vol = nansum(wet .* ΔV, dims=(1,2,3))[1,1,1]
close(ds_b1)

# ---- open velocity segments + build global (deduped) time index ----
vel_ds = NCDataset[]
seg    = Int[]
loc    = Int[]
time   = Float64[]

let t_last = -Inf
    for s in segments
        ds = NCDataset(joinpath(data_dir, "velocities_seg$(s).nc"), "r")
        t_seg = ds["time"][:]
        valid = findall(t_seg .> t_last)
        if isempty(valid)
            @info "  vel seg $s: all duplicate time steps — skipping"
            close(ds)
            continue
        end
        push!(vel_ds, ds)
        di = length(vel_ds)
        for k in valid[1]:length(t_seg)
            push!(seg, di); push!(loc, k); push!(time, t_seg[k])
        end
        t_last = t_seg[end]
    end
end

Nt = length(time)
dt = Nt >= 2 ? time[2] - time[1] : 0.1
Nt_window = max(1, round(Int, mean_window_time / dt))
@info "Grid Nx=$Nx Ny=$Ny Nz=$Nz   Nt=$Nt (t=$(time[1])..$(time[end]))   Nt_window=$Nt_window (≈$mean_window_time τ)"

u = SegTimeVar(vel_ds, "u", seg, loc)
v = SegTimeVar(vel_ds, "v", seg, loc)
w = SegTimeVar(vel_ds, "w", seg, loc)

# ---- compute KE reservoirs (physics from src/) ----
MKE_t, TKE_t, KE_t, MKE_xz, TKE_xz =
    calc_kinetic_energies(u, v, w, wet, ΔV, wet_Vol, Nx, Ny, Nz, Nt; Nt_window=Nt_window)

# ---- line plot ----
fig = Figure(size=(800,800))
ax = Axis(fig[1,1],
          xlabel = "time",
          ylabel = "⟨E⟩ [m²/s²]",
          title = @sprintf("%s — Volume-Averaged Kinetic Energies, Ra = %.2e", tag, Ra))
lines!(ax, time, MKE_t, linewidth=2, color=:purple, label="⟨MKE⟩")
lines!(ax, time, TKE_t, linewidth=2, color=:blue, label="⟨TKE⟩")
lines!(ax, time, MKE_t .+ TKE_t, linewidth=2, linestyle=:dash, color=:red, label="⟨MKE⟩ + ⟨TKE⟩")
lines!(ax, time, KE_t, linewidth=2, linestyle=:dash, color=:black, label="⟨KE⟩ (velocities)")
Legend(fig[1,2], ax)
save(joinpath(plot_dir, "KE_$(tag).png"), fig)
@info "saved KE plot → $(joinpath(plot_dir, "KE_$(tag).png"))"

# ---- save ----
output_file = joinpath(data_dir, "KE.nc")
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

    ds_out["time"][:] = time
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
    ds_out["KE"].attrib["long_name"] = "Volume-Averaged Total Kinetic Energy From Velocities"
    ds_out["KE"].attrib["description"] = "⟨KE⟩ = ∫ 0.5(u²+v²+w²) dV / V_wet"

    ds_out["MKE_Density"].attrib["units"] = "m²/s²"
    ds_out["MKE_Density"].attrib["long_name"] = "Mean Kinetic Energy Density"
    ds_out["MKE_Density"].attrib["description"] = "MKE(x,z,t) = 0.5 * (ū² + v̄² + w̄²)"

    ds_out["TKE_Density"].attrib["units"] = "m²/s²"
    ds_out["TKE_Density"].attrib["long_name"] = "Turbulent Kinetic Energy Density"
    ds_out["TKE_Density"].attrib["description"] = "TKE(x,z,t) = 0.5 * ((u'²)‾ + (v'²)‾ + (w'²)‾)"

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
    ds_out.attrib["Nt_window"] = Nt_window
end
@info "Energetics Saved Successfully"

# ---- consistency check: Oceanostics ke vs velocity KE at shared times ----
# Oceanostics ke is a midplane (y = Ny÷2) slice on a coarser time grid; compare it
# to the velocity KE on the SAME slice at the matching times (oceanostics times are
# a subset of the velocity times, so the comparison is exact — no interpolation).
ymid    = max(1, Ny ÷ 2)
ΔV_mid  = ΔV[:, ymid, :]              # [Nx, Nz] column volume of the midplane slab
wet_mid = wet[:, ymid, :]
V_mid   = nansum(wet_mid .* ΔV_mid)

t_ocn      = Float64[]
KE_ocn     = Float64[]
KE_vel_mid = Float64[]

let t_last = -Inf
    for s in segments
        ofile = joinpath(data_dir, "oceanostics_seg$(s).nc")
        isfile(ofile) || continue
        dso = NCDataset(ofile, "r")
        to  = dso["time"][:]
        valid = findall(to .> t_last)
        if isempty(valid); close(dso); continue; end
        for k in valid[1]:length(to)
            gi = findfirst(t -> isapprox(t, to[k]; atol=1e-6), time)
            gi === nothing && continue          # no matching velocity step

            keₖ = Array(dso["ke"][:, 1, :, k])
            mask_land!(keₖ, wet_mid)
            push!(KE_ocn, nansum(keₖ .* ΔV_mid) / V_mid)

            uu = Array(u[1:Nx, ymid, 1:Nz, gi])
            vv = Array(v[1:Nx, ymid, 1:Nz, gi])
            ww = Array(w[1:Nx, ymid, 1:Nz, gi])
            ke_vel = 0.5 .* (uu.^2 .+ vv.^2 .+ ww.^2)
            mask_land!(ke_vel, wet_mid)
            push!(KE_vel_mid, nansum(ke_vel .* ΔV_mid) / V_mid)

            push!(t_ocn, to[k])
        end
        t_last = to[end]
        close(dso)
    end
end

if isempty(t_ocn)
    @warn "no Oceanostics times matched the velocity times — skipping consistency check"
else
    figc = Figure(size=(800,500))
    axc = Axis(figc[1,1], xlabel="time", ylabel="midplane ⟨KE⟩ [m²/s²]",
               title="$(tag) — Oceanostics ke vs velocity KE (midplane, shared times)")
    lines!(axc, t_ocn, KE_vel_mid, linewidth=2, color=:black, label="0.5(u²+v²+w²) from velocities")
    scatter!(axc, t_ocn, KE_ocn, color=:red, markersize=8, label="Oceanostics ke")
    axislegend(axc, position=:rb)
    save(joinpath(plot_dir, "KE_consistency_$(tag).png"), figc)
    rel = maximum(abs.(KE_vel_mid .- KE_ocn) ./ max.(abs.(KE_ocn), eps()))
    @info "saved KE consistency check → $(joinpath(plot_dir, "KE_consistency_$(tag).png"))  (max rel diff $(round(rel*100, digits=2))%)"
end

# ---- animation of ⟨MKE⟩ and ⟨TKE⟩ densities ----
@info "creating animation of ⟨MKE⟩ and ⟨TKE⟩..."

frame = Observable(1)

logMKE = replace(log10.(MKE_xz .+ eps()), NaN => -10.0, Inf => -10.0, -Inf => -10.0)
logTKE = replace(log10.(TKE_xz .+ eps()), NaN => -10.0, Inf => -10.0, -Inf => -10.0)

MKE_obs = @lift logMKE[:, :, $frame]
TKE_obs = @lift logTKE[:, :, $frame]

MKE_lims = (-8, -3)
TKE_lims = (-7, -2)

hill_xz = fill(NaN, Nx, Nz)
hill_xz[.!wet[:, ymid, :]] .= 1.0     # overlay immersed cells on the midplane

title_mke = @lift @sprintf("Log₁₀ ⟨MKE⟩ density, Ra = %.2e, t = %.2f", Ra, time[$frame])
title_tke = @lift @sprintf("Log₁₀ ⟨TKE⟩ density, Ra = %.2e, t = %.2f", Ra, time[$frame])

fig2 = Figure(size=(800,1200))
ax1 = Axis(fig2[1,1], xlabel="x", ylabel="z", title=title_mke)
ax2 = Axis(fig2[2,1], xlabel="x", ylabel="z", title=title_tke)

hm1 = heatmap!(ax1, x, z, MKE_obs; colormap=:deep, colorrange=MKE_lims)
heatmap!(ax1, x, z, hill_xz, colormap=:turbid)
Colorbar(fig2[1,2], hm1)

hm2 = heatmap!(ax2, x, z, TKE_obs; colormap=:matter, colorrange=TKE_lims)
heatmap!(ax2, x, z, hill_xz, colormap=:turbid)
Colorbar(fig2[2,2], hm2)

stride = 50
frames = 1:stride:Nt

record(fig2, anim_file, frames; framerate = 8) do i
    frame[] = i
end
@info "saved animation → $anim_file"

foreach(close, vel_ds)
