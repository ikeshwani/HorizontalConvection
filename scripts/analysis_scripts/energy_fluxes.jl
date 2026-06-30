# energy_fluxes.jl
#
# Plot the reversible buoyancy flux ϕ_z and the energy-supply flux ϕ_i as
# functions of time.
#
# Thin script: vertical_b_flux / phi_i physics live in
# TopographicHorizontalConvection; this file loads the segmented dataset, calls
# them per segment (dropping overlapping time steps), and plots.
#
# Run from scripts/ with:  julia --project=../ analysis_scripts/energy_fluxes.jl

using TopographicHorizontalConvection   # physics: vertical_b_flux, phi_i
using NCDatasets
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

# ---- compute fluxes over segments, skipping overlapping time steps ----
time = Float64[]
ϕ_i  = Float64[]
ϕ_z  = Float64[]

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

        ϕ_i_seg = phi_i(ds)            # whole-segment time series
        ϕ_z_seg = vertical_b_flux(ds)

        for k in valid[1]:length(t_seg)
            push!(time, t_seg[k])
            push!(ϕ_i,  ϕ_i_seg[k])
            push!(ϕ_z,  ϕ_z_seg[k])
        end

        t_last = t_seg[end]
        close(ds)
        @info "  seg $s: processed through t = $(round(t_last, digits=2))  (total $(length(time)) steps)"
    end
end

# ---- plot ----
fig = Figure()
ax = Axis(fig[1,1], xlabel="time", ylabel="flux",
          title = "$(tag) — ϕ_z and ϕ_i versus time")

scatter!(ax, time, ϕ_i, label = "ϕ_i")
scatter!(ax, time, ϕ_z, label = "ϕ_z")

fig[1,2] = axislegend()

save(joinpath(plot_dir, "phi_i_z_$(tag).png"), fig)
@info "saved flux plot → $(joinpath(plot_dir, "phi_i_z_$(tag).png"))"
