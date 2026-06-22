using NCDatasets
using CairoMakie
using Statistics
using Printf

# =========================================================
# PARAMETERS
# =========================================================
experiment  = "3hill"
Ra_str      = "RA1e8"
stretch_str = "4x_stretch"
grid_str    = "512_128"
avg_window  = 10.0

data_dir       = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/$(Ra_str)/$(stretch_str)/$(grid_str)/"
psib_file      = joinpath(data_dir, "psi_b_t385.nc")
gmix_file      = joinpath(data_dir, "Gmix_regions_v2_RA1e8_seg1to14.nc")
ctrl_data_dir  = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/Control/$(Ra_str)/$(stretch_str)/$(grid_str)/"
ctrl_psib_file = joinpath(ctrl_data_dir, "psi_b_Control_RA1e8_seg1to12.nc")
ctrl_gmix_file = joinpath(ctrl_data_dir, "Gmix_regions_Control_RA1e8_seg1to12.nc")
plot_dir       = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU/GRC/$(experiment)/$(Ra_str)/$(stretch_str)/figures/"
mkpath(plot_dir)

# =========================================================
# grid metadata
# =========================================================
ds1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
Lx  = Float64(ds1.attrib["Lx"])
Ra  = Float64(ds1.attrib["Ra"])
close(ds1)
zBL = -(round(Lx * Ra^(-1/5); digits=2) + 0.02)

# =========================================================
# shared t_end: anchor to control's last time so both
# experiments average over the same physical window
# =========================================================
ds_ctrl_ref = NCDataset(ctrl_psib_file)
t_end = Float64(ds_ctrl_ref["time"][end])
close(ds_ctrl_ref)
@printf("shared averaging window: t = %.1f → %.1f\n", t_end - avg_window, t_end)

# =========================================================
# load ψ(x, b, t) — hill
# =========================================================
println("loading hill ψ(x, b) ...")
ds_psi = NCDataset(psib_file)
x_psi  = Float64.(ds_psi["x"][:])
b_psi  = Float64.(ds_psi["b_out"][:])
t_psi  = Float64.(ds_psi["time"][:])
i_avg  = searchsortedfirst(t_psi, t_end - avg_window):length(t_psi)
ψ_mean = dropdims(mean(Float64.(ds_psi["ψ_b"][:, :, i_avg]), dims=3), dims=3)  # [Nx, n_b]
close(ds_psi)
@printf("  hill: averaged over %d steps  (t = %.1f → %.1f)\n", length(i_avg), t_psi[i_avg[1]], t_end)

# =========================================================
# load ψ(x, b, t) — control
# =========================================================
println("loading control ψ(x, b) ...")
ds_ctrl      = NCDataset(ctrl_psib_file)
x_ctrl       = Float64.(ds_ctrl["x"][:])
b_ctrl       = Float64.(ds_ctrl["b"][:])
t_ctrl       = Float64.(ds_ctrl["time"][:])
i_avg_ctrl   = searchsortedfirst(t_ctrl, t_end - avg_window):length(t_ctrl)
ψ_ctrl_mean  = dropdims(mean(Float64.(ds_ctrl["psi_b"][:, :, i_avg_ctrl]), dims=3), dims=3)  # [Nx, n_b]
close(ds_ctrl)
@printf("  control: averaged over %d steps  (t = %.1f → %.1f)\n",
        length(i_avg_ctrl), t_ctrl[i_avg_ctrl[1]], t_end)

# =========================================================
# load G_mix per region (hill) — same window
# =========================================================
println("loading G_mix by region ...")
ds_g  = NCDataset(gmix_file)
t_g   = Float64.(ds_g["time"][:])
b_g   = Float64.(ds_g["b"][:])   # 499 inner bins matching b_psi[2:end-1]
i_g   = searchsortedfirst(t_g, t_end - avg_window):length(t_g)

region_keys = ["Gmix_hill1", "Gmix_basin1", "Gmix_hill2",
               "Gmix_basin2", "Gmix_hill3", "Gmix_basin3"]
gmix_means  = Dict(
    rk => dropdims(mean(Float64.(ds_g[rk][:, i_g]), dims=2), dims=2)
    for rk in region_keys
)
close(ds_g)
@printf("  G_mix: averaged over %d steps\n", length(i_g))

println("loading control G_mix by region ...")
ds_gc    = NCDataset(ctrl_gmix_file)
t_gc     = Float64.(ds_gc["time"][:])
b_g_ctrl = Float64.(ds_gc["b"][:])
i_gc     = searchsortedfirst(t_gc, t_end - avg_window):length(t_gc)
gmix_ctrl_means = Dict(
    rk => dropdims(mean(Float64.(ds_gc[rk][:, i_gc]), dims=2), dims=2)
    for rk in region_keys
)
close(ds_gc)
@printf("  G_mix ctrl: averaged over %d steps\n", length(i_gc))

# =========================================================
# regions — Basin 0 excluded
# =========================================================
regions = [
    ("Hill 1",  "Gmix_hill1",  -1.35, -0.65),
    ("Basin 1", "Gmix_basin1", -0.65, -0.35),
    ("Hill 2",  "Gmix_hill2",  -0.35,  0.35),
    ("Basin 2", "Gmix_basin2",  0.35,  0.65),
    ("Hill 3",  "Gmix_hill3",   0.65,  1.35),
    ("Basin 3", "Gmix_basin3",  1.35,  x_psi[end]),
]
n_reg = length(regions)   # 6

nearest_xi(xv, xt) = argmin(abs.(xv .- xt))

# align ψ to the 499-bin b_g grid (same as G_mix b-axis)
ψ_inner      = ψ_mean[:, 2:end-1]        # [Nx, 499]  hill
ψ_ctrl_inner = ψ_ctrl_mean[:, 2:end-1]   # [Nx, 499]  control

db = b_g[2] - b_g[1]

# per-region boundary fluxes
ψ_L_all      = Vector{Vector{Float64}}(undef, n_reg)
ψ_R_all      = Vector{Vector{Float64}}(undef, n_reg)
Δψ_all       = Vector{Vector{Float64}}(undef, n_reg)
g_all        = Vector{Vector{Float64}}(undef, n_reg)
g_ctrl_all   = Vector{Vector{Float64}}(undef, n_reg)
ψ_L_ctrl_all = Vector{Vector{Float64}}(undef, n_reg)
ψ_R_ctrl_all = Vector{Vector{Float64}}(undef, n_reg)
Δψ_ctrl_all  = Vector{Vector{Float64}}(undef, n_reg)
x_L_used     = Vector{Float64}(undef, n_reg)
x_R_used     = Vector{Float64}(undef, n_reg)

for (idx, (_, rk, xL, xR)) in enumerate(regions)
    iL = nearest_xi(x_psi, xL);  iR = nearest_xi(x_psi, xR)
    ψ_L_all[idx]  = ψ_inner[iL, :]
    ψ_R_all[idx]  = ψ_inner[iR, :]
    Δψ_all[idx]   = ψ_inner[iR, :] .- ψ_inner[iL, :]
    g_all[idx]    = gmix_means[rk]
    x_L_used[idx] = x_psi[iL]
    x_R_used[idx] = x_psi[iR]

    iL_c = nearest_xi(x_ctrl, xL);  iR_c = nearest_xi(x_ctrl, xR)
    ψ_L_ctrl_all[idx] = ψ_ctrl_inner[iL_c, :]
    ψ_R_ctrl_all[idx] = ψ_ctrl_inner[iR_c, :]
    Δψ_ctrl_all[idx]  = ψ_ctrl_inner[iR_c, :] .- ψ_ctrl_inner[iL_c, :]
    g_ctrl_all[idx]   = gmix_ctrl_means[rk]
end

# =========================================================
# shared axis styles for poster
# =========================================================
poster_axis_kw = (
    xlabelsize       = 26,
    ylabelsize       = 26,
    xticklabelsize   = 20,
    yticklabelsize   = 20,
    titlesize        = 30,
)
legend_kw = (
    orientation  = :horizontal,
    tellwidth    = false,
    framevisible = false,
    labelsize    = 22,
    patchsize    = (35, 22),
)

# =========================================================
# Figure A — Hill: ψ_in, ψ_out, Δψ  (no G_mix)
#            6 panels side by side,  b on y-axis
# =========================================================
println("plotting Figure A: hill  ψ_in / ψ_out / Δψ  (no G_mix) ...")
let
    fig = Figure(size=(n_reg * 370, 620))
    axes = Axis[]
    for (idx, (label, _, _, _)) in enumerate(regions)
        ψ_L = ψ_L_all[idx];  ψ_R = ψ_R_all[idx];  Δψ = Δψ_all[idx]

        ax = Axis(fig[1, idx];
            ylabel  = "b / b★",
            xlabel  = "ψ",
            title   = label,
            poster_axis_kw...
        )
        push!(axes, ax)
        idx > 1 && hideydecorations!(ax; ticks=false, grid=false)

        lines!(ax, ψ_L, b_g; color=:steelblue, linewidth=2.5)
        lines!(ax, ψ_R, b_g; color=:orangered, linewidth=2.5)
        lines!(ax, Δψ,  b_g; color=:purple,    linewidth=2.5, linestyle=:dash)
        hlines!(ax, 0; color=:gray, linewidth=0.8, linestyle=:dot)
        vlines!(ax, 0; color=:gray, linewidth=0.8, linestyle=:dot)
    end
    linkyaxes!(axes...)

    Legend(fig[2, 1:n_reg],
        [LineElement(color=:steelblue, linewidth=2.5),
         LineElement(color=:orangered, linewidth=2.5),
         LineElement(color=:purple,    linewidth=2.5, linestyle=:dash)],
        ["ψ in  (left boundary)", "ψ out  (right boundary)", "Δψ = ψ_out − ψ_in"];
        legend_kw...)

    save(joinpath(plot_dir, "psi_b_flux_noGmix_hill.png"),    fig; px_per_unit=2)
    println("  saved → psi_b_flux_noGmix_hill.png")
end

# =========================================================
# Figure B — Control: ψ_in, ψ_out, Δψ  (no G_mix)
#            6 panels side by side,  b on y-axis
# =========================================================
println("plotting Figure B: control  ψ_in / ψ_out / Δψ  (no G_mix) ...")
let
    fig = Figure(size=(n_reg * 370, 620))
    axes = Axis[]
    for (idx, (label, _, _, _)) in enumerate(regions)
        ψ_L = ψ_L_ctrl_all[idx];  ψ_R = ψ_R_ctrl_all[idx];  Δψ = Δψ_ctrl_all[idx]

        ax = Axis(fig[1, idx];
            ylabel  = "b / b★",
            xlabel  = "ψ",
            title   = label,
            poster_axis_kw...
        )
        push!(axes, ax)
        idx > 1 && hideydecorations!(ax; ticks=false, grid=false)

        lines!(ax, ψ_L, b_g; color=:steelblue, linewidth=2.5)
        lines!(ax, ψ_R, b_g; color=:orangered, linewidth=2.5)
        lines!(ax, Δψ,  b_g; color=:purple,    linewidth=2.5, linestyle=:dash)
        hlines!(ax, 0; color=:gray, linewidth=0.8, linestyle=:dot)
        vlines!(ax, 0; color=:gray, linewidth=0.8, linestyle=:dot)
    end
    linkyaxes!(axes...)

    Legend(fig[2, 1:n_reg],
        [LineElement(color=:steelblue, linewidth=2.5),
         LineElement(color=:orangered, linewidth=2.5),
         LineElement(color=:purple,    linewidth=2.5, linestyle=:dash)],
        ["ψ in  (left boundary)", "ψ out  (right boundary)", "Δψ = ψ_out − ψ_in"];
        legend_kw...)

    save(joinpath(plot_dir, "psi_b_flux_noGmix_control.png"), fig; px_per_unit=2)
    println("  saved → psi_b_flux_noGmix_control.png")
end

# =========================================================
# Figure C — Hill: ψ_in, ψ_out, Δψ + G_mix on same axis
#            not normalised,  b on y-axis
# =========================================================
println("plotting Figure C: hill  ψ_in / ψ_out / Δψ + G_mix  (same axis) ...")
let
    fig = Figure(size=(n_reg * 390, 800); figure_padding=30)
    Label(fig[0, 1:n_reg],
          "3-hill  Ra = 1e8 — ψ_in, ψ_out, Δψ and G_mix by region  (t = $(@sprintf("%.1f", t_end - avg_window)) → $(@sprintf("%.1f", t_end)))";
          fontsize=28, font=:bold, tellwidth=false)
    axes = Axis[]
    for (idx, (label, _, _, _)) in enumerate(regions)
        ψ_L = ψ_L_all[idx];  ψ_R = ψ_R_all[idx]
        Δψ  = Δψ_all[idx];   g   = g_all[idx]

        ax = Axis(fig[1, idx];
            ylabel  = "b / b★",
            xlabel  = "ψ  /  G_mix",
            title   = label,
            limits  = (-0.005, 0.005, nothing, nothing),
            poster_axis_kw...
        )
        push!(axes, ax)
        idx > 1 && hideydecorations!(ax; ticks=false, grid=false)

        lines!(ax, ψ_L, b_g; color=:steelblue, linewidth=2.5)
        lines!(ax, ψ_R, b_g; color=:orangered, linewidth=2.5)
        lines!(ax, Δψ,  b_g; color=:purple,    linewidth=2.5, linestyle=:dash)
        lines!(ax, g,   b_g; color=:seagreen,  linewidth=2.5, linestyle=:dot)
        hlines!(ax, 0; color=:gray, linewidth=0.8, linestyle=:dot)
        vlines!(ax, 0; color=:gray, linewidth=0.8, linestyle=:dot)
    end
    linkyaxes!(axes...)
    rowgap!(fig.layout, 1, 30)

    Legend(fig[2, 1:n_reg],
        [LineElement(color=:steelblue, linewidth=2.5),
         LineElement(color=:orangered, linewidth=2.5),
         LineElement(color=:purple,    linewidth=2.5, linestyle=:dash),
         LineElement(color=:seagreen,  linewidth=2.5, linestyle=:dot)],
        ["ψ in", "ψ out", "Δψ = ψ_out − ψ_in", "G_mix"];
        legend_kw...)

    save(joinpath(plot_dir, "psi_b_flux_withGmix_hill.png"),  fig; px_per_unit=2)
    println("  saved → psi_b_flux_withGmix_hill.png")
end

# =========================================================
# Figure C_ctrl — Control: ψ_in, ψ_out, Δψ + G_mix on same axis
#                 not normalised,  b on y-axis
# =========================================================
println("plotting Figure C_ctrl: control  ψ_in / ψ_out / Δψ + G_mix  (same axis) ...")
let
    fig = Figure(size=(n_reg * 390, 800); figure_padding=30)
    Label(fig[0, 1:n_reg],
          "Control  Ra = 1e8 — ψ_in, ψ_out, Δψ and G_mix by region  (t = $(@sprintf("%.1f", t_end - avg_window)) → $(@sprintf("%.1f", t_end)))";
          fontsize=28, font=:bold, tellwidth=false)
    axes = Axis[]
    for (idx, (label, _, _, _)) in enumerate(regions)
        ψ_L = ψ_L_ctrl_all[idx];  ψ_R = ψ_R_ctrl_all[idx]
        Δψ  = Δψ_ctrl_all[idx];   g   = g_ctrl_all[idx]

        ax = Axis(fig[1, idx];
            ylabel  = "b / b★",
            xlabel  = "ψ  /  G_mix",
            title   = label,
            limits  = (-0.005, 0.005, nothing, nothing),
            poster_axis_kw...
        )
        push!(axes, ax)
        idx > 1 && hideydecorations!(ax; ticks=false, grid=false)

        lines!(ax, ψ_L, b_g_ctrl; color=:steelblue, linewidth=2.5)
        lines!(ax, ψ_R, b_g_ctrl; color=:orangered, linewidth=2.5)
        lines!(ax, Δψ,  b_g_ctrl; color=:purple,    linewidth=2.5, linestyle=:dash)
        lines!(ax, g,   b_g_ctrl; color=:seagreen,  linewidth=2.5, linestyle=:dot)
        hlines!(ax, 0; color=:gray, linewidth=0.8, linestyle=:dot)
        vlines!(ax, 0; color=:gray, linewidth=0.8, linestyle=:dot)
    end
    linkyaxes!(axes...)
    rowgap!(fig.layout, 1, 30)

    Legend(fig[2, 1:n_reg],
        [LineElement(color=:steelblue, linewidth=2.5),
         LineElement(color=:orangered, linewidth=2.5),
         LineElement(color=:purple,    linewidth=2.5, linestyle=:dash),
         LineElement(color=:seagreen,  linewidth=2.5, linestyle=:dot)],
        ["ψ in", "ψ out", "Δψ = ψ_out − ψ_in", "G_mix"];
        legend_kw...)

    save(joinpath(plot_dir, "psi_b_flux_withGmix_control.png"),  fig; px_per_unit=2)
    println("  saved → psi_b_flux_withGmix_control.png")
end

# =========================================================
# Figure D — ψ(x, b) heatmap with region boundary vlines
# =========================================================
println("plotting Figure D: ψ(x, b) heatmap ...")
let
    x_bounds = [-1.8, -1.35, -0.65, -0.35, 0.35, 0.65, 1.35]
    ψ_lim    = quantile(abs.(ψ_mean[:]), 0.99)

    fig4 = Figure(size=(1100, 430))
    ax   = Axis(fig4[1, 1];
        xlabel    = "x / H",
        ylabel    = "b / b★",
        title     = "$(experiment)  Ra = $(Ra_str) — time-mean ψ(x, b)  [last $(avg_window) τ]",
        titlesize = 13,
    )
    hm = heatmap!(ax, x_psi, b_psi, ψ_mean;
        colormap = :balance, colorrange = (-ψ_lim, ψ_lim))
    Colorbar(fig4[1, 2], hm; label = "ψ(x, b)")

    hlines!(ax, 0; color=:white, linewidth=0.8, linestyle=:dot)
    for xb in x_bounds
        vlines!(ax, xb; color=:white, linewidth=1.5, linestyle=:dash)
    end
    vlines!(ax, -1.8; color=(:yellow, 0.9), linewidth=2.0)

    region_centers = [
        (-1.80 + -1.35)/2, (-1.35 + -0.65)/2, (-0.65 + -0.35)/2, (-0.35 + 0.35)/2,
        (0.35 + 0.65)/2,   (0.65 + 1.35)/2,   (1.35 + x_psi[end])/2
    ]
    for (xc, lbl) in zip(region_centers, ["B0", "H1", "B1", "H2", "B2", "H3", "B3"])
        text!(ax, xc, b_psi[end] * 0.88; text=lbl, color=:white,
              align=(:center, :top), fontsize=11, font=:bold)
    end

    save(joinpath(plot_dir, "psi_b_xb_heatmap_regions.png"), fig4; px_per_unit=2)
    println("  saved → psi_b_xb_heatmap_regions.png")
end

# =========================================================
# Figure E — ψ(x, b) heatmap zoomed to dense classes b[200:250]
# =========================================================
println("plotting Figure E: ψ(x, b) heatmap — dense classes b[200:250] ...")
let
    b_idx  = 200:250
    b_zoom = b_psi[b_idx]
    ψ_zoom = ψ_mean[:, b_idx]
    ψ_lim  = quantile(abs.(ψ_zoom[:]), 0.99)

    x_bounds = [-1.8, -1.35, -0.65, -0.35, 0.35, 0.65, 1.35]
    y_label  = b_zoom[1] + 0.9 * (b_zoom[end] - b_zoom[1])

    fig5 = Figure(size=(1100, 430))
    ax   = Axis(fig5[1, 1];
        xlabel    = "x / H",
        ylabel    = "b / b★",
        title     = "$(experiment)  Ra = $(Ra_str) — ψ(x, b)  dense classes [b[200:250]]  [last $(avg_window) τ]",
        titlesize = 13,
    )
    hm = heatmap!(ax, x_psi, b_zoom, ψ_zoom;
        colormap = :balance, colorrange = (-ψ_lim, ψ_lim))
    Colorbar(fig5[1, 2], hm; label = "ψ(x, b)")

    hlines!(ax, 0; color=:white, linewidth=0.8, linestyle=:dot)
    for xb in x_bounds
        vlines!(ax, xb; color=:white, linewidth=1.5, linestyle=:dash)
    end
    vlines!(ax, -1.8; color=(:yellow, 0.9), linewidth=2.0)

    region_centers = [
        (-1.80 + -1.35)/2, (-1.35 + -0.65)/2, (-0.65 + -0.35)/2, (-0.35 + 0.35)/2,
        (0.35 + 0.65)/2,   (0.65 + 1.35)/2,   (1.35 + x_psi[end])/2
    ]
    for (xc, lbl) in zip(region_centers, ["B0", "H1", "B1", "H2", "B2", "H3", "B3"])
        text!(ax, xc, y_label; text=lbl, color=:white,
              align=(:center, :top), fontsize=11, font=:bold)
    end

    save(joinpath(plot_dir, "psi_b_xb_heatmap_dense_zoom.png"), fig5; px_per_unit=2)
    println("  saved → psi_b_xb_heatmap_dense_zoom.png")
end

# =========================================================
# Figure F — Control vs 3-hill ψ(x, b) side-by-side, dense classes
# =========================================================
println("plotting Figure F: Control vs 3-hill ψ(x, b) comparison — dense classes ...")
let
    b_idx        = 200:250
    b_ctrl_zoom  = b_ctrl[b_idx]
    b_hill_zoom  = b_psi[b_idx]
    ψ_ctrl_zoom  = ψ_ctrl_mean[:, b_idx]
    ψ_hill_zoom  = ψ_mean[:, b_idx]

    ψ_lim = max(quantile(abs.(ψ_ctrl_zoom[:]), 0.99),
                quantile(abs.(ψ_hill_zoom[:]), 0.99))

    x_bounds = [-1.8, -1.35, -0.65, -0.35, 0.35, 0.65, 1.35]

    fig6 = Figure(size=(1600, 480))

    ax1 = Axis(fig6[1, 1];
        xlabel    = "x / H",
        ylabel    = "b / b★",
        title     = "Control — ⟨ψ(x,b)⟩  dense classes [b[200:250]]  [last $(avg_window) τ]",
        titlesize = 14,
    )
    heatmap!(ax1, x_ctrl, b_ctrl_zoom, ψ_ctrl_zoom;
        colormap = :balance, colorrange = (-ψ_lim, ψ_lim))

    ax2 = Axis(fig6[1, 2];
        xlabel    = "x / H",
        ylabel    = "b / b★",
        title     = "3-hill — ⟨ψ(x,b)⟩  dense classes [b[200:250]]  [last $(avg_window) τ]",
        titlesize = 14,
    )
    hm = heatmap!(ax2, x_psi, b_hill_zoom, ψ_hill_zoom;
        colormap = :balance, colorrange = (-ψ_lim, ψ_lim))
    Colorbar(fig6[1, 3], hm; label = "ψ(x, b)")

    for ax in (ax1, ax2)
        hlines!(ax, 0; color=:white, linewidth=0.8, linestyle=:dot)
        for xb in x_bounds
            vlines!(ax, xb; color=:white, linewidth=1.5, linestyle=:dash)
        end
        vlines!(ax, -1.8; color=(:yellow, 0.9), linewidth=2.0)
    end

    save(joinpath(plot_dir, "psi_b_heatmap_comparison_dense_zoom.png"), fig6; px_per_unit=2)
    println("  saved → psi_b_heatmap_comparison_dense_zoom.png")
end

# =========================================================
# Figure G — Hill − Control difference:  Δψ and G_mix
#            isolates the topographic signal in each region
# =========================================================
println("plotting Figure G: hill − control difference  Δψ and G_mix ...")
let
    fig = Figure(size=(n_reg * 390, 800); figure_padding=30)
    Label(fig[0, 1:n_reg],
          "3-hill − Control  Ra = 1e8 — Δψ and G_mix difference by region  (t = $(@sprintf("%.1f", t_end - avg_window)) → $(@sprintf("%.1f", t_end)))";
          fontsize=28, font=:bold, tellwidth=false)
    axes = Axis[]
    for (idx, (label, _, _, _)) in enumerate(regions)
        ΔΔψ = Δψ_all[idx]  .- Δψ_ctrl_all[idx]
        Δg  = g_all[idx]   .- g_ctrl_all[idx]

        ax = Axis(fig[1, idx];
            ylabel  = "b / b★",
            xlabel  = "hill − ctrl",
            title   = label,
            poster_axis_kw...
        )
        push!(axes, ax)
        idx > 1 && hideydecorations!(ax; ticks=false, grid=false)

        lines!(ax, ΔΔψ, b_g; color=:purple,   linewidth=2.5, linestyle=:dash)
        lines!(ax, Δg,  b_g; color=:seagreen, linewidth=2.5, linestyle=:dot)
        hlines!(ax, 0; color=:gray, linewidth=0.8, linestyle=:dot)
        vlines!(ax, 0; color=:gray, linewidth=0.8, linestyle=:dot)
    end
    linkyaxes!(axes...)
    rowgap!(fig.layout, 1, 30)

    Legend(fig[2, 1:n_reg],
        [LineElement(color=:purple,   linewidth=2.5, linestyle=:dash),
         LineElement(color=:seagreen, linewidth=2.5, linestyle=:dot)],
        ["Δψ_hill − Δψ_ctrl", "G_mix_hill − G_mix_ctrl"];
        legend_kw...)

    save(joinpath(plot_dir, "psi_b_flux_difference_hill_ctrl.png"), fig; px_per_unit=2)
    println("  saved → psi_b_flux_difference_hill_ctrl.png")
end

# =========================================================
# Figure H — Overlay: ψ_in and ψ_out for hill vs. control
#            hill = solid lines,  control = dashed lines
# =========================================================
println("plotting Figure H: hill vs. control overlay  ψ_in / ψ_out ...")
let
    fig = Figure(size=(n_reg * 390, 800); figure_padding=30)
    Label(fig[0, 1:n_reg],
          "3-hill vs. Control  Ra = 1e8 — ψ_in / ψ_out by region  (t = $(@sprintf("%.1f", t_end - avg_window)) → $(@sprintf("%.1f", t_end)))";
          fontsize=28, font=:bold, tellwidth=false)
    axes = Axis[]
    for (idx, (label, _, _, _)) in enumerate(regions)
        ax = Axis(fig[1, idx];
            ylabel  = "b / b★",
            xlabel  = "ψ",
            title   = label,
            limits  = (-0.005, 0.005, nothing, nothing),
            poster_axis_kw...
        )
        push!(axes, ax)
        idx > 1 && hideydecorations!(ax; ticks=false, grid=false)

        lines!(ax, ψ_L_all[idx],      b_g; color=:steelblue, linewidth=2.5)
        lines!(ax, ψ_R_all[idx],      b_g; color=:orangered, linewidth=2.5)
        lines!(ax, ψ_L_ctrl_all[idx], b_g; color=:steelblue, linewidth=2.0, linestyle=:dash)
        lines!(ax, ψ_R_ctrl_all[idx], b_g; color=:orangered, linewidth=2.0, linestyle=:dash)
        hlines!(ax, 0; color=:gray, linewidth=0.8, linestyle=:dot)
        vlines!(ax, 0; color=:gray, linewidth=0.8, linestyle=:dot)
    end
    linkyaxes!(axes...)
    rowgap!(fig.layout, 1, 30)

    Legend(fig[2, 1:n_reg],
        [LineElement(color=:steelblue, linewidth=2.5),
         LineElement(color=:orangered, linewidth=2.5),
         LineElement(color=:steelblue, linewidth=2.0, linestyle=:dash),
         LineElement(color=:orangered, linewidth=2.0, linestyle=:dash)],
        ["ψ_in  (hill)", "ψ_out  (hill)", "ψ_in  (ctrl)", "ψ_out  (ctrl)"];
        legend_kw...)

    save(joinpath(plot_dir, "psi_b_flux_overlay_hill_ctrl.png"), fig; px_per_unit=2)
    println("  saved → psi_b_flux_overlay_hill_ctrl.png")
end

println("\ndone.")
