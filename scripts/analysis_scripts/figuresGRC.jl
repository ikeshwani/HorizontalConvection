using TopographicHorizontalConvection   # physics: seafloor_profile, boundary_layer_depth
using NCDatasets
using NaNStatistics
using CairoMakie
using Statistics
using Printf

# =========================================================
# PARAMETERS — edit here to switch between experiments
# =========================================================
experiment  = "3hill"       # label used in filenames; "Control" or "3hill"
Ra_str      = "RA1e8"
stretch_str = "4x_stretch"
grid_str    = "512_128"
seg_range   = 1:19
avg_window  = 10.0          # time units to average at end of run

numhill     = 3             # 0 = flat bottom, 1–3 = hills
h₀_frac     = 0.5           # hill height as fraction of H (0 for Control)

# Control:
# data_dir  = ".../GRC/Control/RA1e8/4x_stretch/512_128/"
# gmix_file = joinpath(data_dir, "Gmix_regions_Control_RA1e8_seg1to9.nc")
# numhill=0, h₀_frac=0.0, seg_range=1:9, use_combined=false
data_dir       = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/$(Ra_str)/$(stretch_str)/$(grid_str)/"
gmix_file      = joinpath(data_dir, "Gmix_regions_v2_RA1e8_seg1to14.nc")
ctrl_data_dir  = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/Control/$(Ra_str)/$(stretch_str)/$(grid_str)/"
ctrl_gmix_file = joinpath(ctrl_data_dir, "Gmix_regions_Control_RA1e8_seg1to12.nc")
plot_dir  = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU/GRC/$(experiment)/$(Ra_str)/$(stretch_str)/figures/"
mkpath(plot_dir)

# =========================================================
# grid metadata
# =========================================================
ds1 = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
x   = Float64.(ds1["x_caa"][:])
z   = Float64.(ds1["z_aac"][:])
Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
Lx  = Float64(ds1.attrib["Lx"])
H   = Float64(ds1.attrib["H"])
b★  = Float64(ds1.attrib["b★"])
Ra  = Float64(ds1.attrib["Ra"])
Δx_vec = Float64.(ds1["Δx_caa"][:])
Δz_vec = Float64.(ds1["Δz_aac"][:])
close(ds1)

# =========================================================
# region boundaries
# =========================================================
zBL      = boundary_layer_depth(Lx, Ra)
x_plume  = -1.8                                        # plume / rest-of-domain boundary
x_sub_BL = [-1.35, -0.65, -0.35, 0.35, 0.65, 1.35]   # sub-BL region x-boundaries

# fractional z-axis position of zBL (used to clip sub-BL vlines below the hline)
z_frac_BL = (zBL - z[1]) / (z[end] - z[1])

# =========================================================
# analytic seafloor profile (flat for Control; Gaussian hills otherwise)
# =========================================================
z_sf = seafloor_profile(x, H, Lx, h₀_frac, numhill)

# =========================================================
# load ψ — time-averaged over last avg_window time units
# reads only the needed time slice from the Gmix nc file
# =========================================================
println("loading ψ from Gmix file...")
ds_g   = NCDataset(gmix_file)
time_ψ = ds_g["time"][:]

# anchor averaging window to control's t_end so both scripts use the same window
ds_ctrl_ref = NCDataset(ctrl_gmix_file)
t_end = Float64(ds_ctrl_ref["time"][end])
close(ds_ctrl_ref)

i_start = searchsortedfirst(time_ψ, t_end - avg_window)
i_ψ     = i_start:length(time_ψ)
ψ_mean  = dropdims(mean(ds_g["psi"][:, :, i_ψ], dims=3), dims=3)   # [Nx, Nz]
close(ds_g)
@printf("ψ: averaged over %d steps  (t = %.1f → %.1f)\n", length(i_ψ), time_ψ[i_start], t_end)

# sign convention in the saved file is flipped for the 3-hill run
if experiment == "3hill"
    ψ_mean = -ψ_mean
end

# =========================================================
# load b — y- and time-averaged over last avg_window time units
# (loops segment files, deduplicating overlapping steps)
# =========================================================
b_mean = let
    println("loading b from segments...")
    b_sum  = zeros(Float64, Nx, Nz)
    n_tavg = 0
    t_last = -Inf

    for s in seg_range
        bfile  = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
        t_seg  = bfile["time"][:]
        valid  = findall(t_seg .> t_last)
        isempty(valid) && (close(bfile); continue)

        t_range = valid[1]:valid[end]
        t_last  = t_seg[t_range[end]]

        in_win = findall(t_seg[t_range] .>= t_end - avg_window)
        if !isempty(in_win)
            t_win   = t_range[in_win[1]:in_win[end]]
            b_chunk = Float64.(bfile["b"][:, :, :, t_win])  # [Nx, Ny, Nz, n]
            b_sum  .+= dropdims(sum(b_chunk, dims=(2, 4)), dims=(2, 4))
            n_tavg += length(t_win) * Ny
        end
        close(bfile)
    end

    n_tavg == 0 && error("no b time steps found in averaging window (t ≥ $(t_end - avg_window))")
    @printf("b: averaged over %d time steps × %d y-levels (segments)\n", n_tavg ÷ Ny, Ny)
    Float32.(b_sum ./ n_tavg)
end   # [Nx, Nz]

# =========================================================
# buoyancy contour levels — log-spaced in magnitude
# dense near ±b★, sparse near 0; avoids bunching in the interior
# =========================================================
b_levels = b★ .* [-0.7, -0.6, -0.5, -0.25, -0.22, -0.21, -0.2, -0.15, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

# =========================================================
# figure
# =========================================================
fig = Figure(size=(1100, 400))
ax  = Axis(fig[1, 1];
    xlabel         = "x / H",
    ylabel         = "z / H",
    title          = "$(experiment)  Ra = $(Ra_str) — time-mean ψ with b contours (last $(avg_window) τ)",
    limits         = (x[1], x[end], z[1], 0.0),
    titlesize      = 26,
    xlabelsize     = 20,
    ylabelsize     = 20,
    xticklabelsize = 14,
    yticklabelsize = 14,
)

ψ_lim = 0.004   # hardcoded so both experiments share the same colorrange
hm = heatmap!(ax, x, z, -ψ_mean; colormap=:balance, colorrange=(-ψ_lim, ψ_lim))
Colorbar(fig[1, 2], hm; label="ψ", labelsize=18, ticklabelsize=14)

# buoyancy contours — log-spaced levels
contour!(ax, x, z, b_mean; levels=b_levels, color=:black, linewidth=0.7, labels=true, labelsize=9)

# plume boundary — full height
vlines!(ax, x_plume; color=(:white, 0.85), linewidth=2.5, linestyle=:dash)

# BL depth — full width
hlines!(ax, zBL; color=(:white, 0.85), linewidth=2.5, linestyle=:dash)

# sub-BL region boundaries — ymax=z_frac_BL so they stop at the BL hline
for xb in x_sub_BL
    vlines!(ax, xb; ymax=z_frac_BL, color=(:white, 0.85), linewidth=2.0, linestyle=:dot)
end

# seafloor mask: NaN where ocean (transparent), 1 where inside hill (brown via :turbid)
# drawn last so it covers ψ contours and b contours inside the topography
if numhill > 0
    dry_mask = fill(NaN, Nx, Nz)
    for i in 1:Nx, k in 1:Nz
        z[k] <= z_sf[i] && (dry_mask[i, k] = 0.6)
    end
    heatmap!(ax, x, z, dry_mask; colormap=:turbid, colorrange=(0, 1))
end

# region labels
let region_label_data = [
    ((x[1]   + x_plume) / 2,  z[1] / 2,            "Plume"),
    ((x_plume + x[end]) / 2,  zBL  / 2,            "BL"),
    ((-1.8   + -1.35)   / 2,  (z[1] + zBL) / 2,   "Basin 0"),
    ((-1.35  + -0.65)   / 2,  (z[1] + zBL) / 2,   "Hill 1"),
    ((-0.65  + -0.35)   / 2,  (z[1] + zBL) / 2,   "Basin 1"),
    ((-0.35  +  0.35)   / 2,  (z[1] + zBL) / 2,   "Hill 2"),
    (( 0.35  +  0.65)   / 2,  (z[1] + zBL) / 2,   "Basin 2"),
    (( 0.65  +  1.35)   / 2,  (z[1] + zBL) / 2,   "Hill 3"),
    (( 1.35  + x[end])  / 2,  (z[1] + zBL) / 2,   "Basin 3"),
]
    for (xc, zc, lab) in region_label_data
        text!(ax, xc, zc; text=lab, color=:white, fontsize=11, align=(:center, :center))
    end
end

outpath = joinpath(plot_dir, "psi_contour.png")
save(outpath, fig; px_per_unit=2)
println("saved → $outpath")

# =========================================================
# load χ and ε — time-averaged over last avg_window time units
# reads from oceanostics segment files
# =========================================================
println("loading χ and ε from oceanostics segments...")
chi_mean, eps_mean = let
    chi_sum    = zeros(Float64, Nx, Nz)
    eps_sum    = zeros(Float64, Nx, Nz)
    n_oce      = 0
    t_last_oce = -Inf

    for s in seg_range
        ofile = joinpath(data_dir, "oceanostics_seg$(s).nc")
        isfile(ofile) || continue
        ds_o  = NCDataset(ofile)
        t_seg = Float64.(ds_o["time"][:])
        valid = findall(t_seg .> t_last_oce)
        isempty(valid) && (close(ds_o); continue)

        t_range    = valid[1]:valid[end]
        t_last_oce = t_seg[t_range[end]]

        in_win = findall(t_seg[t_range] .>= t_end - avg_window)
        if !isempty(in_win)
            t_win     = t_range[in_win[1]:in_win[end]]
            chi_chunk = Float64.(ds_o["χ"][:, 1, :, t_win])   # [Nx, Nz, n]
            eps_chunk = Float64.(ds_o["ε"][:, 1, :, t_win])   # [Nx, Nz, n]
            chi_sum  .+= dropdims(sum(chi_chunk, dims=3), dims=3)
            eps_sum  .+= dropdims(sum(eps_chunk, dims=3), dims=3)
            n_oce    += length(t_win)
        end
        close(ds_o)
    end

    n_oce == 0 && error("no oceanostics time steps found in averaging window (t ≥ $(t_end - avg_window))")
    @printf("χ,ε: averaged over %d steps\n", n_oce)

    Float32.(chi_sum ./ n_oce), Float32.(eps_sum ./ n_oce)
end

# log10; set non-positive (dry cells / zeros) to NaN
chi_log = map(v -> v > 0 ? Float32(log10(v)) : NaN32, chi_mean)
eps_log = map(v -> v > 0 ? Float32(log10(v)) : NaN32, eps_mean)

# =========================================================
# χ figure
# =========================================================
let
    valid_chi = filter(!isnan, chi_log[:])
    clim_lo   = quantile(valid_chi, 0.02)
    clim_hi   = quantile(valid_chi, 0.98)

    fig = Figure(size=(1100, 400))
    ax  = Axis(fig[1, 1];
        xlabel         = "x / H",
        ylabel         = "z / H",
        title          = "$(experiment)  Ra = $(Ra_str) — time-mean log₁₀(χ) with b contours (last $(avg_window) τ)",
        limits         = (x[1], x[end], z[1], 0.0),
        titlesize      = 26,
        xlabelsize     = 20,
        ylabelsize     = 20,
        xticklabelsize = 14,
        yticklabelsize = 14,
    )

    hm = heatmap!(ax, x, z, chi_log; colormap=:delta, colorrange=(clim_lo, clim_hi))
    Colorbar(fig[1, 2], hm; label="log₁₀(χ)", labelsize=18, ticklabelsize=14)

    contour!(ax, x, z, b_mean; levels=b_levels, color=:white, linewidth=0.7, labels=true, labelsize=9)

    vlines!(ax, x_plume; color=(:white, 0.85), linewidth=2.5, linestyle=:dash)
    hlines!(ax, zBL;     color=(:white, 0.85), linewidth=2.5, linestyle=:dash)
    for xb in x_sub_BL
        vlines!(ax, xb; ymax=z_frac_BL, color=(:white, 0.85), linewidth=2.0, linestyle=:dot)
    end

    if numhill > 0
        dry_mask = fill(NaN, Nx, Nz)
        for i in 1:Nx, k in 1:Nz
            z[k] <= z_sf[i] && (dry_mask[i, k] = 0.6)
        end
        heatmap!(ax, x, z, dry_mask; colormap=:turbid, colorrange=(0, 1))
    end

    outpath = joinpath(plot_dir, "chi_contour.png")
    save(outpath, fig; px_per_unit=2)
    println("saved → $outpath")
end

# =========================================================
# ε figure
# =========================================================
let
    valid_eps = filter(!isnan, eps_log[:])
    clim_lo   = quantile(valid_eps, 0.02)
    clim_hi   = quantile(valid_eps, 0.98)

    fig = Figure(size=(1100, 400))
    ax  = Axis(fig[1, 1];
        xlabel         = "x / H",
        ylabel         = "z / H",
        title          = "$(experiment)  Ra = $(Ra_str) — time-mean log₁₀(ε) with b contours (last $(avg_window) τ)",
        limits         = (x[1], x[end], z[1], 0.0),
        titlesize      = 26,
        xlabelsize     = 20,
        ylabelsize     = 20,
        xticklabelsize = 14,
        yticklabelsize = 14,
    )

    hm = heatmap!(ax, x, z, eps_log; colormap=:curl, colorrange=(clim_lo, clim_hi))
    Colorbar(fig[1, 2], hm; label="log₁₀(ε)", labelsize=18, ticklabelsize=14)

    contour!(ax, x, z, b_mean; levels=b_levels, color=:white, linewidth=0.7, labels=true, labelsize=9)

    vlines!(ax, x_plume; color=(:white, 0.85), linewidth=2.5, linestyle=:dash)
    hlines!(ax, zBL;     color=(:white, 0.85), linewidth=2.5, linestyle=:dash)
    for xb in x_sub_BL
        vlines!(ax, xb; ymax=z_frac_BL, color=(:white, 0.85), linewidth=2.0, linestyle=:dot)
    end

    if numhill > 0
        dry_mask = fill(NaN, Nx, Nz)
        for i in 1:Nx, k in 1:Nz
            z[k] <= z_sf[i] && (dry_mask[i, k] = 0.6)
        end
        heatmap!(ax, x, z, dry_mask; colormap=:turbid, colorrange=(0, 1))
    end

    outpath = joinpath(plot_dir, "epsilon_contour.png")
    save(outpath, fig; px_per_unit=2)
    println("saved → $outpath")
end

# =========================================================
# G_mix(b) time-mean per region — Control vs 3-hill
# =========================================================
let
    ctrl_file = joinpath(ctrl_data_dir, "Gmix_regions_Control_RA1e8_seg1to12.nc")
    hill_file = joinpath(data_dir, "Gmix_regions_v2_RA1e8_seg1to14.nc")

    ds_ctrl = NCDataset(ctrl_file)
    ds_hill = NCDataset(hill_file)

    b_out      = Float64.(ds_ctrl["b"][:])       # buoyancy bins [n_b]
    n_b        = length(b_out)

    t_ctrl = Float64.(ds_ctrl["time"][:])
    t_hill = Float64.(ds_hill["time"][:])

    # truncate hill to the same time span as control
    t_common   = min(t_ctrl[end], t_hill[end])
    i_hill_end = searchsortedlast(t_hill, t_common)
    t_hill_use = t_hill[1:i_hill_end]
    @printf("G_mix figure: control t=%.1f–%.1f (%d steps), hill truncated to t=%.1f–%.1f (%d steps)\n",
            t_ctrl[1], t_ctrl[end], length(t_ctrl),
            t_hill[1], t_hill_use[end], i_hill_end)

    region_keys = ["Gmix_plume", "Gmix_boundary_layer", "Gmix_basin0",
                   "Gmix_hill1",  "Gmix_basin1",
                   "Gmix_hill2",  "Gmix_basin2",
                   "Gmix_hill3",  "Gmix_basin3"]
    region_labels = ["Plume", "Boundary Layer", "Basin 0",
                     "Hill 1", "Basin 1",
                     "Hill 2", "Basin 2",
                     "Hill 3", "Basin 3"]

    fig9 = Figure(size=(1200, 1100))

    for (idx, (rk, rl)) in enumerate(zip(region_keys, region_labels))
        row = (idx - 1) ÷ 3 + 1
        col = (idx - 1) % 3 + 1
        ax  = Axis(fig9[row, col];
            xlabel = "mean G_mix density",
            ylabel = "b",
            title  = rl,
        )

        # reshape flat storage → (n_b, n_time) then time-average
        g_ctrl_flat = Float64.(ds_ctrl[rk][:])
        g_hill_flat = Float64.(ds_hill[rk][:])

        n_t_ctrl = length(t_ctrl)
        n_t_hill = length(t_hill)

        g_ctrl_mat = reshape(g_ctrl_flat, n_b, n_t_ctrl)   # (n_b, n_time_ctrl)
        g_hill_mat = reshape(g_hill_flat, n_b, n_t_hill)    # (n_b, n_time_hill)

        g_ctrl_mean = vec(mean(g_ctrl_mat,              dims=2))
        g_hill_mean = vec(mean(g_hill_mat[:, 1:i_hill_end], dims=2))

        lines!(ax, g_ctrl_mean, b_out; label="Control", color=:steelblue,  linewidth=1.5)
        lines!(ax, g_hill_mean, b_out; label="3-hill",  color=:orangered,  linewidth=1.5)
    end

    close(ds_ctrl)
    close(ds_hill)

    # shared legend in an empty slot (row=4, col=2 centred)
    Legend(fig9[4, 1:3], [LineElement(color=:steelblue, linewidth=1.5),
                           LineElement(color=:orangered,  linewidth=1.5)],
           ["Control", "3-hill"]; orientation=:horizontal, tellwidth=false)

    outpath = joinpath(plot_dir, "Gmix_regions_time_mean.png")
    save(outpath, fig9; px_per_unit=2)
    println("saved → $outpath")
end

# =========================================================
# Volume-averaged χ and ε by region: Plume / BL / Interior
# Uses chi_mean and eps_mean already averaged over last avg_window τ
# =========================================================
let
    # cell volumes [Nx, Nz]  (Δy factors cancel in volume-avg ratios)
    ΔV  = Δx_vec .* Δz_vec'

    # wet mask from seafloor profile: cell (i,k) is fluid if z[k] > z_sf[i]
    wet = [z[k] > z_sf[i] for i in 1:Nx, k in 1:Nz]

    # 2D coordinate arrays [Nx, Nz]
    X2d = repeat(x,  1,  Nz)
    Z2d = repeat(z', Nx, 1)

    mask_plume = X2d .< x_plume
    mask_bl    = (Z2d .> zBL) .& (X2d .>= x_plume)
    mask_int   = .!(mask_plume .| mask_bl)   # everything below BL excluding plume

    regions = [("Plume", mask_plume), ("Boundary Layer", mask_bl), ("Interior", mask_int)]
    colors  = [:steelblue, :orangered, :seagreen]

    # domain-wide volume integrals of χ and ε (wet cells only)
    chi_total = sum(chi_mean[wet] .* ΔV[wet])
    eps_total = sum(eps_mean[wet] .* ΔV[wet])

    chi_vol_avgs = Float64[]
    eps_vol_avgs = Float64[]
    chi_fracs    = Float64[]
    eps_fracs    = Float64[]
    rlabels      = String[]

    println("\nVolume-averaged χ and ε by region (last $(avg_window) τ):")
    println("─"^70)
    for (label, mask) in regions
        m       = mask .& wet
        V_r     = sum(ΔV[m])
        chi_int = sum(chi_mean[m] .* ΔV[m])
        eps_int = sum(eps_mean[m] .* ΔV[m])
        push!(rlabels,      label)
        push!(chi_vol_avgs, chi_int / V_r)
        push!(eps_vol_avgs, eps_int / V_r)
        push!(chi_fracs,    chi_int / chi_total)
        push!(eps_fracs,    eps_int / eps_total)
        @printf("%-20s  <χ> = %.3e  <ε> = %.3e  frac_χ = %.3f  frac_ε = %.3f\n",
                label, chi_int/V_r, eps_int/V_r,
                chi_int/chi_total, eps_int/eps_total)
    end

    xs = 1:3

    fig = Figure(size=(900, 700))

    ax1 = Axis(fig[1, 1];
        title  = "Volume-averaged χ  (last $(avg_window) τ)",
        ylabel = "⟨χ⟩",
        xticks = (xs, rlabels),
        xticklabelrotation = π/6,
    )
    barplot!(ax1, xs, chi_vol_avgs; color=colors)

    ax2 = Axis(fig[1, 2];
        title  = "Volume-averaged ε  (last $(avg_window) τ)",
        ylabel = "⟨ε⟩",
        xticks = (xs, rlabels),
        xticklabelrotation = π/6,
    )
    barplot!(ax2, xs, eps_vol_avgs; color=colors)

    ax3 = Axis(fig[2, 1];
        title  = "Fraction of total ∫χ dV",
        ylabel = "fraction",
        xticks = (xs, rlabels),
        xticklabelrotation = π/6,
        limits = (nothing, (0, 1)),
    )
    barplot!(ax3, xs, chi_fracs; color=colors)

    ax4 = Axis(fig[2, 2];
        title  = "Fraction of total ∫ε dV",
        ylabel = "fraction",
        xticks = (xs, rlabels),
        xticklabelrotation = π/6,
        limits = (nothing, (0, 1)),
    )
    barplot!(ax4, xs, eps_fracs; color=colors)

    outpath = joinpath(plot_dir, "chi_eps_region_stats.png")
    save(outpath, fig; px_per_unit=2)
    println("saved → $outpath")
end

# =========================================================
# ψ(x, b) heatmap — Control vs 3-hill, shared time window
# =========================================================
let
    ctrl_psib_file = joinpath(ctrl_data_dir, "psi_b_Control_RA1e8_seg1to12.nc")
    hill_psib_file = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/psi_b_t385.nc"

    ds_ctrl = NCDataset(ctrl_psib_file)
    ds_hill = NCDataset(hill_psib_file)

    t_ctrl = Float64.(ds_ctrl["time"][:])
    t_hill = Float64.(ds_hill["time"][:])

    t_avg_end   = t_end               # same anchor used for ψ/b/χ/ε contours above
    t_avg_start = t_avg_end - avg_window
    @printf("ψ(x,b): averaging over t = %.1f → %.1f\n", t_avg_start, t_avg_end)

    x_ctrl = Float64.(ds_ctrl["x"][:])
    b_ctrl = Float64.(ds_ctrl["b"][:])
    x_hill = Float64.(ds_hill["x"][:])
    b_hill = Float64.(ds_hill["b_out"][:])

    i_ctrl = searchsortedfirst(t_ctrl, t_avg_start):searchsortedlast(t_ctrl, t_avg_end)
    i_hill = searchsortedfirst(t_hill, t_avg_start):searchsortedlast(t_hill, t_avg_end)

    ψ_ctrl = dropdims(mean(Float32.(ds_ctrl["psi_b"][:, :, i_ctrl]), dims=3), dims=3)  # [Nx, Nb]
    ψ_hill = dropdims(mean(Float32.(ds_hill["ψ_b"][:, :, i_hill]),  dims=3), dims=3)  # [Nx, Nb]

    close(ds_ctrl); close(ds_hill)
    @printf("ctrl: %d steps averaged,  hill: %d steps averaged\n", length(i_ctrl), length(i_hill))

    # shared symmetric colorrange
    ψ_lim = max(maximum(abs.(ψ_ctrl)), maximum(abs.(ψ_hill)))

    fig = Figure(size=(1400, 500))

    ax1 = Axis(fig[1, 1];
        xlabel = "x / H",
        ylabel = "b",
        title  = "Control — ⟨ψ(x,b)⟩  (t = $(t_avg_start) → $(t_avg_end))",
        titlesize = 20,
    )
    hm = heatmap!(ax1, x_ctrl, b_ctrl, ψ_ctrl; colormap=:balance, colorrange=(-ψ_lim, ψ_lim))

    ax2 = Axis(fig[1, 2];
        xlabel = "x / H",
        ylabel = "b",
        title  = "3-hill — ⟨ψ(x,b)⟩  (t = $(t_avg_start) → $(t_avg_end))",
        titlesize = 20,
    )
    heatmap!(ax2, x_hill, b_hill, ψ_hill; colormap=:balance, colorrange=(-ψ_lim, ψ_lim))

    Colorbar(fig[1, 3], hm; label="ψ(x, b)")

    outpath = joinpath(plot_dir, "psi_b_heatmap_comparison.png")
    save(outpath, fig; px_per_unit=2)
    println("saved → $outpath")
end
