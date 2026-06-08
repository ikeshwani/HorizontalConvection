using NCDatasets
using NaNStatistics
using CairoMakie
using Statistics
using Printf

# =========================================================
# PARAMETERS — Control (flat bottom) experiment
# =========================================================
experiment  = "Control"
Ra_str      = "RA1e8"
stretch_str = "4x_stretch"
grid_str    = "512_128"
seg_range   = 1:9
avg_window  = 10.0          # time units to average at end of run

numhill     = 0
h₀_frac     = 0.0

data_dir  = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/Control/$(Ra_str)/$(stretch_str)/$(grid_str)/"
gmix_file = joinpath(data_dir, "Gmix_regions_Control_RA1e8_seg1to9.nc")
plot_dir  = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU/GRC/$(experiment)/$(Ra_str)/$(stretch_str)/figures/"
mkpath(plot_dir)

use_combined = false

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
close(ds1)

# =========================================================
# region boundaries
# =========================================================
zBL      = -(round(Lx * Ra^(-1/5); digits=2) + 0.02)
x_plume  = -1.8
x_sub_BL = [-1.35, -0.65, -0.35, 0.35, 0.65, 1.35]

z_frac_BL = (zBL - z[1]) / (z[end] - z[1])

# =========================================================
# analytic seafloor profile (flat for Control)
# =========================================================
function seafloor_profile(x, H, Lx, h₀_frac, numhill)
    h₀ = h₀_frac * H
    hl = Lx / 32
    h1 = numhill >= 1 ? h₀      .* exp.(-(x .+ Lx/4).^2 ./ (2hl^2)) : zeros(length(x))
    h2 = numhill >= 2 ? 0.75h₀  .* exp.(-(x        ).^2 ./ (2hl^2)) : zeros(length(x))
    h3 = numhill >= 3 ? 0.5h₀   .* exp.(-(x .- Lx/4).^2 ./ (2hl^2)) : zeros(length(x))
    return -H .+ h1 .+ h2 .+ h3
end
z_sf = seafloor_profile(x, H, Lx, h₀_frac, numhill)

# =========================================================
# load ψ — time-averaged over last avg_window time units
# =========================================================
println("loading ψ from Gmix file...")
ds_g   = NCDataset(gmix_file)
time_ψ = ds_g["time"][:]
t_end  = time_ψ[end]
i_start = searchsortedfirst(time_ψ, t_end - avg_window)
i_ψ     = i_start:length(time_ψ)
ψ_mean  = dropdims(mean(ds_g["psi"][:, :, i_ψ], dims=3), dims=3)   # [Nx, Nz]
close(ds_g)
@printf("ψ: averaged over %d steps  (t = %.1f → %.1f)\n", length(i_ψ), time_ψ[i_start], t_end)

# =========================================================
# load b — y- and time-averaged over last avg_window time units
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
# buoyancy contour levels
# =========================================================
b_levels = b★ .* [-1.0, -0.8, -0.75, -0.7, -0.6, -0.5, -0.25, -0.2, -0.15, -0.1, 0.1, 0.25, 0.5, 0.75, 1.0]

# =========================================================
# figure
# =========================================================
fig = Figure(size=(1100, 400))
ax  = Axis(fig[1, 1];
    xlabel = "x / H",
    ylabel = "z / H",
    title  = "$(experiment)  Ra = $(Ra_str) — time-mean ψ with b contours (last $(avg_window) τ)",
    limits = (x[1], x[end], z[1], 0.0),
)

ψ_lim = 0.004   # hardcoded so both experiments share the same colorrange
hm = heatmap!(ax, x, z, ψ_mean; colormap=:balance, colorrange=(-ψ_lim, ψ_lim))
Colorbar(fig[1, 2], hm; label="ψ")

# buoyancy contours
contour!(ax, x, z, b_mean; levels=b_levels, color=:black, linewidth=0.7, labels=true, labelsize=9)

# plume boundary — full height
vlines!(ax, x_plume; color=(:white, 0.85), linewidth=2.5, linestyle=:dash)

# BL depth — full width
hlines!(ax, zBL; color=(:white, 0.85), linewidth=2.5, linestyle=:dash)

# sub-BL region boundaries
for xb in x_sub_BL
    vlines!(ax, xb; ymax=z_frac_BL, color=(:white, 0.85), linewidth=2.0, linestyle=:dot)
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
        xlabel = "x / H",
        ylabel = "z / H",
        title  = "$(experiment)  Ra = $(Ra_str) — time-mean log₁₀(χ) with b contours (last $(avg_window) τ)",
        limits = (x[1], x[end], z[1], 0.0),
    )

    hm = heatmap!(ax, x, z, chi_log; colormap=:delta, colorrange=(clim_lo, clim_hi))
    Colorbar(fig[1, 2], hm; label="log₁₀(χ)")

    contour!(ax, x, z, b_mean; levels=b_levels, color=:black, linewidth=0.7, labels=true, labelsize=9)

    vlines!(ax, x_plume; color=(:white, 0.85), linewidth=2.5, linestyle=:dash)
    hlines!(ax, zBL;     color=(:white, 0.85), linewidth=2.5, linestyle=:dash)
    for xb in x_sub_BL
        vlines!(ax, xb; ymax=z_frac_BL, color=(:white, 0.85), linewidth=2.0, linestyle=:dot)
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
        xlabel = "x / H",
        ylabel = "z / H",
        title  = "$(experiment)  Ra = $(Ra_str) — time-mean log₁₀(ε) with b contours (last $(avg_window) τ)",
        limits = (x[1], x[end], z[1], 0.0),
    )

    hm = heatmap!(ax, x, z, eps_log; colormap=:curl, colorrange=(clim_lo, clim_hi))
    Colorbar(fig[1, 2], hm; label="log₁₀(ε)")

    contour!(ax, x, z, b_mean; levels=b_levels, color=:black, linewidth=0.7, labels=true, labelsize=9)

    vlines!(ax, x_plume; color=(:white, 0.85), linewidth=2.5, linestyle=:dash)
    hlines!(ax, zBL;     color=(:white, 0.85), linewidth=2.5, linestyle=:dash)
    for xb in x_sub_BL
        vlines!(ax, xb; ymax=z_frac_BL, color=(:white, 0.85), linewidth=2.0, linestyle=:dot)
    end

    outpath = joinpath(plot_dir, "epsilon_contour.png")
    save(outpath, fig; px_per_unit=2)
    println("saved → $outpath")
end
