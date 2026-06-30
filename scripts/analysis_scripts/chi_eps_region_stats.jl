using TopographicHorizontalConvection   # physics: seafloor_profile, boundary_layer_depth, load_chi_eps_mean
using NCDatasets, CairoMakie, Statistics, Printf

avg_window = 30.0
GRC        = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC"
plot_dir   = joinpath(GRC, "ra1e8_4xstretch_threehill_baseforcing_zerostart", "figures")
mkpath(plot_dir)

experiments = [
    (label     = "Control",
     data_dir  = joinpath(GRC, "ra1e8_4xstretch_flat_baseforcing_zerostart"),
     seg_range = 1:12,
     numhill   = 0,
     h₀_frac   = 0.0),
    (label     = "3-hill",
     data_dir  = joinpath(GRC, "ra1e8_4xstretch_threehill_baseforcing_zerostart"),
     seg_range = 1:20,
     numhill   = 3,
     h₀_frac   = 0.5),
]

# ── t_end taken from control (first experiment) so all windows align ──────────
t_end = let ctrl = experiments[1]
    t = -Inf
    for s in reverse(ctrl.seg_range)
        ofile = joinpath(ctrl.data_dir, "oceanostics_seg$(s).nc")
        isfile(ofile) || continue
        ds = NCDataset(ofile)
        t  = Float64(ds["time"][end])
        close(ds)
        break
    end
    t
end
@printf("shared averaging window: t = %.1f → %.1f  (from Control)\n", t_end - avg_window, t_end)

# ── compute stats per experiment ──────────────────────────────────────────────
results = []

for exp in experiments
    println("\n=== $(exp.label) ===")
    ds1    = NCDataset(joinpath(exp.data_dir, "buoyancy_seg1.nc"))
    x      = Float64.(ds1["x_caa"][:])
    z      = Float64.(ds1["z_aac"][:])
    Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
    Lx     = Float64(ds1.attrib["Lx"])
    H      = Float64(ds1.attrib["H"])
    Ra     = Float64(ds1.attrib["Ra"])
    Δx_vec = Float64.(ds1["Δx_caa"][:])
    Δy_vec = Float64.(ds1["Δy_aca"][:])
    Δz_vec = Float64.(ds1["Δz_aac"][:])
    close(ds1)

    zBL     = boundary_layer_depth(Lx, Ra)
    x_plume = -1.8
    z_sf    = seafloor_profile(x, H, Lx, exp.h₀_frac, exp.numhill)

    @printf("  t_end = %.1f,  zBL = %.3f\n", t_end, zBL)

    chi_mean, eps_mean = load_chi_eps_mean(
        exp.data_dir, exp.seg_range, avg_window, t_end, Nx, Ny, Nz)

    # ΔV[i,j,k] = Δx[i] * Δy[j] * Δz[k]  — full 3D cell volume
    ΔV = reshape(Δx_vec, Nx, 1, 1) .* reshape(Δy_vec, 1, Ny, 1) .* reshape(Δz_vec, 1, 1, Nz)

    # seafloor mask in x-z, then broadcast over y
    wet_2d = [z[k] > z_sf[i] for i in 1:Nx, k in 1:Nz]
    wet    = repeat(reshape(wet_2d, Nx, 1, Nz), 1, Ny, 1)

    # region masks defined in x-z space, broadcast over y
    X2d = repeat(x,            1,  Nz)
    Z2d = repeat(transpose(z), Nx, 1)
    mask_plume_2d = X2d .< x_plume
    mask_bl_2d    = (Z2d .> zBL) .& (X2d .>= x_plume)
    mask_int_2d   = .!(mask_plume_2d .| mask_bl_2d)

    mask_plume = repeat(reshape(mask_plume_2d, Nx, 1, Nz), 1, Ny, 1)
    mask_bl    = repeat(reshape(mask_bl_2d,    Nx, 1, Nz), 1, Ny, 1)
    mask_int   = repeat(reshape(mask_int_2d,   Nx, 1, Nz), 1, Ny, 1)

    chi_total = sum(chi_mean[wet] .* ΔV[wet])
    eps_total = sum(eps_mean[wet] .* ΔV[wet])

    chi_avgs  = Float64[]; eps_avgs  = Float64[]
    chi_fracs = Float64[]; eps_fracs = Float64[]

    V_wet_total = sum(ΔV[wet])
    @printf("  Domain wet volume = %.4f  (total ∫χ dV = %.3e,  total ∫ε dV = %.3e)\n",
            V_wet_total, chi_total, eps_total)
    println("  Region               V_wet    <χ>          ∫χ dV       <ε>          ∫ε dV       frac_χ  frac_ε")
    println("  ", "─"^95)
    for (lbl, mask) in [("Plume", mask_plume), ("Boundary Layer", mask_bl), ("Interior", mask_int)]
        m       = mask .& wet
        V_r     = sum(ΔV[m])
        chi_int = sum(chi_mean[m] .* ΔV[m])
        eps_int = sum(eps_mean[m] .* ΔV[m])
        push!(chi_avgs,  chi_int / V_r)
        push!(eps_avgs,  eps_int / V_r)
        push!(chi_fracs, chi_int / chi_total)
        push!(eps_fracs, eps_int / eps_total)
        @printf("  %-18s  %.4f  %.3e  %.3e  %.3e  %.3e  %.3f   %.3f\n",
                lbl, V_r, chi_int/V_r, chi_int, eps_int/V_r, eps_int,
                chi_int/chi_total, eps_int/eps_total)
    end

    push!(results, (label=exp.label, chi_avgs=chi_avgs, eps_avgs=eps_avgs,
                    chi_fracs=chi_fracs, eps_fracs=eps_fracs))
end

# ── figure ────────────────────────────────────────────────────────────────────
rlabels = ["Plume", "Boundary\nLayer", "Interior"]
xs      = 1:3
colors  = [:steelblue, :orangered]

dodge(xs, i, n=2) = xs .+ (i - (n + 1) / 2) .* 0.35

fig = Figure(size=(1100, 750))

panel_specs = [
    (1, 1, "Volume-averaged χ  (last $(avg_window) τ)",  "⟨χ⟩",     :chi_avgs,  false),
    (1, 2, "Volume-averaged ε  (last $(avg_window) τ)",  "⟨ε⟩",     :eps_avgs,  false),
    (2, 1, "Fraction of total ∫χ dV (last $(avg_window) τ)",                   "fraction", :chi_fracs, true),
    (2, 2, "Fraction of total ∫ε dV (last $(avg_window) τ) ",                   "fraction", :eps_fracs, true),
]

for (row, col, title, ylabel, field, clamp_ylim) in panel_specs
    ax = Axis(fig[row, col];
        title  = title,
        ylabel = ylabel,
        xticks = (collect(xs), rlabels),
    )
    clamp_ylim && ylims!(ax, 0, 1)
    for (i, res) in enumerate(results)
        vals = getfield(res, field)
        barplot!(ax, dodge(collect(xs), i), vals;
                 width=0.32, color=(colors[i], 0.85), label=res.label)
    end
end

Legend(fig[3, 1:2],
       [PolyElement(color=(c, 0.85)) for c in colors],
       [r.label for r in results];
       orientation=:horizontal, tellwidth=false)

outpath = joinpath(plot_dir, "chi_eps_region_stats_comparison.png")
save(outpath, fig; px_per_unit=2)
println("\nsaved → $outpath")
