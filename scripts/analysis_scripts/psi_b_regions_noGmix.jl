using NCDatasets
using CairoMakie
using Printf
using Statistics

# ---- paths ----
hill_psib_file = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/psi_b_t385.nc"
ctrl_psib_file = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/Control/RA1e8/4x_stretch/512_128/psi_b_Control_RA1e8_seg1to12.nc"
plot_dir       = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU/GRC/3hill/RA1e8/4x_stretch/figures/"
mkpath(plot_dir)

avg_window = 10.0   # time units to average at the end

# ---- region boundaries: (label, x_L, x_R) ----
regions = [
    ("Hill 1",  -1.35, -0.65),
    ("Basin 1", -0.65, -0.35),
    ("Hill 2",  -0.35,  0.35),
    ("Basin 2",  0.35,  0.65),
    ("Hill 3",   0.65,  1.35),
    ("Basin 3",  1.35,  Inf),
]

# ---- shared t_end from control ----
ds_tmp   = NCDataset(ctrl_psib_file)
t_ctrl   = Float64.(ds_tmp["time"][:])
t_end    = t_ctrl[end]
close(ds_tmp)
t_start  = t_end - avg_window
@printf("shared averaging window: t = %.2f → %.2f\n", t_start, t_end)

# ---- loader: returns (x, b, ψ_b_tavg[Nx, Nb]) ----
# handles both naming conventions:
#   control: psi_b variable, b coordinate
#   hill:    ψ_b  variable, b_out coordinate
function load_psib_tavg(file, t_start, t_end)
    ds    = NCDataset(file)
    x     = Float64.(ds["x"][:])
    b_key = haskey(ds, "b") ? "b" : "b_out"
    ψ_key = haskey(ds, "psi_b") ? "psi_b" : "ψ_b"
    b     = Float64.(ds[b_key][:])
    time  = Float64.(ds["time"][:])
    idx   = findall((time .>= t_start) .& (time .<= t_end))
    isempty(idx) && error("no time steps in window [$t_start, $t_end] in $file")
    @printf("  %s: averaging %d steps (t = %.2f → %.2f)\n",
            basename(file), length(idx), time[idx[1]], time[idx[end]])
    ψ_mean = dropdims(mean(Float64.(ds[ψ_key][:, :, idx[1]:idx[end]]), dims=3), dims=3)
    close(ds)
    return x, b, ψ_mean
end

println("loading hill psi_b...")
x_h, b_h, ψ_h = load_psib_tavg(hill_psib_file, t_start, t_end)

println("loading control psi_b...")
x_c, b_c, ψ_c = load_psib_tavg(ctrl_psib_file, t_start, t_end)

nearest_xi(x, xv) = argmin(abs.(x .- xv))

# ---- figure builder ----
function make_fig(x, b, ψ, regions, fig_title, fname)
    n   = length(regions)
    fig = Figure(size = (280 * n, 650))

    Label(fig[1, 1:n], fig_title; fontsize=24, font=:bold, tellwidth=false)

    axs = Vector{Axis}(undef, n)
    for (i, (rname, x_L, x_R)) in enumerate(regions)
        axs[i] = Axis(fig[2, i];
            title          = rname,
            titlesize      = 26,
            xlabel         = "ψ",
            ylabel         = i == 1 ? "b" : "",
            xlabelsize     = 20,
            ylabelsize     = 20,
            xticklabelsize = 14,
            yticklabelsize = 14,
        )

        iL   = nearest_xi(x, x_L)
        iR   = x_R == Inf ? length(x) : nearest_xi(x, x_R)

        ψ_in  = ψ[iL, :]
        ψ_out = ψ[iR, :]
        Δψ    = ψ_out .- ψ_in

        lines!(axs[i], ψ_in,  b; color=:royalblue,  linewidth=2.5, label="ψ_in")
        lines!(axs[i], ψ_out, b; color=:orangered,   linewidth=2.5, label="ψ_out")
        lines!(axs[i], Δψ,    b; color=:black,       linewidth=2.5,
               linestyle=:dash, label="Δψ")
        vlines!(axs[i], [0.0]; color=(:gray, 0.4), linewidth=1)
    end

    linkyaxes!(axs...)

    Legend(fig[3, 1:n],
           [LineElement(color=c, linewidth=2.5) for c in (:royalblue, :orangered, :black)],
           ["ψ_in", "ψ_out", "Δψ"];
           orientation=:horizontal, tellwidth=false, framevisible=false,
           labelsize=18, patchsize=(30, 18))

    save(joinpath(plot_dir, fname), fig; px_per_unit=2)
    println("saved → $(joinpath(plot_dir, fname))")
    return fig
end

println("plotting hill figure...")
make_fig(x_h, b_h, ψ_h, regions,
         "3-Hill  (Ra=1e8,  t = $(@sprintf("%.1f", t_start))–$(@sprintf("%.1f", t_end)))",
         "psi_b_regions_hill_noGmix.png")

println("plotting control figure...")
make_fig(x_c, b_c, ψ_c, regions,
         "Control  (Ra=1e8,  t = $(@sprintf("%.1f", t_start))–$(@sprintf("%.1f", t_end)))",
         "psi_b_regions_ctrl_noGmix.png")

println("done.")
