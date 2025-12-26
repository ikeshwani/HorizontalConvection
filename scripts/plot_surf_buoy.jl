using NCDatasets
using CairoMakie
using Printf

output_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/CPU_test/bss_coldstart/256_32/"
buoy_file = joinpath(output_dir, "buoyancy.nc")

ds_b = NCDataset(buoy_file)
b★ = ds_b.attrib["b★"]
Lx = ds_b.attrib["Lx"]
H = ds_b.attrib["H"]
seasonal_amplitude = 0.0

plot_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/CPU_test/base_turb/verification_plots"
mkpath(plot_dir)

function expected_surface_bc(x, t, b★, Lx)
    bss = tanh(3 * (x+Lx/3))
    return b★ * bss
end

println("loading data from : $buoy_file")

b = ds_b["b"][:, :, :, :]
x = ds_b["x_caa"][:]
y = ds_b["y_aca"][:]
z = ds_b["z_aac"][:]
time = ds_b["time"][:]

if ndims(b) == 4
    b_surface = b[:, 1, end, :] # y=1 z=end=top
elseif ndims(b) == 3 #2d sim
    b_surface = b[:, end, :] #no y dim
end

Nx = ds_b.attrib["Nx"]
Ny = ds_b.attrib["Ny"]
Nz = ds_b.attrib["Nz"]
Nt = length(time)

println("Loaded data:")
println("  Nx = $Nx")
println("  Nt = $Nt")
println("  Time range: $(time[1]) to $(time[end])")
println("  x range: $(x[1]) to $(x[end])")
println()

println("creating plot 1: surface buoyancy versus x at diff times")

fig1 = Figure(size=(1000,600))
ax1 = Axis(fig1[1,1], 
    xlabel="x", 
    ylabel="Surface Buoyancy", 
    title = "Surface Buoyancy Evolution Over Time")

n_snaps = min(5, Nt)
time_ind = round.(Int, range(1, Nt, length=n_snaps))

colors = [:blue, :green, :orange, :red, :purple]
for (i, tidx) in enumerate(time_ind)
    lines!(ax1, x, b_surface[:, tidx],
        label = @sprintf("t=%.2f", time[tidx]),
        color = colors[i],
        linewidth=2
    )
end

hlines!(ax1, [0], color=:black, linestyle=:dash, linewidth=1, label="b=0")
hlines!(ax1, [-b★, b★], color=:gray, linestyle=:dot, linewidth=1)

axislegend(ax1, position=:lt)

save(joinpath(plot_dir, "1_surface_b_vs_x.png"), fig1)
println(" saved 1_surface_b_vs_x.png")


println("creating plot to compare actual surface buoyancy vs expected")

fig2 = Figure(size=(1200,800))

n_comparison = min(4, Nt)
comparison_ind = round.(Int, range(1, Nt, length=n_comparison))

for (plot_idx, tidx) in enumerate(comparison_ind)
    row = ((plot_idx -1) ÷ 2) + 1
    col = ((plot_idx -1) % 2) + 1

    ax = Axis(fig2[row,col], 
            xlabel = "x",
            ylabel = "Surface Buoyancy", 
            title = @sprintf("t = %.2f", time[tidx]))

    b_actual = b_surface[:, tidx]

    b_expected = [expected_surface_bc(xi, time[tidx], b★, Lx)
                    for xi in x]

    lines!(ax, x, b_actual, label="actual (simulation)",
            color=:blue, linewidth=2)

    lines!(ax, x, b_expected, label="expected (BC function)",
            color=:red, linestyle=:dash, linewidth=2)
    axislegend(ax, position=:lt)
end

save(joinpath(plot_dir, "2_actual_vs_expected.png"), fig2)
println(" saved: 2_actual_vs_expected.png")


