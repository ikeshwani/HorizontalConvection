using Oceananigans
using CairoMakie
using NCDatasets
using Statistics
using Printf

output_dir =  "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/chebytest/b_base/Ra1e6/256_32/"

buoy_file = joinpath(output_dir, "buoyancy.nc")
#oc_file = joinpath(output_dir, "oceanostics.nc")
#vel_file = joinpath(output_dir, "velocities.nc")

ds_b = NCDataset(buoy_file, "r")
#ds_o = NCDataset(oc_file, "r")
# ds_v = NCDataset(vel_file, "r")

# b★ = ds_b.attrib["b★"]
# Lx = ds_b.attrib["Lx"]
# H = ds_b.attrib["H"]
#ν = ds_b.attrib["ν"]

println("loading data from : $buoy_file")

b = ds_b["b"][:, :, :, :]
x = ds_b["x_caa"][:]
y = ds_b["y_aca"][:]
z = ds_b["z_aac"][:]
time = ds_b["time"][:]

println("size of b data : ", size(b))

Δx = ds_b["Δx_caa"][:]
Δz = ds_b["Δz_aac"][:]

println("first few x spacings : ", Δx[1:5])
println("last few x spacings : ", Δx[end-4:end])

println("\n first few z spacings : ", Δz[1:5])
println("last few z spacings : ", Δz[end-4:end])

plot_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU_test/cheby_grid_verification/"
mkpath(plot_dir)

println("creating plot 1: buoyancy heatmap with grid visualization")

n = 40
t = round(time[40])

fig = Figure(size=(1000,600))
ax = Axis(fig[1,1], xlabel = "x", ylabel="z", title = "Ra=1e6 experiment : buoyancy heatmap for variable grid at time $t")

hm = heatmap!(ax, x, z, b[:, 8, :, n])
Colorbar(fig[1,2], hm)

for i in x[1:5:end]
    vlines!(ax, x, color=:white, alpha=0.3, linewidth=0.5)
end

for j in z[1:5:end]
    hlines!(ax, z, color=:white, alpha=0.3, linewidth=0.5)
end

save(joinpath(plot_dir, "b_hm_chebygrid.png"), fig)
println("saved first figure")


println("making second figure : grid spacing size as function of index")

fig2 = Figure(size = (1000, 600))

ax1 = Axis(fig2[1,1], 
            xlabel = "grid index i", ylabel = "Δx", title="x-spacing")

ax2 = Axis(fig2[1,2], 
           xlabel = "grid index k", ylabel = "Δz", title="z-spacing" 
)

lines!(ax1, 1:length(Δx), Δx)
lines!(ax2, 1:length(Δz), Δz)

save(joinpath(plot_dir, "xz_chebyspacing.png"), fig2)

println("saved second figure")

close(ds_b)

# ε = ds_o["ε"]
# χ = ds_o["χ"]

# function is_kolmogorov_resolved_from_ε_max(Δx, ε_max, ν)
#     η_min = (ν^3 / ε_max)^(1/4)
#     return Δx < η_min, η_min
# end

# function compute_ε_max(ds_o)
#     ε = ds_o["ε"]
#     Nx, Ny, Nz, Nt = size(ε)

#     ε_max = 0.0

#     for t in 1:Nt
#         for k in 1:Nz
#             ε_plane = view(ε, :, :, k, t)
#             ε_max = max(ε_max, maximum(ε_plane))
#         end
#     end

#     return ε_max
# end

# ε_max = compute_ε_max(ds_o)
# is_it, η_min = is_kolmogorov_resolved_from_ε_max(Δx, ε_max, ν)


# println("kolmogorov scale resolved for the Ra1e8 simulation: ", is_it)
# println("kolmogorov scale of Ra1e8 sim : ", η_min)
# # println("maximum KE dissipation : ", ε_max)
# # println()






