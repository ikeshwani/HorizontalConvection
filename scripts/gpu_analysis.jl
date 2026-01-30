using Oceananigans
using CairoMakie
using NCDatasets
using Statistics
using Printf

output_dir =  "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/test/Ra1e6/320_40/"

buoy_file = joinpath(output_dir, "buoyancy.nc")
oc_file = joinpath(output_dir, "oceanostics.nc")
vel_file = joinpath(output_dir, "velocities.nc")

ds_b = NCDataset(buoy_file, "r")
ds_o = NCDataset(oc_file, "r")
# ds_v = NCDataset(vel_file, "r")

# b★ = ds_b.attrib["b★"]
# Lx = ds_b.attrib["Lx"]
# H = ds_b.attrib["H"]
ν = ds_b.attrib["ν"]

b = ds_b["b"][:, :, :, :] # DONT LOAD YET THE FILE IS TOO BIG
x = ds_b["x_caa"][:]
# y = ds_b["y_aca"][:]
# z = ds_b["z_aac"][:]
time = ds_b["time"][:]

println("size of b data : ", size(b))

# Δx = minimum(diff(x))

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






