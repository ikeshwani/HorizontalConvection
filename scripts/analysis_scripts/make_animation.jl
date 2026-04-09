using Printf
using NCDatasets
using CairoMakie
using Observables
using JLD2

# looking at checkpoint files and seeing the last time 

# for f in [
#     "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/chapter1/RA1e10/4x_stretch/512_128/turbulentnothingnothing_summeronly_flat_0.0_Ra1.0e10_zerostart_checkpoint_iteration4038.jld2",
#     "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/chapter1/RA1e10/4x_stretch/512_128/turbulentnothingnothing_summeronly_flat_0.0_Ra1.0e10_zerostart_checkpoint_iteration4218.jld2",
#     "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/chapter1/RA1e10/4x_stretch/512_128/turbulentnothingnothing_summeronly_flat_0.0_Ra1.0e10_zerostart_checkpoint_iteration4426.jld2",
#     "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/chapter1/RA1e10/4x_stretch/512_128/turbulentnothingnothing_summeronly_flat_0.0_Ra1.0e10_zerostart_checkpoint_iteration4632.jld2", 
#     "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/chapter1/RA1e10/4x_stretch/512_128/turbulentnothingnothing_summeronly_flat_0.0_Ra1.0e10_zerostart_checkpoint_iteration4632.jld2"
#           ]

#     jldopen(f, "r") do file
#         clock = file["NonhydrostaticModel/clock"]
#         println("t = ", clock.time)
#     end
# end


ds = NCDataset(
    "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/chapter1/RA1e10/4x_stretch/512_128/buoyancy_combined.nc",
    "r"
)

# # ds_v = NCDataset(
# #     "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/chapter1/4x_stretch/512_128/velocities.nc", 
# #     "r"
# # )

b = ds["b"]                # DO NOT slice here
# # u = ds_v["u"]
# # v = ds_v["v"]
# # w = ds_v["w"]
# time = ds["time"][:]
# print(time[1], time[end])

x = ds["x_caa"][:]
y = ds["y_aca"][:]
z = ds["z_aac"][:]

# #Metadata

Ra = ds.attrib["Ra"]
H  = ds.attrib["H"]
Lx = ds.attrib["Lx"]
Ly = ds.attrib["Ly"]
Nx = ds.attrib["Nx"]
Ny = ds.attrib["Ny"]

Nt = length(time)
yidx = Int(Ny ÷ 2)
xidx = Int(1) #we want the b(y,z) to be sliced at the left x boundary 

@info "Creating animation for Ra = $Ra with $Nt frames"

n = Observable(1)

title_bxz = @lift @sprintf(
    "buoyancy on x-z plane [m/s²], Ra = %.2e, t = %.2f",
    Ra, time[$n]
)

title_byz = @lift @sprintf(
    "buoyancy on y-z plane [m/s²], Ra = %.2e, t = %.2f", 
    Ra, time[$n]
)

# Lazy read: one (x,z) slice at one time index
b_xzₙ = @lift begin
    Array(b[:, yidx, :, $n])
end

b_yzₙ = @lift begin
    Array(b[xidx, :, :, $n])
end

# # Uₙ = @lift begin
# #     u_slice = Array(u[1:512, yidx, 1:64, $n])
# #     v_slice = Array(v[1:512, yidx, 1:64, $n])
# #     w_slice = Array(w[1:512, yidx, 1:64, $n])
# #     sqrt.(u_slice.^2 .+ v_slice.^2 .+ w_slice.^2)
# # end

# #the size of Uₙ is Nx, Nz

# #velocity magnitude

b_ref = Array(b[:, yidx, :, 2]) #im using any time index that is not the initial in case theres no cold start

b_ref_yz = Array(b[xidx, :, :, 2]) #second ref for yz plane example
wet_xz = b_ref .!= 0.0  # bool array : true = fluid, false = hills # size = Nx, Nz
wet_yz = b_ref_yz .!= 0.0 #bool array : true = fluid, false = hiills # size = Ny, Nz

# println("size of wet mask", size(wet_xz))

wet_masked_xz = Float64.(copy(wet_xz))
wet_masked_xz[wet_xz] .= NaN

wet_masked_yz = Float64.(copy(wet_yz))
wet_masked_yz[wet_yz] .= NaN

fig = Figure(size = (800, 1200))

ax_bxz = Axis(
    fig[1, 1];
    title = title_bxz,
    xlabel = L"x / Lx",
    ylabel = L"z / H",
    limits = ((-Lx/2, Lx/2), (-H, 0)),
    aspect = Lx / H,
    titlesize = 20
)

ax_byz = Axis(
    fig[2, 1];
    title = title_byz, 
    xlabel = L"y / Ly",
    ylabel = L"z / H",
    limits = ((-Lx/2, Lx/2), (-H, 0)),
    aspect = Lx / H,
    titlesize = 20
)

B_lims = (-1.0, 1.0)

hm_1 = heatmap!(
    ax_bxz, x, z, b_xzₙ;
    colormap = :balance,
    colorrange = B_lims
)

hm_hill = heatmap!(
    ax_bxz, x, z, wet_masked_xz[:, :], colormap=:turbid
)

Colorbar(fig[1, 2], hm_1)

# U_lims = (0.0, 0.4)

hm_2 = heatmap!(
    ax_byz, y, z, b_yzₙ;
    colormap = :balance, 
    colorrange = B_lims, 
)

hm_hill = heatmap!(
    ax_byz, y, z, wet_masked_yz[:, :], colormap=:turbid
)

Colorbar(fig[2, 2], hm_2)

frames = 1:Nt   # or 1:5:Nt to subsample

output_file = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/animations/GPU/chapter1/RA1e10/4x_stretch/first28sec.mp4"

record(fig, output_file, frames; framerate = 16) do i
    n[] = i
end

close(ds)

@info "Animation saved to $output_file"
