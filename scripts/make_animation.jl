using Printf
using NCDatasets
using CairoMakie
using Observables

ds = NCDataset(
    "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/b_winteronly/Ra1e6/320_40/buoyancy.nc",
    "r"
)

ds_v = NCDataset(
    "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/b_winteronly/Ra1e6/320_40/velocities.nc", 
    "r"
)

b = ds["b"]                # DO NOT slice here
u = ds_v["u"]
v = ds_v["v"]
w = ds_v["w"]
time = ds["time"][:]

x = ds["x_caa"][:]
y = ds["y_aca"][:]
z = ds["z_aac"][:]

#Metadata

Ra = ds.attrib["Ra"]
H  = ds.attrib["H"]
Lx = ds.attrib["Lx"]
Ly = ds.attrib["Ly"]
Ny = ds.attrib["Ny"]

Nt = length(time)
yidx = Int(Ny ÷ 2)

@info "Creating animation for Ra = $Ra with $Nt frames"

n = Observable(1)

title_b = @lift @sprintf(
    "buoyancy [m/s²], Ra = %.2e, t = %.2f",
    Ra, time[$n]
)

title_v = @lift @sprintf(
    "velocity (magnitude) [m/s], Ra = %.2e, t = %.2f", 
    Ra, time[$n]
)

# Lazy read: one (x,z) slice at one time index
bₙ = @lift begin
    Array(b[:, yidx, :, $n])
end

Uₙ = @lift begin
    u_slice = Array(u[1:320, yidx, 1:40, $n])
    v_slice = Array(v[1:320, yidx, 1:40, $n])
    w_slice = Array(w[1:320, yidx, 1:40, $n])
    sqrt.(u_slice.^2 .+ v_slice.^2 .+ w_slice.^2)
end

#the size of Uₙ is Nx, Nz

#velocity magnitude

b_ref = Array(b[:, yidx, :, 2]) #im using any time index that is not the initial in case theres no cold start
wet = b_ref .!= 0.0  # bool array : true = fluid, false = hills # size = Nx, Nz
println("size of wet mask", size(wet))

wet_masked = Float64.(copy(wet))
wet_masked[wet] .= NaN

fig = Figure(size = (800, 1200))

ax_b = Axis(
    fig[1, 1];
    title = title_b,
    xlabel = L"x / H",
    ylabel = L"z / H",
    limits = ((-Lx/2, Lx/2), (-H, 0)),
    aspect = Lx / H,
    titlesize = 20
)

ax_v = Axis(
    fig[2, 1];
    title = title_v, 
    xlabel = L"x / H",
    ylabel = L"z / H",
    limits = ((-Lx/2, Lx/2), (-H, 0)),
    aspect = Lx / H,
    titlesize = 20
)

B_lims = (-0.7, 1.0)

hm = heatmap!(
    ax_b, x, z, bₙ;
    colormap = :balance,
    colorrange = B_lims
)

hm_hill = heatmap!(
    ax_b, x, z, wet_masked[:, :], colormap=:turbid
)

Colorbar(fig[1, 2], hm)

U_lims = (0.0, 0.4)

hm_u = heatmap!(
    ax_v, x, z, Uₙ;
    colormap = :speed, 
    colorrange = U_lims, 
)

hm_hill = heatmap!(
    ax_v, x, z, wet_masked[:, :], colormap=:turbid
)

Colorbar(fig[2, 2], hm_u)

frames = 1:Nt   # or 1:5:Nt to subsample

output_file = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/animations/GPU/b_winteronly.mp4"

record(fig2, output_file, frames; framerate = 8) do i
    n[] = i
end

close(ds)

@info "Animation saved to $output_file"
