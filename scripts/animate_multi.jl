using Printf
using NCDatasets
using CairoMakie

function make_animation(input_path, output_path)
    println("Processing: $input_path")
    
    ds = NCDataset(input_path)
    b = ds["b"]
    time = ds["time"]
    x = ds["x_caa"][:]
    y = ds["y_aca"][:]
    z = ds["z_aac"][:]
    Ra = ds.attrib["Ra"]
    H  = ds.attrib["H"]
    Lx = ds.attrib["Lx"]
    Ly = ds.attrib["Ly"]
    Ny = ds.attrib["Ny"]

    n = Observable(1)
    mid_Ny = div(Ny, 2)

    b_ref = Array(b[:, mid_Ny, :, 2]) #im using any time index that is not the initial in case theres no cold start
    wet = b_ref .!= 0.0  # bool array : true = fluid, false = hills # size = Nx, Nz
    wet_masked = Float64.(copy(wet))
    wet_masked[wet] .= NaN

    title = @lift @sprintf("buoyancy [m/s²] Ra = %.2e, t = %.2f", Ra, time[$n])
    bₙ    = @lift Array(b[:, mid_Ny, :, $n])

    axis_kwargs = (xlabel    = L"x / H",
                   ylabel    = L"z / H",
                   limits    = ((-Lx/2, Lx/2), (-H, 0)),
                   aspect    = Lx / H,
                   titlesize = 20)

    fig   = Figure(size=(800, 600))
    ax_B  = Axis(fig[1, 1]; title=title, axis_kwargs...)
    B_lims = (-1.0, 1.0)
    hm_B  = heatmap!(ax_B, x, z, bₙ; colorrange=B_lims, colormap=:balance)
    hm_hill = heatmap!(ax_B, x, z, wet_masked[:, :], colormap=:turbid)
    Colorbar(fig[1, 2], hm_B)

    frames = 1:length(time)
    record(fig, output_path, frames, framerate=20) do i
        n[] = i
    end

    close(ds)
    println("Saved: $output_path")
end

# define all nine simulations as (input, output) pairs
base_in  = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU_test/"
base_out = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/animations/GPU/"

simulations = [
    ("cheb_2x_stretch/b_base/Ra1e7/512_64/buoyancy.nc",  "cheb_2x_stretch/b_base/512x64/Ra1e7.mp4"),
    ("cheb_2x_stretch/b_base/Ra1e8/512_64/buoyancy.nc",  "cheb_2x_stretch/b_base/512x64/Ra1e8.mp4"),
    ("cheb_2x_stretch/b_base/Ra1e9/512_64/buoyancy.nc",  "cheb_2x_stretch/b_base/512x64/Ra1e9.mp4"),
    ("cheb_2x_stretch/b_base/Ra1e10/512_64/buoyancy.nc", "cheb_2x_stretch/b_base/512x64/Ra1e10.mp4"),
    ("cheb_5x_stretch/b_base/Ra1e7/512_64/buoyancy.nc", "cheb_5x_stretch/b_base/512x64/Ra1e7.mp4"),
    ("cheb_5x_stretch/b_base/Ra1e8/512_64/buoyancy.nc", "cheb_5x_stretch/b_base/512x64/Ra1e8.mp4"),
    # ("cheb_5x_stretch/b_base/Ra1e9/512_64/buoyancy.nc", "cheb_5x_stretch/b_base/512x64/Ra1e9.mp4"),
    ("cheb_5x_stretch/b_base/Ra1e10/512_64/buoyancy.nc", "cheb_5x_stretch/b_base/512x64/Ra1e10.mp4"),
    ("cheb_8x_stretch/b_base/Ra1e7/512_64/buoyancy.nc", "cheb_8x_stretch/b_base/512x64/Ra1e7.mp4"),
    ("cheb_8x_stretch/b_base/Ra1e8/512_64/buoyancy.nc", "cheb_8x_stretch/b_base/512x64/Ra1e8.mp4"),
    # ("cheb_8x_stretch/b_base/Ra1e9/512_64/buoyancy.nc", "cheb_8x_stretch/b_base/512x64/Ra1e9.mp4"),
    ("cheb_8x_stretch/b_base/Ra1e10/512_64/buoyancy.nc", "cheb_8x_stretch/b_base/512x64/Ra1e10.mp4")
    # ... add remaining sims
]

for (rel_in, fname_out) in simulations
    input_path  = joinpath(base_in, rel_in)
    output_path = joinpath(base_out, fname_out)
    mkpath(dirname(output_path))
    make_animation(input_path, output_path)
end