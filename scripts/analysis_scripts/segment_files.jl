using Printf
using NCDatasets
using CairoMakie
using Observables

# we have our data in several segments 

function main()

    basepath = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/chapter1/RA1e6/4x_stretch/512_128/"
    n_segments = 7
    
    datasets = [NCDataset(joinpath(basepath, "buoyancy_seg$(i).nc")) for i in 1:n_segments]

    # println("\n - segment time info")
    # for (i, ds) in enumerate(datasets)
    #     t = ds["time"][:]
    #     @printf(" segment %d | start: %.4f | end: %.4f | nsteps: %d\n", i, t[1], t[end], length(t))
    # end

    #read in spatial coords and global attributes from segment 1 cuz its all the same

    ds1 = datasets[1]

    x = ds1["x_caa"][:]
    y = ds1["y_aca"][:]
    z = ds1["z_aac"][:]

    Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
    Lx, Ly, H = ds1.attrib["Lx"], ds1.attrib["Ly"], ds1.attrib["H"]

    t_combined = Float64[]
    b_combined = nothing
    t_end = -Inf

    for i in 1:n_segments
        t_i = datasets[i]["time"][:]

        # find the index in the next file where time exceeds the current end
        new_idx = findfirst(t -> t > t_end, t_i)

        if isnothing(new_idx)
            @printf(" segment %d is fully overlapping - skipping entirely \n", i)
            continue
        end

        n_overlap = new_idx - 1
        @printf("segment %d : trimming %d overlapping timestep(s)\n", i, n_overlap)

        valid_range = new_idx:length(t_i)

        b_i = datasets[i]["b"][:, :, :, valid_range]
        append!(t_combined, t_i[valid_range])

        if b_combined === nothing
            b_combined = b_i
        else
            b_combined = cat(b_combined, b_i; dims=4)
        end

        t_end = t_combined[end]

        @printf(" -> kept %d steps , new t_end is = %.4f\n", length(valid_range), t_end)

    end

    println("\nCombined: $(length(t_combined)) timesteps | t_start=$(t_combined[1]) | t_end=$(t_combined[end])\n")

    return x, y, z, Lx, Ly, H, t_combined, b_combined
end

x, y, z, Lx, Ly, H, t, b = main()

Nx = length(x)
Ny = length(y)
Nz = length(z)

Nt = length(t)

yidx = Int(Ny ÷ 2)
xidx = Int(64) #we want the b(y,z) to be sliced at the left x boundary

Ra = 1e6 ############################################################## make this better bro ###########

@info "Creating animation for Ra = $Ra with $Nt frames"

n = Observable(1)

title_bxz = @lift @sprintf(
    "buoyancy on x-z plane [m/s²], Ra = %.2e, t = %.2f",
    Ra, t[$n]
)

title_byz = @lift @sprintf(
    "buoyancy on y-z plane [m/s²], Ra = %.2e, t = %.2f", 
    Ra, t[$n]
)

# Lazy read: one (x,z) slice at one time index
b_xzₙ = @lift begin
    b[:, yidx, :, $n]
end

b_yzₙ = @lift begin
    b[xidx, :, :, $n]
end

# # Uₙ = @lift begin
# #     u_slice = Array(u[1:512, yidx, 1:64, $n])
# #     v_slice = Array(v[1:512, yidx, 1:64, $n])
# #     w_slice = Array(w[1:512, yidx, 1:64, $n])
# #     sqrt.(u_slice.^2 .+ v_slice.^2 .+ w_slice.^2)
# # end

# #the size of Uₙ is Nx, Nz

# #velocity magnitude

b_ref = b[:, yidx, :, 2] #im using any time index that is not the initial in case theres no cold start

b_ref_yz = b[xidx, :, :, 2] #second ref for yz plane example
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

B_lims = (-0.6, 0.6)

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

output_file = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/animations/GPU/chapter1/RA1e6/4x_stretch/first16sec_xind64.mp4"

record(fig, output_file, frames; framerate = 16) do i
    n[] = i
end

@info "Animation saved to $output_file"

