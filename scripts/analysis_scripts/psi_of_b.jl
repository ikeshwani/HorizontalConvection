using NCDatasets
using CairoMakie
using Printf
using Interpolations
using Statistics
using NaNStatistics

plot_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU/GRC/RA1e8/4x_stretch/figures/"

ds = NCDataset("/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/combined_t194.nc")

b = ds["b"];
χ = ds["chi"];
u = ds["u"]
x = ds["x_caa"][:]
y = ds["y_aca"][:]
z = ds["z_aac"][:]
time = ds["time"][:]

Nx, Ny, Nz = ds.attrib["Nx"], ds.attrib["Ny"], ds.attrib["Nz"]

Δx = reshape(ds["Δx_caa"][:], Nx, 1, 1)
Δy = reshape(ds["Δy_aca"][:], 1, Ny, 1)
Δz = reshape(ds["Δz_aac"][:], 1, 1, Nz)

ΔV = Δx .* Δy .* Δz;

#compute wet mask from the second time step (b=0 every intially so it would be wrong)
wet = Array(b[:, :, :, 2]) .!= 0 # size [Nx, Ny, Nz]

b_range = (nanminimum(Array(b[:, :, :, end])[wet]), nanmaximum(Array(b[:, :, :, end])[wet]))

function get_ψb(u, b, wet, ds; b_range, n_b_bins = 100)

    Nx, Ny, Nz = ds.attrib["Nx"], ds.attrib["Ny"], ds.attrib["Nz"]
    Δy = reshape(ds["Δy_aca"][:], 1, Ny, 1)
    Nt = size(u, 4)
    b_min, b_max = b_range
    b_bins = range(b_min, b_max, length=n_b_bins)

    # integrate u over y to get array w size[Nx, Nz, Nt]

    u_full = Array(u[1:Nx, 1:Ny, 1:Nz, :]) #[Nx, Ny, Nz, Nt]
    b_full = Array(b[:, :, :, :])           #[Nx, Ny, Nz, Nt]
    
    #apply wet masks
    wet4d = repeat(wet, 1, 1, 1, Nt) #[Nx, Ny, Nz, Nt]
    u_full[.!wet4d] .= NaN
    b_full[.!wet4d] .= NaN

    #take mean of b overy y dim
    b_xzt = nanmean(b_full, dims=2)[:, 1, :, :] # [Nx, Nz, Nt]

    # integrate u over y and weight by dz in one calculation
    ∫udy_dz = nansum(u_full .* Δy, dims=2)[:, 1, :, :]  .*
            reshape(ds["Δz_aac"][:], 1, Nz, 1) #[Nx, Nz, Nt]

    # ψ(x, b, t) streamfunction in buoyancy space
    ψ_b = zeros(Float32, Nx, n_b_bins, Nt) # [Nx, Nb, Nt]

    for (i, b_0) in enumerate(b_bins)
        M = b_xzt .< b_0
        ψ_b[:, i, :] = nansum(∫udy_dz .* M, dims=2)[:, 1, :] #[Nx, Nt]
    end

    return ψ_b, collect(b_bins)
end

ψ_b, b_out = get_ψb(u, b, wet, ds; b_range)


function plot_ψ_snapshots(ψ_b, t, x, b)

    # find time indices closest to your target times
    target_times = [12, 25, 40, 55]
    t_indices    = [argmin(abs.(t .- τ)) for τ in target_times]

    clim = maximum(abs.(filter(!isnan, vec(ψ_b))))

    fig = Figure(size=(1200, 800))

    for (n, tidx) in enumerate(t_indices)
        row = (n - 1) ÷ 2 + 1
        col = (n - 1) % 2 + 1

        ax = Axis(fig[row, col],
            title  = "ψ_b at t = $(round(t[tidx], digits=1))s",
            xlabel = "x",
            ylabel = "b"
        )

        ψ_snap = ψ_b[:, :, tidx]                        # [Nx, Nz]

        hm = heatmap!(ax, x, b, ψ_snap;
            colormap = :RdBu,
            colorrange = (-clim, clim)
        )
    end
    Colorbar(fig[1:2, 3], fig.content[1], label="ψ_b")

    return fig
end

@info("creating plot 1: surface buoyancy versus x at diff times")

fig3 = plot_ψ_snapshots(ψ_b, time, x, b_out)

save(joinpath(plot_dir, "psi_snapshots.png"), fig3)
@info(" saved psi_snapshots.png")

output_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/"

out_path = joinpath(output_dir, "psi_b_t194.nc")

NCDataset(out_path, "c") do ds_out
    # define dimensions
    defDim(ds_out, "x", Nx)
    defDim(ds_out, "b",    length(b_out))
    defDim(ds_out, "time", length(time))

    v_x = defVar(ds_out, "x", Float64, ("x",))
    v_x[:] = x
    v_x.attrib["long_name"] = "x position"

    # b_out
    v_b = defVar(ds_out, "b_out", Float64, ("b",))
    v_b[:] = b_out
    v_b.attrib["long_name"] = "buoyancy bin centers"

    # time
    v_t = defVar(ds_out, "time", Float64, ("time",))
    v_t[:] = time
    v_t.attrib["long_name"] = "time"

    # G_mix_all  [n_b_bins, Nt]
    v_g = defVar(ds_out, "ψ_b", Float32, ("x", "b", "time"))
    v_g[:, :, :] = ψ_b
    v_g.attrib["long_name"] = "streamfunction in buoyancy space"

    # optional: copy a few global attributes for provenance
    ds_out.attrib["source_file"] = "combined_t194.nc"
    ds_out.attrib["RA"]          = "1e8"
end

println("saved → $out_path")