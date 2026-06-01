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

#define the wet mask outside

wet = Array(b[:, :, :, 2] .!= 0)

function gaussian_smooth(x::Vector, σ::Real)
    n      = length(x)
    kernel_half = ceil(Int, 3σ)
    out    = similar(x, Float64)
    for i in 1:n
        wsum = 0.0; vsum = 0.0
        for j in max(1,i-kernel_half):min(n,i+kernel_half)
            w     = exp(-0.5*((i-j)/σ)^2)
            wsum += w
            vsum += w * x[j]
        end
        out[i] = vsum / wsum
    end
    return out
end

function G_mix_calc(b, χ, ΔV, wet; b_range, n_b_bins=100)
    """
    this function takes in arrays of b and χ so size [Nx, Ny, Nz] at a single time!
        and calculates g_mix by making a mask for buoyancies less than b_0 and iterating  
        thru all buoyancy values. then take the integral of χdV, smooth out that array
        using a guassian smoother (shoutout claude), and finally takes the second derivative
        wrt to buoyancy * 1/2 --> g_mix
    """
    b = copy(b)

    # make dry points = nans (hills)
    b[.!wet] .= NaN
    χ[.!wet] .= NaN

    b_min, b_max = b_range
    b_bins = range(b_min, b_max, length=n_b_bins)
    
    χdV = χ .* ΔV # [Nx, Ny, Nz]

    integral_vals = zeros(n_b_bins)

    for (i, b_0) in enumerate(b_bins)
        M = b .< b_0
        integral_vals[i] = nansum(χdV .* M)
    end

    #smooth before taking the second derivative
    σ = 15
    integral_smooth = gaussian_smooth(integral_vals, σ)


    db = step(b_bins)
    G_mix = 0.5 .* (diff(diff(integral_smooth)) ./ db^2) #this is G_mix at one time
    b_out = collect(b_bins)[2:end-1]

    return b_out, G_mix
    
end

Nt = length(time)
b_range = (nanminimum(b), nanmaximum(b))
b_init = Array(b[:, :, :, 1])
χ_init = Array(χ[:, :, :, 1])

b_out = G_mix_calc(b_init, χ_init, ΔV, wet; b_range)[1]

G_mix_all = zeros(Float32, length(b_out), Nt)

for t in 1:Nt
    G_mix_all[:, t] = G_mix_calc(Array(b[:, :, :, t]), Array(χ[:, :, :, t]), ΔV, wet; b_range)[2]
end

@info("creating plot 1: heatmap of G_mix in buoyancy and temporal space")

gmixlim   = maximum(abs.(filter(!isnan, G_mix_all)))

fig = Figure(size=(1200,800))
ax = Axis(fig[1,1], xlabel = "time", ylabel="buoyancy range", title=L"\mathcal{G}_{mix} \mathrm{\, heatmap}")
hm = heatmap!(ax, time, b_out, G_mix_all', colormap=:balance, colorrange=(-gmixlim, gmixlim))

Colorbar(fig[1,2], hm, label = L"\mathcal{G}_{mix}")

save(joinpath(plot_dir, "G_heatmap_Maskmethod2.png"), fig)
@info(" saved G_heatmap.png")

@info("creating plot 2: time averaged G_mix versus buoyancy space")

G_b = mean(G_mix_all, dims=2)

fig2 = Figure(size=(1200,800))
ax2 = Axis(fig2[1, 1], xlabel = "buoyancy values", ylabel="time averaged G_mix", title="time averaged G_mix vs buoyancy")
lines!(ax2, b_out, G_b[:])

save(joinpath(plot_dir, "G_avg_b_Maskmethod2.png"), fig2)
@info("saved G_avg_b plot")

######save the output cuz it takes 5eva to run

@info("saving G_mix_all and b_out to NetCDF...")

output_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/"

out_path = joinpath(output_dir, "G_mix_t194.nc")

NCDataset(out_path, "c") do ds_out
    # define dimensions
    defDim(ds_out, "b",    length(b_out))
    defDim(ds_out, "time", length(time))

    # b_out
    v_b = defVar(ds_out, "b_out", Float64, ("b",))
    v_b[:] = b_out
    v_b.attrib["long_name"] = "buoyancy bin centers"

    # time
    v_t = defVar(ds_out, "time", Float64, ("time",))
    v_t[:] = time
    v_t.attrib["long_name"] = "time"

    # G_mix_all  [n_b_bins, Nt]
    v_g = defVar(ds_out, "G_mix", Float32, ("b", "time"))
    v_g[:, :] = G_mix_all
    v_g.attrib["long_name"] = "mixing distribution G_mix"

    # optional: copy a few global attributes for provenance
    ds_out.attrib["source_file"] = "combined_t194.nc"
    ds_out.attrib["RA"]          = "1e8"
end

println("saved → $out_path")


# function get_ψ(u, x, z, t, ds)

#     Nx, Ny, Nz = ds.attrib["Nx"], ds.attrib["Ny"], ds.attrib["Nz"]
#     Lx, Ly, H  = ds.attrib["Lx"], ds.attrib["Ly"], ds.attrib["H"]
#     Δy = reshape(ds["Δy_aca"][:], 1, Ny, 1)
#     Δz = ds["Δz_aac"][:]
#     # t = ds["time"][:] will be defined post segment
#     Nt = length(t)

#     # integrate u over y to get array w size[Nx, Nz, Nt]
#     # u_full = ds["u"][:, :, :, :]   # size here[Nx, Ny, Nz, Nt]
#     u_full = u
#     u_full[u_full .== 0] .= NaN    # mask dry cells

#     ∫udy = nansum(u_full .* Δy, dims=2)[:, 1, :, :]  # [Nx, Nz, Nt]

#     # cumulative integral over z to get ψ of size [Nx, Nz, Nt]
#     # ψ(x, z, t) = ∫(∫u dy) dz'  cumsum in z weighted by Δz
#     ψ = zeros(Float32, Nx, Nz, Nt)

#     for k in 1:Nz
#         ψ[:, k, :] = nansum(∫udy[:, 1:k, :] .* reshape(Δz[1:k], 1, k, 1), dims=2)[:, 1, :]
#     end

#     return ψ
# end
