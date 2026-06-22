using NCDatasets
using CairoMakie
using Printf
using Statistics
using NaNStatistics

# ---- paths ----
data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/Control/RA1e8/4x_stretch/512_128/"
plot_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU/GRC/Control/RA1e8/4x_stretch/figures/"
outfile  = joinpath(data_dir, "Gmix_regions_Control_RA1e8_seglast.nc")
mkpath(plot_dir)

segments = 9:10

# ---- load grid info from seg1 ----
ds1    = NCDataset(joinpath(data_dir, "buoyancy_seg1.nc"))
x      = ds1["x_caa"][:]
y      = ds1["y_aca"][:]
z      = ds1["z_aac"][:]
Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
Lx     = ds1.attrib["Lx"]
Δx_vec = ds1["Δx_caa"][:]
Δy_vec = ds1["Δy_aca"][:]
Δz_vec = ds1["Δz_aac"][:]
close(ds1)

Δx = reshape(Δx_vec, Nx, 1, 1)
Δy = reshape(Δy_vec, 1, Ny, 1)
Δz = reshape(Δz_vec, 1, 1, Nz)
ΔV = Δx .* Δy .* Δz
ΔA = Δx .* Δy

# ---- load segments 1–9, skipping overlapping time steps ----
println("loading b, χ, u from segments 1–10...")
b_segs    = Vector{Array{Float32,4}}()
χ_segs    = Vector{Array{Float32,4}}()
u_segs    = Vector{Array{Float32,4}}()
time_segs = Vector{Vector{Float64}}()

let t_last = -Inf
    for s in segments
        bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
        vfile = NCDataset(joinpath(data_dir, "velocities_seg$(s).nc"))

        t_seg = bfile["time"][:]
        valid = findall(t_seg .> t_last)

        if isempty(valid)
            @printf("  seg %d: all %d steps are duplicates — skipping\n", s, length(t_seg))
            close(bfile); close(vfile)
            continue
        end

        n_skip = valid[1] - 1
        n_skip > 0 && @printf("  seg %d: skipping first %d overlapping step(s)\n", s, n_skip)

        # valid is always contiguous (monotone time); use UnitRange to avoid
        # NCDatasets failing on mixed UnitRange+Vector indexing for u.
        # Clip to the velocity file's time dimension — it may be shorter than b.
        n_v     = size(vfile["u"], 4)
        t_range = valid[1]:min(valid[end], n_v)
        push!(b_segs,    Array(bfile["b"][:, :, :, t_range]))
        push!(χ_segs,    Array(bfile["chi"][:, :, :, t_range]))
        push!(u_segs,    Array(vfile["u"][1:Nx, :, :, t_range]))
        push!(time_segs, t_seg[t_range])

        t_last = t_seg[valid[end]]
        close(bfile); close(vfile)
        @printf("  seg %d: loaded %d steps (t = %.2f → %.2f)\n", s, length(valid), t_seg[valid[1]], t_last)
    end
end

b_all = cat(b_segs...; dims=4)
χ_all = cat(χ_segs...; dims=4)
u_all = cat(u_segs...; dims=4)
time  = vcat(time_segs...)
Nt    = length(time)
println("total time steps: $Nt  (t = $(time[1]) → $(time[end]))")

# wet mask: flat-bottom (Control) has no immersed boundary so all cells are wet
wet = trues(Nx, Ny, Nz)

# ---- region masks ----
zBL = -(round(Lx * (1e8)^(-1/5); digits=2) + 0.02)
X = reshape(x, :, 1, 1)
Z = reshape(z, 1, 1, :)

region_masks = [
    ("plume",          X .< -1.8),
    ("boundary_layer", (Z .> zBL) .& (X .>= -1.8)),
    ("basin0",         (X .>= -1.8) .& (X .< -1.35) .& (Z .< zBL)),
    ("hill1",          (X .>= -1.35) .& (X .< -0.65) .& (Z .< zBL)),
    ("basin1",         (X .>= -0.65) .& (X .< -0.35) .& (Z .< zBL)),
    ("hill2",          (X .>= -0.35) .& (X .< 0.35)  .& (Z .< zBL)),
    ("basin2",         (X .>= 0.35)  .& (X .< 0.65)  .& (Z .< zBL)),
    ("hill3",          (X .>= 0.65)  .& (X .< 1.35)  .& (Z .< zBL)),
    ("basin3",         (X .>= 1.35)  .& (Z .< zBL)),
]

ΔA_2d = dropdims(ΔA, dims=3)
region_precomp = map(region_masks) do (name, mask)
    idxs     = findall(vec(mask .& wet))
    hmask    = dropdims(any(mask, dims=3), dims=3)
    A_region = sum(ΔA_2d .* hmask)
    (; name, idxs, A_region)
end

# ---- functions ----

function gaussian_smooth(x::Vector, σ::Real)
    n           = length(x)
    kernel_half = ceil(Int, 3σ)
    out         = similar(x, Float64)
    for i in 1:n
        wsum = 0.0; vsum = 0.0
        for j in max(1, i - kernel_half):min(n, i + kernel_half)
            w     = exp(-0.5 * ((i - j) / σ)^2)
            wsum += w
            vsum += w * x[j]
        end
        out[i] = vsum / wsum
    end
    return out
end

function G_mix_calc(b_region::Vector, χdV_region::Vector, b_range; n_b_bins=500)
    b_min, b_max = b_range
    b_bins = range(b_min, b_max, length=n_b_bins)

    perm     = sortperm(b_region)
    b_sorted = b_region[perm]
    cum_χdV  = cumsum(χdV_region[perm])

    χdV_smooth = gaussian_smooth(cum_χdV, 20)


    # integral_smooth = gaussian_smooth(integral_vals, 15)
    db    = step(b_bins)
    G_mix = 0.5*diff(diff(integral_vals)) ./ db^2
    b_out = collect(b_bins)[2:end-1]
    return b_out, G_mix
end

function get_ψ(u_all, Δy_vec, Δz_vec, Nx, Nz, Nt)
    u_all[u_all .== 0] .= NaN

    ∫udy = dropdims(nansum(u_all .* reshape(Δy_vec, 1, :, 1, 1), dims=2), dims=2)

    ∫udy_w = ∫udy .* reshape(Δz_vec, 1, Nz, 1)
    ∫udy_w[isnan.(∫udy_w)] .= 0
    return Float32.(-cumsum(∫udy_w, dims=2))  # [Nx, Nz, Nt]
end

# ---- b_range and output b axis ----
b_range  = (-1, 1)
n_b_bins = 501
b_out    = collect(range(b_range[1], b_range[2], length=n_b_bins))[2:end-1]
n_b      = length(b_out)

# ---- main time loop ----
Gmix_regions = Dict(r.name => zeros(Float32, n_b, Nt) for r in region_precomp)

println("computing G_mix: $Nt time steps × $(length(region_precomp)) regions...")
for t in 1:Nt
    b_flat   = vec(b_all[:, :, :, t])
    χdV_flat = vec(χ_all[:, :, :, t] .* ΔV)

    for r in region_precomp
        _, G = G_mix_calc(b_flat[r.idxs], χdV_flat[r.idxs], b_range; n_b_bins)
        Gmix_regions[r.name][:, t] = G
    end

    t % 50 == 0 && @printf("  t = %d / %d\n", t, Nt)
end

# ---- streamfunction ----
println("computing streamfunction ψ...")
ψ = get_ψ(u_all, Δy_vec, Δz_vec, Nx, Nz, Nt)

# ---- save ----
function save_gmix_regions(outfile, b_out, time, Gmix_regions, region_precomp, ψ, x, z)
    NCDataset(outfile, "c") do ds_out
        defDim(ds_out, "b",    length(b_out))
        defDim(ds_out, "time", length(time))
        defDim(ds_out, "x",    size(ψ, 1))
        defDim(ds_out, "z",    size(ψ, 2))

        defVar(ds_out, "b",    b_out, ("b",))
        defVar(ds_out, "time", time,  ("time",))
        defVar(ds_out, "x",    x,     ("x",))
        defVar(ds_out, "z",    z,     ("z",))

        for r in region_precomp
            defVar(ds_out, "Gmix_$(r.name)", Gmix_regions[r.name], ("b", "time"))
        end

        defVar(ds_out, "psi", ψ, ("x", "z", "time"))
    end
    println("saved → $outfile")
end

save_gmix_regions(outfile, b_out, time, Gmix_regions, region_precomp, ψ, x, z)

# ---- plot ----
function plot_gmix_regions(b_out, time, Gmix_regions, region_precomp, plot_dir)
    names     = [r.name for r in region_precomp]
    n_regions = length(names)
    fig  = Figure(size=(300 * n_regions, 400))
    clim = 0.008
    for (i, name) in enumerate(names)
        ax = Axis(fig[1, 2i-1], xlabel="time", ylabel="buoyancy", title="Gmix density: $name")
        hm = heatmap!(ax, time, b_out, Gmix_regions[name]', colormap=:balance, colorrange=(-clim, clim))
        Colorbar(fig[1, 2i], hm)
    end
    figname = "Gmix_density_regions_Control_RA1e8_seg1to10.png"
    save(joinpath(plot_dir, figname), fig)
    println("saved figure → $(joinpath(plot_dir, figname))")
    return fig
end

fig = plot_gmix_regions(b_out, time, Gmix_regions, region_precomp, plot_dir)
