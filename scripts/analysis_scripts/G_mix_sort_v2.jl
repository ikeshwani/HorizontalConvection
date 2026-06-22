using NCDatasets
using CairoMakie
using Printf
using Statistics
using NaNStatistics

# ---- paths ----
data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/Control/RA1e8/4x_stretch/512_128/"
plot_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU/GRC/Control/RA1e8/4x_stretch/figures/"
outfile  = joinpath(data_dir, "Gmix_regions_Control_RA1e8_seg1to12.nc")
mkpath(plot_dir)

segments = 1:12

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

# flat-bottom control: all cells are wet
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

ΔA    = reshape(Δx_vec, Nx, 1) .* reshape(Δy_vec, 1, Ny)
region_precomp = map(region_masks) do (name, mask)
    idxs     = findall(vec(mask .& wet))
    hmask    = dropdims(any(mask, dims=3), dims=3)
    A_region = sum(ΔA .* hmask)
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

# G_mix via product rule: d²/db²[V(b)·χ̄(b)] = V·χ̄'' + 2·V'·χ̄' + V''·χ̄
function G_mix_calc_v2(b_region::Vector, χdV_region::Vector, dV_region::Vector, b_range;
                       n_b_bins=500, σ=10)
    b_min, b_max = b_range
    b_bins = range(b_min, b_max, length=n_b_bins)
    db = step(b_bins)

    perm     = sortperm(b_region)
    b_sorted = b_region[perm]
    cum_χdV  = cumsum(χdV_region[perm])
    cum_dV   = cumsum(dV_region[perm])

    χ̄ = cum_χdV ./ cum_dV

    χ̄_binned = zeros(n_b_bins)
    V_binned  = zeros(n_b_bins)
    for (i, b_0) in enumerate(b_bins)
        idx = searchsortedlast(b_sorted, b_0)
        χ̄_binned[i] = idx > 0 ? χ̄[idx] : 0.0
        V_binned[i]  = idx > 0 ? cum_dV[idx] : 0.0
    end

    term1 = 0.5 .* V_binned[2:end-1] .* diff(diff(χ̄_binned)) ./ db^2
    term2 = 0.5 .* χ̄_binned[2:end-1] .* diff(diff(V_binned)) ./ db^2
    term3 = (diff(V_binned) .* diff(χ̄_binned)) ./ db^2

    G_mix = term1 .+ term2 .+ term3[1:end-1]
    return gaussian_smooth(G_mix, σ)
end

# ---- precompute time-invariant dV flat vector ----
dV_flat = vec(ΔV)

# ---- b_range and output b axis ----
b_range  = (-1, 1)
n_b_bins = 501
b_out    = collect(range(b_range[1], b_range[2], length=n_b_bins))[2:end-1]
n_b      = length(b_out)

# ---- Pass 1: collect time vector (cheap — no field data loaded) ----
println("Pass 1: scanning time vectors from segments $(segments[1])–$(segments[end])...")
time_all = Float64[]
let t_last = -Inf
    for s in segments
        bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
        t_seg = Float64.(bfile["time"][:])
        close(bfile)
        valid = findall(t_seg .> t_last)
        isempty(valid) && continue
        append!(time_all, t_seg[valid])
        t_last = t_seg[valid[end]]
    end
end
Nt   = length(time_all)
time = time_all
println("total time steps: $Nt  (t = $(time[1]) → $(time[end]))")

# ---- pre-allocate output arrays ----
Gmix_regions = Dict(r.name => zeros(Float32, n_b, Nt) for r in region_precomp)
ψ_all        = zeros(Float32, Nx, Nz, Nt)

# ---- Pass 2: process one segment at a time ----
println("Pass 2: computing G_mix + ψ segment by segment...")

let t_last = -Inf, t_offset = 0
for s in segments
    bfile = NCDataset(joinpath(data_dir, "buoyancy_seg$(s).nc"))
    vfile = NCDataset(joinpath(data_dir, "velocities_seg$(s).nc"))

    t_seg = Float64.(bfile["time"][:])
    valid = findall(t_seg .> t_last)

    if isempty(valid)
        @printf("  seg %d: all steps are duplicates — skipping\n", s)
        close(bfile); close(vfile)
        continue
    end

    n_skip = valid[1] - 1
    n_skip > 0 && @printf("  seg %d: skipping first %d overlapping step(s)\n", s, n_skip)

    n_v     = size(vfile["u"], 4)
    t_range = valid[1]:min(valid[end], n_v)
    nt      = length(t_range)
    gi      = t_offset+1 : t_offset+nt   # global index range

    @printf("  seg %d: loading %d steps (t = %.2f → %.2f)...\n",
            s, nt, t_seg[t_range[1]], t_seg[t_range[end]])

    b_seg = Array(bfile["b"][:, :, :, t_range])
    χ_seg = Array(bfile["chi"][:, :, :, t_range])
    u_seg = Array(vfile["u"][1:Nx, :, :, t_range])
    close(bfile); close(vfile)

    # ---- G_mix for each time step in this segment ----
    for (ti, g) in enumerate(gi)
        b_flat   = vec(b_seg[:, :, :, ti])
        χdV_flat = vec(χ_seg[:, :, :, ti] .* ΔV)
        for r in region_precomp
            G = G_mix_calc_v2(b_flat[r.idxs], χdV_flat[r.idxs], dV_flat[r.idxs], b_range; n_b_bins)
            Gmix_regions[r.name][:, g] = G
        end
        g % 100 == 0 && @printf("    G_mix: step %d / %d\n", g, Nt)
    end

    # ---- ψ for this segment: ψ[x,z,t] = -cumsum_z(∫u dy · Δz) ----
    u_seg[u_seg .== 0] .= NaN
    ∫udy   = dropdims(nansum(u_seg .* reshape(Δy_vec, 1, :, 1, 1), dims=2), dims=2)  # [Nx, Nz, nt]
    ∫udy_w = ∫udy .* reshape(Δz_vec, 1, Nz, 1)
    ∫udy_w[isnan.(∫udy_w)] .= 0.0f0
    ψ_all[:, :, gi] = Float32.(-cumsum(∫udy_w, dims=2))

    t_last   = t_seg[t_range[end]]
    t_offset += nt

    # free segment arrays before loading the next segment
    b_seg = nothing; χ_seg = nothing; u_seg = nothing
    GC.gc()

    @printf("  seg %d: done\n", s)
end  # for s
end  # let t_last

# ---- save ----
println("saving → $outfile")
NCDataset(outfile, "c") do ds_out
    defDim(ds_out, "b",    n_b)
    defDim(ds_out, "time", Nt)
    defDim(ds_out, "x",    Nx)
    defDim(ds_out, "z",    Nz)

    defVar(ds_out, "b",    b_out, ("b",))
    defVar(ds_out, "time", time,  ("time",))
    defVar(ds_out, "x",    x,     ("x",))
    defVar(ds_out, "z",    z,     ("z",))

    for r in region_precomp
        defVar(ds_out, "Gmix_$(r.name)", Gmix_regions[r.name], ("b", "time"))
    end

    defVar(ds_out, "psi", ψ_all, ("x", "z", "time"))
end
println("saved → $outfile")

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
    figname = "Gmix_density_regions_v2_Control_RA1e8_seg1to12.png"
    save(joinpath(plot_dir, figname), fig)
    println("saved figure → $(joinpath(plot_dir, figname))")
    return fig
end

plot_gmix_regions(b_out, time, Gmix_regions, region_precomp, plot_dir)
