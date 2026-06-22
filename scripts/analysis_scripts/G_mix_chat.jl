using NCDatasets
using CairoMakie
using Statistics
using NaNStatistics
using Base.Threads

ds_b       = NCDataset("/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/combined_t385.nc")
ds_psi = NCDataset("/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/psi_b_t385.nc")

#ds_psi_control = NCDataset("/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/Control/RA1e8/4x_stretch/512_128/psi_b_Control_RA1e8_seg1to10.nc")
#ds_gmix = NCDataset("/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/Gmix_regions_t194.nc")

# ds_gmix = NCDataset("/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/G_mix_t194.nc")

# gmix_end = ds_gmix["G_mix"][:, end-10:end]
# gmix_mean = mean(gmix, dims=2)[:]

# lines(gmix_mean)

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

fig = Figure()
axis = Axis(fig[1,1])


for n in 3638:3648
b = ds_b["b"][:, :, :, n]
time = ds_b["time"][n]
chi = ds_b["chi"][:, :, :, n]

b_min, b_max = -1, 1
n_b_bins = 501
b_bins = range(b_min, b_max, length=n_b_bins)

Δx_vec = ds_b["Δx_caa"][:]
Δy_vec = ds_b["Δy_aca"][:]
Δz_vec = ds_b["Δz_aac"][:]

Nx, Ny, Nz = ds_b.attrib["Nx"], ds_b.attrib["Ny"], ds_b.attrib["Nz"]

Δx = reshape(Δx_vec, Nx, 1, 1)
Δy = reshape(Δy_vec, 1, Ny, 1)
Δz = reshape(Δz_vec, 1, 1, Nz)
ΔV = Δx .* Δy .* Δz

b_region = vec(b[:,:,:])
χdV_region = vec(chi[:, :, :] .* ΔV)

perm     = sortperm(b_region)
b_sorted = b_region[perm]
cum_χdV  = cumsum(χdV_region[perm])

integral_vals = zeros(n_b_bins)
for (i, b_0) in enumerate(b_bins)
    idx = searchsortedlast(b_sorted, b_0)
    integral_vals[i] = idx > 0 ? cum_χdV[idx] : 0.0
end

#integral_vals = gaussian_smooth(integral_vals, 2)

db    = step(b_bins)
G_mix = 0.5*diff(diff(integral_vals)) ./ db^2
b_out = collect(b_bins)[2:end-1]

G_mix_smooth = gaussian_smooth(G_mix, 10)

#lines!(axis, b_bins, integral_vals)
lines!(axis, b_out, G_mix_smooth)

db    = step(b_bins)
G_mix = 0.5*diff(diff(integral_vals)) ./ db^2
b_out = collect(b_bins)[2:end-1]

G_mix_smooth = gaussian_smooth(G_mix, 20)

#lines!(axis, b_bins, integral_vals)
lines!(axis, b_out, G_mix_smooth)

end
fig

#integral_vals_copy = copy(integral_vals)

#ds_gmix_control = NCDataset("/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/Control/RA1e8/4x_stretch/512_128/Gmix_regions_Control_RA1e8_seg1to10.nc")
#psiControl = ds_gmix_control["psi"]
#heatmap(psiControl[:, :, end])

fig = Figure()
axis = Axis(fig[1,1])

n = 3600
b = ds_b["b"][:, :, :, n]
time = ds_b["time"][n]
chi = ds_b["chi"][:, :, :, n]

b_min, b_max = -1, 1
n_b_bins = 501
b_bins = range(b_min, b_max, length=n_b_bins)

Δx_vec = ds_b["Δx_caa"][:]
Δy_vec = ds_b["Δy_aca"][:]
Δz_vec = ds_b["Δz_aac"][:]

Nx, Ny, Nz = ds_b.attrib["Nx"], ds_b.attrib["Ny"], ds_b.attrib["Nz"]

Δx = reshape(Δx_vec, Nx, 1, 1)
Δy = reshape(Δy_vec, 1, Ny, 1)
Δz = reshape(Δz_vec, 1, 1, Nz)
ΔV = Δx .* Δy .* Δz

b_region = vec(b[:,:,:])
χdV_region = vec(chi[:, :, :] .* ΔV)
dV_region = vec(ΔV)

perm     = sortperm(b_region)
b_sorted = b_region[perm]
cum_χdV  = cumsum(χdV_region[perm])
cum_dV  = cumsum(dV_region[perm])

χ̄ = cum_χdV./cum_dV

χ̄_binned = zeros(n_b_bins)
V_binned = zeros(n_b_bins)
for (i, b_0) in enumerate(b_bins)
    idx = searchsortedlast(b_sorted, b_0)
    χ̄_binned[i] = idx > 0 ? χ̄[idx] : 0.0
    V_binned[i] = idx > 0 ? cum_dV[idx] : 0.0
end

db    = step(b_bins)

term1 = 0.5*V_binned[2:end-1].*diff(diff(χ̄_binned)) ./ db^2
term2 = 0.5*χ̄_binned[2:end-1].*diff(diff(V_binned)) ./ db^2
term3 = 0.5*2*diff(V_binned).*diff(χ̄_binned) ./ db^2

G_mix = term1 .+ term2 .+ term3[1:end-1]

gmix_smooth = gaussian_smooth(G_mix, 10)

lines(gmix_smooth)

# G_mix = 0.5*diff(diff(integral_vals)) ./ db^2
# b_out = collect(b_bins)[2:end-1]

# G_mix_smooth = gaussian_smooth(G_mix, 10)

# #lines!(axis, b_bins, integral_vals)
# lines!(axis, b_out, G_mix_smooth)

# db    = step(b_bins)
# G_mix = 0.5*diff(diff(integral_vals)) ./ db^2
# b_out = collect(b_bins)[2:end-1]

# G_mix_smooth = gaussian_smooth(G_mix, 20)

# #lines!(axis, b_bins, integral_vals)
# lines!(axis, b_out, G_mix_smooth)

# fig



#b = ds["b"]
# x = ds["x_caa"][:]
# y = ds["y_aca"][:]
# z = ds["z_aac"][:]
b_out = ds_psi["b_out"][:]
x = ds_psi["x"][:]

psi_b = ds_psi["ψ_b"]
time = ds_psi["time"][end]

b_out[200:250]
psi = ds_gmix["psi"]

#b_300ish = Array(b[:, 12, :, 100])
psi_b_t = Array(psi_b[:, :, end-10:end])
psi_b_mean = mean(psi_b_t, dims=3)[:, :, 1]

#psi_end = Array(psi[:, :, end-10:end])
#psi_mean_end = mean(psi_end, dims=3)

fig = Figure()
ax = Axis(fig[1,1])
hm = heatmap!(x, b_out, psi_b_mean, colormap=:balance, colorrange=(-0.005, 0.005))
Colorbar(fig[1,2], hm)

fig














# #my script g_mix_calc.jl was taking too long so i put it into chat 
# #and asked it to make the script run faster , this is it. lets see how it does

# plot_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU/GRC/RA1e8/4x_stretch/figures/"

# ds = NCDataset("/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/combined_t194.nc")

# # =========================
# # lazy dataset handles
# # =========================

# b_ds = ds["b"]
# χ_ds = ds["chi"]
# u_ds = ds["u"]

# x = ds["x_caa"][:]
# y = ds["y_aca"][:]
# z = ds["z_aac"][:]
# time = ds["time"][:]

# Nx, Ny, Nz = ds.attrib["Nx"], ds.attrib["Ny"], ds.attrib["Nz"]
# Nt = length(time)

# Δx = reshape(ds["Δx_caa"][:], Nx, 1, 1)
# Δy = reshape(ds["Δy_aca"][:], 1, Ny, 1)
# Δz = reshape(ds["Δz_aac"][:], 1, 1, Nz)

# ΔV = Δx .* Δy .* Δz

# # ============================================================
# # compute buoyancy range WITHOUT loading full 4D array at once
# # ============================================================

# println("finding buoyancy range...")

# local_bmin = Inf
# local_bmax = -Inf

# for tt in 1:Nt

#     b_t = @views b_ds[:, :, :, tt]

#     wet = b_t .!= 0

#     if any(wet)

#         global local_bmin = min(local_bmin, minimum(b_t[wet]))
#         global local_bmax = max(local_bmax, maximum(b_t[wet]))

#     end
# end

# b_range = (local_bmin, local_bmax)

# println("b_range = ", b_range)

# # ============================================================
# # gaussian smoother
# # ============================================================

# function gaussian_smooth(x::Vector, σ::Real)

#     n = length(x)

#     kernel_half = ceil(Int, 3σ)

#     out = similar(x, Float64)

#     @inbounds for i in 1:n

#         wsum = 0.0
#         vsum = 0.0

#         for j in max(1, i-kernel_half):min(n, i+kernel_half)

#             w = exp(-0.5 * ((i-j)/σ)^2)

#             wsum += w
#             vsum += w * x[j]
#         end

#         out[i] = vsum / wsum
#     end

#     return out
# end

# # ============================================================
# # G_mix calculation
# # ============================================================

# function G_mix_calc(
#     b_ds,
#     χ_ds,
#     ΔV;
#     b_range,
#     n_b_bins = 100
# )

#     b_min, b_max = b_range

#     b_bins = range(b_min, b_max, length=n_b_bins)

#     Nt = size(b_ds, 4)

#     integral_vals = zeros(Float64, n_b_bins, Nt)

#     println("computing G_mix...")

#     # ========================================================
#     # timestep-by-timestep processing
#     # ========================================================

#     for tt in 1:Nt

#         println(" timestep $tt / $Nt")

#         b_t = @views b_ds[:, :, :, tt]
#         χ_t = @views χ_ds[:, :, :, tt]

#         # mask dry cells
#         wet = b_t .!= 0

#         χdV = similar(b_t, Float64)

#         @. χdV = ifelse(wet, χ_t * ΔV, NaN)

#         # threaded buoyancy bin loop
#         Threads.@threads for i in eachindex(b_bins)

#             b_0 = b_bins[i]

#             M = wet .& (b_t .< b_0)

#             integral_vals[i, tt] = nansum(@view χdV[M])
#         end
#     end

#     println("smoothing + differentiating...")

#     G_mix_all = zeros(Float32, n_b_bins - 2, Nt)

#     σ = 8
#     db = step(b_bins)

#     Threads.@threads for tt in 1:Nt

#         s = gaussian_smooth(integral_vals[:, tt], σ)

#         G_mix_all[:, tt] .= 0.5 .* diff(diff(s)) ./ db^2
#     end

#     b_out = collect(b_bins)[2:end-1]

#     return b_out, G_mix_all
# end

# @time b_out, G_mix_all = G_mix_calc(
#     b_ds,
#     χ_ds,
#     ΔV;
#     b_range,
#     n_b_bins = 100
# )

# # ============================================================
# # plot 1
# # ============================================================

# println("creating G_mix heatmap")

# gmixlim = maximum(abs.(filter(!isnan, G_mix_all)))

# fig = Figure(size=(1200,800))

# ax = Axis(
#     fig[1,1],
#     xlabel = "time",
#     ylabel = "buoyancy",
#     title = L"\mathcal{G}_{mix}"
# )

# hm = heatmap!(
#     ax,
#     time,
#     b_out,
#     G_mix_all',
#     colormap = :balance,
#     colorrange = (-gmixlim, gmixlim)
# )

# Colorbar(fig[1,2], hm)

# save(joinpath(plot_dir, "G_heatmap_fast.png"), fig)

# println("saved G_heatmap_fast.png")

# # ============================================================
# # plot 2
# # ============================================================

# println("creating time averaged G_mix plot")

# G_b = mean(G_mix_all, dims=2)

# fig2 = Figure(size=(1200,800))

# ax2 = Axis(
#     fig2[1,1],
#     xlabel = "buoyancy",
#     ylabel = "time averaged G_mix"
# )

# lines!(ax2, b_out, G_b[:])

# save(joinpath(plot_dir, "G_avg_b_fast.png"), fig2)

# println("saved G_avg_b_fast.png")

# # ============================================================
# # ψ_b calculation
# # ============================================================

# function get_ψb(
#     u_ds,
#     b_ds,
#     ds;
#     b_range,
#     n_b_bins = 100
# )

#     Nx, Ny, Nz = ds.attrib["Nx"], ds.attrib["Ny"], ds.attrib["Nz"]

#     Δy = reshape(ds["Δy_aca"][:], 1, Ny, 1)
#     Δz = reshape(ds["Δz_aac"][:], 1, Nz, 1)

#     Nt = size(u_ds, 4)

#     b_min, b_max = b_range

#     b_bins = range(b_min, b_max, length=n_b_bins)

#     ψ_b = zeros(Float32, Nx, n_b_bins, Nt)

#     println("computing ψ_b...")

#     for tt in 1:Nt

#         println(" timestep $tt / $Nt")

#         u_t = @views u_ds[:, :, :, tt]
#         b_t = @views b_ds[:, :, :, tt]

#         wet = b_t .!= 0

#         # mean buoyancy over y
#         b_masked = ifelse.(wet, b_t, NaN)

#         b_xz = nanmean(b_masked, dims=2)[:,1,:]

#         # integrate u over y
#         u_masked = ifelse.(wet, u_t, NaN)

#         ∫udy_dz = (
#             nansum(u_masked .* Δy, dims=2)[:,1,:]
#             .* Δz
#         )

#         Threads.@threads for i in eachindex(b_bins)

#             b_0 = b_bins[i]

#             M = b_xz .< b_0

#             ψ_b[:, i, tt] .= nansum(∫udy_dz .* M, dims=2)[:]
#         end
#     end

#     return ψ_b, collect(b_bins)
# end

# @time ψ_b, b_out2 = get_ψb(
#     u_ds,
#     b_ds,
#     ds;
#     b_range,
#     n_b_bins = 100
# )

# # ============================================================
# # snapshot plotting
# # ============================================================

# function plot_ψ_snapshots(ψ_b, t, x, b)

#     target_times = [12, 25, 40, 55]

#     t_indices = [argmin(abs.(t .- τ)) for τ in target_times]

#     fig = Figure(size=(1200, 800))

#     for (n, tidx) in enumerate(t_indices)

#         row = (n - 1) ÷ 2 + 1
#         col = (n - 1) % 2 + 1

#         ax = Axis(
#             fig[row, col],
#             title  = "ψ at t = $(round(t[tidx], digits=1))",
#             xlabel = "x",
#             ylabel = "b"
#         )

#         ψ_snap = ψ_b[:, :, tidx]

#         clim = maximum(abs.(filter(!isnan, ψ_snap)))

#         hm = heatmap!(
#             ax,
#             x,
#             b,
#             ψ_snap',
#             colormap = :RdBu,
#             colorrange = (-clim, clim)
#         )

#         Colorbar(fig[row, col+2], hm)
#     end

#     return fig
# end

# println("creating ψ snapshots")

# fig3 = plot_ψ_snapshots(ψ_b, time, x, b_out2)

# save(joinpath(plot_dir, "psi_snapshots_fast.png"), fig3)

# println("saved psi_snapshots_fast.png")

# close(ds)