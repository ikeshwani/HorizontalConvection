using Printf
using NCDatasets
using CairoMakie
using Observables

# we have our data in several segments 

# function main()

    # basepath = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/"
    # n_segments = 7
    
    # datasets = [NCDataset(joinpath(basepath, "buoyancy_seg$(i).nc")) for i in 1:n_segments]
    # u_data = [NCDataset(joinpath(basepath, "velocities_seg$(i).nc")) for i in 1:n_segments]

    # # println("\n - segment time info")
    # # for (i, ds) in enumerate(datasets)
    # #     t = ds["time"][:]
    # #     @printf(" segment %d | start: %.4f | end: %.4f | nsteps: %d\n", i, t[1], t[end], length(t))
    # # end

#     #read in spatial coords and global attributes from segment 1 cuz its all the same

#     ds1 = datasets[1]

#     x = ds1["x_caa"][:]
#     y = ds1["y_aca"][:]
#     z = ds1["z_aac"][:]

#     Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
#     Lx, Ly, H = ds1.attrib["Lx"], ds1.attrib["Ly"], ds1.attrib["H"]

#     t_combined = Float64[]
#     b_combined = nothing
#     χ_combined = nothing
#     u_combined = nothing 

#     t_end = -Inf

#     for i in 1:n_segments
#         t_i = datasets[i]["time"][:]

#         # find the index in the next file where time exceeds the current end
#         new_idx = findfirst(t -> t > t_end, t_i)

#         if isnothing(new_idx)
#             @printf(" segment %d is fully overlapping - skipping entirely \n", i)
#             continue
#         end

#         n_overlap = new_idx - 1
#         @printf("segment %d : trimming %d overlapping timestep(s)\n", i, n_overlap)

#         valid_range = new_idx:length(t_i)

#         b_i = datasets[i]["b"][:, :, :, valid_range]
#         χ_i = datasets[i]["chi"][:, :, :, valid_range]
#         u_i = u_data[i]["u"][:, :, :, valid_range]

#         append!(t_combined, t_i[valid_range])

#         if b_combined === nothing
#             b_combined = b_i
#         else
#             b_combined = cat(b_combined, b_i; dims=4)
#         end

#         if χ_combined === nothing
#             χ_combined = χ_i
#         else
#             χ_combined = cat(χ_combined, χ_i; dims=4)
#         end

#         if u_combined === nothing
#             u_combined = u_i
#         else
#             u_combined = cat(u_combined, u_i; dims=4)
#         end

#         t_end = t_combined[end]

#         @printf(" -> kept %d steps , new t_end is = %.4f\n", length(valid_range), t_end)

#     end

#     println("\nCombined: $(length(t_combined)) timesteps | t_start=$(t_combined[1]) | t_end=$(t_combined[end])\n")

#     return x, y, z, Lx, Ly, H, t_combined, b_combined, χ_combined, u_combined
# end

# x, y, z, Lx, Ly, H, t, b, χ, u = main() #this is full combined data!

#use one dataset to get the grid info!
ds = NCDataset("/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/combined_t194.nc")

x = ds["x_caa"][:]
y = ds["y_aca"][:]
z = ds["z_aac"][:]
u = ds["u"][:, :, :, :]
b = ds["b"][:, :, :, :]
chi = ds["chi"][:, :, :, :]
t = ds["time"][:]

Lx, Ly, H  = ds.attrib["Lx"], ds.attrib["Ly"], ds.attrib["H"]
Nx = ds.attrib["Nx"]
Ny = ds.attrib["Ny"]
Nz = ds.attrib["Nz"]
Nt = length(t)

Δx = reshape(ds["Δx_caa"][:], Nx, 1, 1)
Δy = reshape(ds["Δy_aca"][:], 1, Ny, 1)
Δz = reshape(ds["Δz_aac"][:], 1, 1, Nz)

ΔV = Δx .* Δy .* Δz

############################# G_mix Calculation and \psi  calcu

plot_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/figures/GPU/GRC/RA1e8/4x_stretch/figures/"



function get_ψ(u, x, z, t, ds)
    Nx, Ny, Nz = ds.attrib["Nx"], ds.attrib["Ny"], ds.attrib["Nz"]
    Lx, Ly, H  = ds.attrib["Lx"], ds.attrib["Ly"], ds.attrib["H"]
    Δy = ds["Δy_aca"][:]
    Δz = ds["Δz_aac"][:]
    Nt = length(t)

    u_full = copy(u)                        # don't mutate the original
    u_full[u_full .== 0] .= NaN            # mask dry cells

    # ∫u dy : sum over wet y-cells only, dry cells contribute 0
    u_for_int = copy(u_full)
    u_for_int[isnan.(u_for_int)] .= 0.0   # zero out NaNs so they don't poison the sum
    ∫udy = sum(u_for_int .* reshape(Δy, 1, Ny, 1, 1), dims=2)[:, 1, :, :]  # [Nx, Nz, Nt]

    # cumulative integral over z (bottom to top), accounting for wet cells only
    ψ = zeros(Float32, Nx, Nz, Nt)
    for k in 1:Nz
        ψ[:, k, :] = sum(
            ∫udy[:, 1:k, :] .* reshape(Δz[1:k], 1, k, 1),
            dims=2
        )[:, 1, :]
    end

    # mask ψ at dry cells using the u field at any wet time index
    u_ref = u_full[:, 1, :, 2]             # [Nx, Nz] — one y-slice for dry/wet mask
    dry_xz = u_ref .== 0.0 .|| isnan.(u_ref)  # true = hill/dry
    # broadcast dry mask across time
    for n in 1:Nt
        ψ[:, :, n][dry_xz] .= NaN
    end

    return ψ
end

# ψ = get_ψ(u, x, z, t, ds)

# function plot_ψ_snapshots(ψ, u, t, x, z)
#     target_times = [12, 25, 40, 55]
#     t_indices    = [argmin(abs.(t .- τ)) for τ in target_times]

#     # build wet mask from a mid-simulation snapshot (avoid cold start at t=1)
#     u_ref    = u[:, 1, :, 2]               # [Nx, Nz]
#     wet      = .!(u_ref .== 0.0 .|| isnan.(u_ref))   # true = fluid
#     hill_mask               = Float64.(copy(wet))
#     hill_mask[wet]         .= NaN          # NaN over fluid, value over hills
#     # hill_mask[.!wet]       .= 1.0          # solid value for :turbid to color

#     fig = Figure(size=(1600, 800))

#     for (n, tidx) in enumerate(t_indices)
#         row = (n - 1) ÷ 2 + 1
#         col = (n - 1) % 2 + 1

#         ax = Axis(fig[row, col],
#             title  = "ψ at t = $(round(t[tidx], digits=1)) s",
#             xlabel = L"x / Lx",
#             ylabel = L"z / H"
#         )

#         ψ_snap = ψ[:, :, tidx]                              # [Nx, Nz]
#         wet_vals = filter(!isnan, ψ_snap)
#         clim = isempty(wet_vals) ? 1.0 : maximum(abs.(wet_vals))

#         # buoyancy/streamfunction field
#         hm = heatmap!(ax, x, z, ψ_snap;
#             colormap   = :RdBu,
#             colorrange = (-clim, clim)
#         )

#         # hill overlay
#         heatmap!(ax, x, z, hill_mask;
#             colormap   = :turbid
#             # colorrange = (0.0, 1.0)
#         )

#         Colorbar(fig[row, col + 2], hm, label = "ψ [m²/s]")
#     end

#     return fig
# end


println("creating plot 1: surface buoyancy versus x at diff times")

# call it — pass u so the mask can be built inside
fig_ψ = plot_ψ_snapshots(ψ, u, t, x, z)


save(joinpath(plot_dir, "psi_snapshots.png"), fig_ψ)
println(" saved psi_snapshots.png")

# Simple Gaussian smoother (no extra packages needed)
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


function compute_Gmix_sort(b, χ, ΔV; n_b_bins=500, b_range=nothing)

    # b_init = b[:, :, :, 1] #b at the first timestep
    # χ_init = b[:, :, :, 1] #chi at first timestep

    #reshape arrays
    N_cells = length(b)                      # works on a single time snapshot
    ΔV_flat = reshape(ΔV, N_cells)
    b_flat = reshape(b, N_cells)
    χ_flat = reshape(χ, N_cells)

    #sort everything by buoyancy value
    sort_idx = sortperm(b_flat)
    b_sort = b_flat[sort_idx]
    χdV_sort = χ_flat[sort_idx] .* ΔV_flat[sort_idx]

    #cumsum function of buoyancy, sampled at every cell's buoyancy
    F_sort  = cumsum(χdV_sort)

    #bin χdV onto uniform grid before cumsum
    b_min, b_max = b_range
    b_uniform = range(b_min, b_max; length=n_b_bins)
    Δb = step(b_uniform)

    #sum χ*dV in each buoyancy bin
    χdV_binned = zeros(Float64, n_b_bins)
    for i in 1:N_cells
        idx = searchsortedlast(b_uniform, b_flat[i])
        idx = clamp(idx, 1, n_b_bins)
        χdV_binned[idx] += χ_flat[i] * ΔV_flat[i]
    end

    F_binned = cumsum(χdV_binned)

    #smooth F before differentiating
    #using gaussian smoother on F
    σ_smooth = max(3, n_b_bins ÷ 40)
    F_smooth = gaussian_smooth(F_binned, σ_smooth)

    #derivative
    #g_mix = 0.5 * d^2F/db^2 = 0.5 * d(χdV_binned)/db
    χdV_smooth = gaussian_smooth(χdV_binned, σ_smooth)
    dχdV_db = diff(χdV_smooth) ./ Δb #first derivative of binned χdV
    b_out = collect(b_uniform[1:end-1]) .+ Δb/2

    return b_out, 0.5 .* dχdV_db ./ Δb
end

Nt = size(b,4)
b_range = (minimum(b), maximum(b))

b_out = compute_Gmix_sort(b[:, :, :, 1], chi[:, :, :, 1], ΔV; b_range)[1] #b_out array at first time
G_mix_all = zeros(Float32, length(b_out), Nt)

for t in 1:Nt
    G_mix_all[:,t] = compute_Gmix_sort(b[:, :, :, t], χ[:, :, :, t], ΔV; b_range)[2]
end

fig1 = Figure(size=(1200,800))
ax1 = Axis(fig1[1,1], xlabel ="time", ylabel="G_mix")
heatmap!(ax1, t, b_out, G_mix_all, colormap=:balance, colorrange=(-0.009, 0.009))

save(joinpath(plot_dir, "G_mix_vs_time"), fig1)
println("saved gmix plot")

# yidx = Int(Ny ÷ 2)
# xidx = Int(64) #we want the b(y,z) to be sliced at the left x boundary

# Ra = 1e8 ############################################################## make this better bro ###########

# @info "Creating animation for Ra = $Ra with $Nt frames"

# n = Observable(1)

# title_bxz = @lift @sprintf(
#     "buoyancy on x-z plane [m/s²], Ra = %.2e, t = %.2f",
#     Ra, t[$n]
# )

# title_byz = @lift @sprintf(
#     "buoyancy on y-z plane [m/s²], Ra = %.2e, t = %.2f", 
#     Ra, t[$n]
# )

# # Lazy read: one (x,z) slice at one time index
# b_xzₙ = @lift begin
#     b[:, yidx, :, $n]
# end

# b_yzₙ = @lift begin
#     b[xidx, :, :, $n]
# end

# # # Uₙ = @lift begin
# # #     u_slice = Array(u[1:512, yidx, 1:64, $n])
# # #     v_slice = Array(v[1:512, yidx, 1:64, $n])
# # #     w_slice = Array(w[1:512, yidx, 1:64, $n])
# # #     sqrt.(u_slice.^2 .+ v_slice.^2 .+ w_slice.^2)
# # # end

# # #the size of Uₙ is Nx, Nz

# # #velocity magnitude

# b_ref = b[:, yidx, :, 2] #im using any time index that is not the initial in case theres no cold start

# b_ref_yz = b[xidx, :, :, 2] #second ref for yz plane example
# wet_xz = b_ref .!= 0.0  # bool array : true = fluid, false = hills # size = Nx, Nz
# wet_yz = b_ref_yz .!= 0.0 #bool array : true = fluid, false = hiills # size = Ny, Nz

# # println("size of wet mask", size(wet_xz))

# wet_masked_xz = Float64.(copy(wet_xz))
# wet_masked_xz[wet_xz] .= NaN

# wet_masked_yz = Float64.(copy(wet_yz))
# wet_masked_yz[wet_yz] .= NaN

# fig = Figure(size = (800, 1200))

# ax_bxz = Axis(
#     fig[1, 1];
#     title = title_bxz,
#     xlabel = L"x / Lx",
#     ylabel = L"z / H",
#     limits = ((-Lx/2, Lx/2), (-H, 0)),
#     aspect = Lx / H,
#     titlesize = 20
# )

# ax_byz = Axis(
#     fig[2, 1];
#     title = title_byz, 
#     xlabel = L"y / Ly",
#     ylabel = L"z / H",
#     limits = ((-Ly/2, Ly/2), (-H, 0)),
#     aspect = Ly / H,
#     titlesize = 20
# )


# B_lims = (-0.5, 0.5)

# hm_1 = heatmap!(
#     ax_bxz, x, z, b_xzₙ;
#     colormap = :balance,
#     colorrange = B_lims
# )

# hm_hill = heatmap!(
#     ax_bxz, x, z, wet_masked_xz[:, :], colormap=:turbid
# )

# Colorbar(fig[1, 2], hm_1)

# # U_lims = (0.0, 0.4)

# hm_2 = heatmap!(
#     ax_byz, y, z, b_yzₙ;
#     colormap = :balance, 
#     colorrange = B_lims, 
# )

# hm_hill = heatmap!(
#     ax_byz, y, z, wet_masked_yz[:, :], colormap=:turbid
# )

# Colorbar(fig[2, 2], hm_2)

# frames = 1:Nt   # or 1:5:Nt to subsample

# output_file = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/animations/GPU/GRC/RA1e8/4x_stretch/first194sec.mp4"

# record(fig, output_file, frames; framerate = 24) do i
#     n[] = i
# end

# @info "Animation saved to $output_file"

