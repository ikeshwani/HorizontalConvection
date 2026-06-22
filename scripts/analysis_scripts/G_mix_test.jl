using CairoMakie
using NCDatasets
using Statistics
using NaNStatistics

ds_b = NCDataset("/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/combined_t385.nc")
ds_psi = NCDataset("/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/psi_b_t385.nc")

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