using TopographicHorizontalConvection
using NCDatasets
using Printf

data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/"

# same snapshot gmix_CODF.jl used to build its axis: last step of the final segment
ds = NCDataset(joinpath(data_dir, "buoyancy_seg32.nc"))
Nx, Ny, Nz = ds.attrib["Nx"], ds.attrib["Ny"], ds.attrib["Nz"]
Δx = reshape(ds["Δx_caa"][:], Nx,1,1)
Δy = reshape(ds["Δy_aca"][:], 1,Ny,1)
Δz = reshape(ds["Δz_aac"][:], 1,1,Nz)
vol = Δx .* Δy .* Δz
wet = ds["b"][:,:,:,2] .!= 0
b_ax = Array{Float64}(ds["b"][:,:,:,end])
close(ds)

wet_idx = findall(vec(wet))
bvals  = vec(b_ax)[wet_idx]
volvals = vec(vol)[wet_idx]
Vtot = sum(volvals)


maximum(bvals)
minimum(bvals)

# reproduce gmix_CODF.jl's exact call
n_b_bins = 101
λ_bins = 1.0
b_edges = blended_b_edges(bvals, volvals, n_b_bins; λ=λ_bins)
println("number of edges: ", length(b_edges))

# global volume fraction in the suspicious gap
sel = (bvals .>= -0.65) .& (bvals .< -0.35)
Vgap = sum(volvals[sel])
@printf("global volume in [-0.65,-0.35) = %.5f%% of total domain wet volume\n", 100*Vgap/Vtot)

# show the actual edges bracketing that range
idxs_near = findall(e -> -0.7 <= e <= -0.3, b_edges)
println("edges landing in/near [-0.7,-0.3]:")
for i in idxs_near
    @printf("  edge[%d] = %.4f\n", i, b_edges[i])
end
println("gaps between consecutive edges in that neighborhood:")
for i in idxs_near[1:end-1]
    @printf("  [%.4f, %.4f)  width = %.4f\n", b_edges[i], b_edges[i+1], b_edges[i+1]-b_edges[i])
end

# for comparison: median edge width overall, and where the FINEST bins are
widths = diff(b_edges)
@printf("\noverall edge widths: median = %.4f, min = %.4f, max = %.4f\n",
        sort(widths)[length(widths)÷2], minimum(widths), maximum(widths))
imax = argmax(widths)
@printf("widest bin: [%.4f, %.4f)  width=%.4f\n", b_edges[imax], b_edges[imax+1], widths[imax])