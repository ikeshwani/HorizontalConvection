# regions.jl
#
# Spatial region definitions used by the regional G_mix / flux diagnostics:
#   - boundary_layer_depth : the zBL scaling z_BL = -(round(Lx·Ra^(-1/5)) + 0.02)
#   - gmix_region_masks    : named boolean masks (plume, boundary layer, basins, hills)
#   - precompute_regions   : flatten masks to linear indices + horizontal areas
#
# Physics only — no plotting.

export boundary_layer_depth, gmix_region_masks, precompute_regions

# Boundary-layer depth scaling.  zBL is negative (measured down from z=0), and
# the 0.02 pad pushes the boundary just below the diffusive layer.
function boundary_layer_depth(Lx, Ra)
    return -(round(Lx * Ra^(-1/5); digits=2) + 0.02)
end

# Build the named region masks on the (x, z) grid.  Each entry is
# (name, mask) where mask is a Bool array broadcast to [Nx, 1, Nz].
# The x-extents below partition the domain into a forcing plume, the surface
# boundary layer, and alternating basin/hill columns beneath the boundary layer.
function gmix_region_masks(x, z, Lx, Ra)
    zBL = boundary_layer_depth(Lx, Ra)
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
    return region_masks
end

# Precompute, per region, the linear indices of its wet cells and its horizontal
# (xy) footprint area.  `ΔA_2d` is the [Nx, Ny] cell-area array and `wet` the
# [Nx, Ny, Nz] wet mask.  Returns a vector of named tuples (; name, idxs, A_region).
function precompute_regions(region_masks, ΔA_2d, wet)
    return map(region_masks) do (name, mask)
        idxs     = findall(vec(mask .& wet))
        hmask    = dropdims(any(mask, dims=3), dims=3)
        A_region = sum(ΔA_2d .* hmask)
        (; name, idxs, A_region)
    end
end
