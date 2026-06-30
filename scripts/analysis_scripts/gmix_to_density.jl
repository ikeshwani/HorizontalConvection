using NCDatasets
using Printf

function compute_region_areas(buoy_file)
    ds  = NCDataset(buoy_file)
    x   = Float64.(ds["x_caa"][:])
    z   = Float64.(ds["z_aac"][:])
    Nx, Ny, Nz = ds.attrib["Nx"], ds.attrib["Ny"], ds.attrib["Nz"]
    Lx  = Float64(ds.attrib["Lx"])
    Ra  = Float64(ds.attrib["Ra"])
    Δx  = reshape(Float64.(ds["Δx_caa"][:]), Nx, 1, 1)
    Δy  = reshape(Float64.(ds["Δy_aca"][:]), 1, Ny, 1)
    b2  = Float64.(ds["b"][:, :, :, 2])   # second snapshot for wet mask
    close(ds)

    wet = b2 .!= 0

    zBL = -(round(Lx * Ra^(-1/5); digits=2) + 0.02)
    X = reshape(x, :, 1, 1)
    Z = reshape(z, 1, 1, :)

    region_masks = [
        ("plume",          X .< -1.8),
        ("boundary_layer", (Z .> zBL)  .& (X .>= -1.8)),
        ("basin0",         (X .>= -1.8) .& (X .< -1.35) .& (Z .<= zBL)),
        ("hill1",          (X .>= -1.35) .& (X .< -0.65) .& (Z .< zBL)),
        ("basin1",         (X .>= -0.65) .& (X .< -0.35) .& (Z .< zBL)),
        ("hill2",          (X .>= -0.35) .& (X .< 0.35)  .& (Z .< zBL)),
        ("basin2",         (X .>= 0.35)  .& (X .< 0.65)  .& (Z .< zBL)),
        ("hill3",          (X .>= 0.65)  .& (X .< 1.35)  .& (Z .<= zBL)),
        ("basin3",         (X .>= 1.35)  .& (Z .<= zBL)),
    ]

    ΔA    = Δx .* Δy
    ΔA_2d = dropdims(ΔA, dims=3)

    areas = Dict{String,Float64}()
    for (name, mask) in region_masks
        hmask       = dropdims(any(mask .& wet, dims=3), dims=3)
        areas[name] = sum(ΔA_2d .* hmask)
        @printf("  %-20s  A = %.6f\n", name, areas[name])
    end
    return areas
end

function normalize_gmix_file(gmix_infile, gmix_outfile, areas)
    region_names = ["plume", "boundary_layer", "basin0",
                    "hill1", "basin1", "hill2", "basin2", "hill3", "basin3"]

    ds_in = NCDataset(gmix_infile)
    b_out = Float64.(ds_in["b"][:])
    time  = Float64.(ds_in["time"][:])

    NCDataset(gmix_outfile, "c") do ds_out
        defDim(ds_out, "b",    length(b_out))
        defDim(ds_out, "time", length(time))
        defVar(ds_out, "b",    b_out, ("b",))
        defVar(ds_out, "time", time,  ("time",))

        for name in region_names
            key = "Gmix_$(name)"
            if !haskey(ds_in, key)
                println("  skip $key (not found in input)")
                continue
            end
            G = Float64.(ds_in[key][:, :])
            A = get(areas, name, NaN)
            @printf("  %-20s  A = %.6f\n", name, A)
            defVar(ds_out, key, Float32.(G ./ A), ("b", "time"))
        end
    end
    close(ds_in)
    println("saved → $gmix_outfile")
end

# ── 3-hill ────────────────────────────────────────────────────────────────────
println("\n=== 3-hill ===")
hill_dir  = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/"
hill_areas = compute_region_areas(joinpath(hill_dir, "buoyancy_seg1.nc"))
normalize_gmix_file(
    joinpath(hill_dir, "Gmix_regions_t385.nc"),
    joinpath(hill_dir, "Gmix_density_regions_t385.nc"),
    hill_areas,
)

# ── Control ───────────────────────────────────────────────────────────────────
println("\n=== Control ===")
ctrl_dir  = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_flat_baseforcing_zerostart/"
ctrl_areas = compute_region_areas(joinpath(ctrl_dir, "buoyancy_seg1.nc"))
normalize_gmix_file(
    joinpath(ctrl_dir, "Gmix_regions_Control_RA1e8_seg1to10.nc"),
    joinpath(ctrl_dir, "Gmix_density_regions_Control_RA1e8.nc"),
    ctrl_areas,
)
