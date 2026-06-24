# io_utils.jl
#
# I/O helpers for the analysis pipeline:
#   - save_gmix_regions : write per-region G_mix(b,t) + ψ(x,z,t) to a NetCDF file
#
# Physics/IO only — no plotting.

using NCDatasets

export save_gmix_regions

# Write the regional G_mix density and the physical-space streamfunction to a
# NetCDF file.  `Gmix_regions` is a Dict name => [n_b, Nt] array and
# `region_precomp` the vector of named tuples from precompute_regions.
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
