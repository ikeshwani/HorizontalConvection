using NCDatasets
using Printf

using NCDatasets
using Printf

function combine_and_save()
    basepath = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/"
    outfile = joinpath(basepath, "combined_t194.nc")

    n_segs = collect(8:10)
    n_segments = length(n_segs)  # = 3

    ds_combined = NCDataset(joinpath(basepath, "combined_t125.nc"))
    datasets = [NCDataset(joinpath(basepath, "buoyancy_seg$(i).nc")) for i in n_segs]
    u_data   = [NCDataset(joinpath(basepath, "velocities_seg$(i).nc")) for i in n_segs]

    # Grid info from the first new segment (or from ds_combined — same grid)
    ds1 = datasets[1]
    x  = ds1["x_caa"][:]
    y  = ds1["y_aca"][:]
    z  = ds1["z_aac"][:]
    Δx = ds1["Δx_caa"][:]
    Δy = ds1["Δy_aca"][:]
    Δz = ds1["Δz_aac"][:]
    Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
    Lx, Ly, H  = ds1.attrib["Lx"], ds1.attrib["Ly"], ds1.attrib["H"]

    # No overlaps between segs 8-10, so take every timestep
    # Use LOCAL indices 1,2,3 (not 8,9,10) to index into valid_ranges
    valid_ranges = Vector{UnitRange{Int}}(undef, n_segments)
    for (local_i, global_i) in enumerate(n_segs)
        nt_i = length(datasets[local_i]["time"][:])
        valid_ranges[local_i] = 1:nt_i
    end

    # Build combined time axis: old + new segments
    t_0        = ds_combined["time"][:]
    Nt_0       = length(t_0)
    t_new      = reduce(vcat, datasets[local_i]["time"][valid_ranges[local_i]] for local_i in 1:n_segments)
    t_combined = vcat(t_0, t_new)
    Nt         = length(t_combined)

    println("total timesteps: $Nt  |  t_start = $(t_combined[1]),  t_end = $(t_combined[end])")

    # Allocate for ALL timesteps (old + new)
    b_combined = zeros(Float32, Nx, Ny, Nz, Nt)
    χ_combined = zeros(Float32, Nx, Ny, Nz, Nt)
    u_combined = zeros(Float32, Nx, Ny, Nz, Nt)

    # Load the already-combined data into the first Nt_0 slots
    b_combined[:, :, :, 1:Nt_0] = ds_combined["b"][:, :, :, :]
    χ_combined[:, :, :, 1:Nt_0] = ds_combined["chi"][:, :, :, :]
    u_combined[:, :, :, 1:Nt_0] = ds_combined["u"][:, :, :, :]

    # Append new segments starting after the old data
    t_offset = Nt_0
    for local_i in 1:n_segments
        rng = valid_ranges[local_i]
        n   = length(rng)
        out = t_offset+1 : t_offset+n
        @printf("writing segment %d (global seg %d): output steps %d:%d\n",
                local_i, n_segs[local_i], out[1], out[end])
        b_combined[:, :, :, out] = datasets[local_i]["b"][:, :, :, rng]
        χ_combined[:, :, :, out] = datasets[local_i]["chi"][:, :, :, rng]
        u_combined[:, :, :, out] = u_data[local_i]["u"][1:Nx, :, :, rng]
        t_offset += n
    end

    # Save to NetCDF
    NCDataset(outfile, "c") do ds
        ds.attrib["Nx"] = Nx; ds.attrib["Ny"] = Ny; ds.attrib["Nz"] = Nz
        ds.attrib["Lx"] = Lx; ds.attrib["Ly"] = Ly; ds.attrib["H"]  = H
        defDim(ds, "x",    Nx)
        defDim(ds, "y",    Ny)
        defDim(ds, "z",    Nz)
        defDim(ds, "time", Nt)
        defVar(ds, "x_caa",   x,          ("x",))
        defVar(ds, "y_aca",   y,          ("y",))
        defVar(ds, "z_aac",   z,          ("z",))
        defVar(ds, "Δx_caa",  Δx,         ("x",))
        defVar(ds, "Δy_aca",  Δy,         ("y",))
        defVar(ds, "Δz_aac",  Δz,         ("z",))
        defVar(ds, "time",    t_combined, ("time",))
        defVar(ds, "b",       b_combined, ("x", "y", "z", "time"))
        defVar(ds, "chi",     χ_combined, ("x", "y", "z", "time"))
        defVar(ds, "u",       u_combined, ("x", "y", "z", "time"))
    end

    # Close all datasets
    close(ds_combined)
    foreach(close, datasets)
    foreach(close, u_data)

    println("done saving woohoo")
    return x, y, z, Lx, Ly, H, t_combined, b_combined, χ_combined, u_combined
end

x, y, z, Lx, Ly, H, t, b, χ, u = combine_and_save()


# function combine_and_save()

#     basepath = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/512_128/"
#     outfile = joinpath(basepath, "combined_t194.nc")

#     #i already have a file that has the first 7 segments combined (up to t=125)
#     # now i just want to combine my "combined_t125.nc" with segments, 8, 9, 10 (up to t=194)
#     n_segs = collect(8:10)

#     ds_combined = NCDataset(joinpath(basepath, "combined_t125.nc"))

#     datasets = [NCDataset(joinpath(basepath, "buoyancy_seg$(i).nc")) for i in n_segs]
#     u_data = [NCDataset(joinpath(basepath, "velocities_seg$(i).nc")) for i in n_segs]

#     n_segments = length(datasets)

#     ds1 = datasets[1]
#     x = ds1["x_caa"][:]
#     y = ds1["y_aca"][:]
#     z = ds1["z_aac"][:]
#     Δx = ds1["Δx_caa"][:]
#     Δy = ds1["Δy_aca"][:]
#     Δz = ds1["Δz_aac"][:]
#     Nx, Ny, Nz = ds1.attrib["Nx"], ds1.attrib["Ny"], ds1.attrib["Nz"]
#     Lx, Ly, H = ds1.attrib["Lx"], ds1.attrib["Ly"], ds1.attrib["H"]

#     # hardcode the known overlap between segment three and segment four (its 8 timesteps)

#     valid_ranges = Vector{UnitRange{Int}}(undef, n_segments)
#     for i in n_segs
#         nt_i = length(datasets[i]["time"][:])
#         # if i == 3
#         #     valid_ranges[i] = 1 : nt_i - 8 #trim the last 8 timesteps from segment 3
#             valid_ranges[i] = 1 : nt_i
#         end
#     end

#     # combine the total time from all segments
#     t_0 = ds_combined["time"][:]
#     t_combined = reduce(vcat, datasets[i]["time"][valid_ranges[i]] for i in n_segs)
#     t_combined = vcat(t_0, t_combined)
#     Nt = length(t_combined)
#     println("total timesteps : $Nt t_start = $(t_combined[1]) , t_end = $(t_combined[end])")

#     b_combined = zeros(Float32, Nx, Ny, Nz, Nt)
#     χ_combined = zeros(Float32, Nx, Ny, Nz, Nt)
#     u_combined = zeros(Float32, Nx, Ny, Nz, Nt)

#     t_offset = 0
#     for i in n_segs
#         rng = valid_ranges[i]
#         n = length(rng)
#         out = t_offset+1 : t_offset+n

#         @printf("writing segment %d: output steps %d:%d\n", i, out[1], out[end])

#         b_combined[:, :, :, out] = datasets[i]["b"][:, :, :, rng]
#         χ_combined[:, :, :, out] = datasets[i]["chi"][:, :, :, rng]
#         u_combined[:, :, :, out] = u_data[i]["u"][1:Nx, :, :, rng]

#         t_offset += n
#     end

#     # now save to netcdf
#     NCDataset(outfile, "c") do ds 

#         ds.attrib["Nx"] = Nx; ds.attrib["Ny"] = Ny; ds.attrib["Nz"] = Nz
#         ds.attrib["Lx"] = Lx; ds.attrib["Ly"] = Ly; ds.attrib["H"] = H

#         defDim(ds, "x", Nx)
#         defDim(ds, "y", Ny)
#         defDim(ds, "z", Nz)
#         defDim(ds, "time", Nt)

#         defVar(ds, "x_caa", x, ("x",))
#         defVar(ds, "y_aca", y, ("y", ))
#         defVar(ds, "z_aac", z, ("z", ) )
#         defVar(ds, "Δx_caa", Δx, ("x", ))
#         defVar(ds, "Δy_aca", Δy, ("y", ))
#         defVar(ds, "Δz_aac", Δz, ("z", ))
#         defVar(ds, "time", t_combined, ("time", ))

#         defVar(ds, "b", b_combined, ("x", "y", "z", "time"))
#         defVar(ds, "chi", χ_combined, ("x", "y", "z", "time"))
#         defVar(ds, "u", u_combined, ("x", "y", "z", "time"))
#     end

#     println("done saving woohoo")
#     return x, y, z, Lx, Ly, H, t_combined, b_combined, χ_combined, u_combined
    
# end

# x, y, z, Lx, Ly, H, t, b, χ, u = combine_and_save()