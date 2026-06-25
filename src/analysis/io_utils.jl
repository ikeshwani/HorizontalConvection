# io_utils.jl
#
# I/O helpers for the analysis pipeline:
#   - save_gmix_regions : write per-region G_mix(b,t) + ψ(x,z,t) to a NetCDF file
#   - load_chi_eps_mean : time-average χ, ε over the final window from oceanostics
#   - load_psib_tavg    : time-average ψ(x,b) from a psi_b NetCDF
#
# Physics/IO only — no plotting.

using NCDatasets
using Statistics
using Printf

export save_gmix_regions, load_chi_eps_mean, load_psib_tavg

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

# Time-average the dissipation diagnostics χ and ε over the last `avg_window`
# time units ending at `t_end`, reading from oceanostics_seg*.nc across
# `seg_range` and deduplicating overlapping segment steps.  Returns
# (chi_mean, eps_mean) as [Nx, Ny, Nz] arrays.
function load_chi_eps_mean(data_dir, seg_range, avg_window, t_end, Nx, Ny, Nz)
    chi_sum = zeros(Float64, Nx, Ny, Nz)
    eps_sum = zeros(Float64, Nx, Ny, Nz)
    n = 0
    t_last = -Inf
    for s in seg_range
        ofile = joinpath(data_dir, "oceanostics_seg$(s).nc")
        isfile(ofile) || continue
        ds = NCDataset(ofile)
        t_seg = Float64.(ds["time"][:])
        valid = findall(t_seg .> t_last)
        isempty(valid) && (close(ds); continue)
        t_range = valid[1]:valid[end]
        t_last  = t_seg[t_range[end]]
        in_win  = findall(t_seg[t_range] .>= t_end - avg_window)
        if !isempty(in_win)
            t_win = t_range[in_win[1]:in_win[end]]
            chi_sum .+= dropdims(sum(Float64.(ds["χ"][:, :, :, t_win]), dims=4), dims=4)
            eps_sum .+= dropdims(sum(Float64.(ds["ε"][:, :, :, t_win]), dims=4), dims=4)
            n += length(t_win)
        end
        close(ds)
    end
    n == 0 && error("no oceanostics steps found in averaging window")
    @printf("  χ,ε: averaged over %d steps\n", n)
    return chi_sum ./ n, eps_sum ./ n
end

# Load and time-average ψ(x,b) over the window [t_start, t_end] from a psi_b
# NetCDF file.  Handles both naming conventions:
#   control: variable "psi_b", coordinate "b"
#   hill:    variable "ψ_b",   coordinate "b_out"
# Returns (x, b, ψ_mean[Nx, Nb]).
function load_psib_tavg(file, t_start, t_end)
    ds    = NCDataset(file)
    x     = Float64.(ds["x"][:])
    b_key = haskey(ds, "b") ? "b" : "b_out"
    ψ_key = haskey(ds, "psi_b") ? "psi_b" : "ψ_b"
    b     = Float64.(ds[b_key][:])
    time  = Float64.(ds["time"][:])
    idx   = findall((time .>= t_start) .& (time .<= t_end))
    isempty(idx) && error("no time steps in window [$t_start, $t_end] in $file")
    @printf("  %s: averaging %d steps (t = %.2f → %.2f)\n",
            basename(file), length(idx), time[idx[1]], time[idx[end]])
    ψ_mean = dropdims(mean(Float64.(ds[ψ_key][:, :, idx[1]:idx[end]]), dims=3), dims=3)
    close(ds)
    return x, b, ψ_mean
end
