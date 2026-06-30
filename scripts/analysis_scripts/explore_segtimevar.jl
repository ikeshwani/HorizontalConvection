# explore_segtimevar.jl
#
# A tiny PLAYGROUND to understand the SegTimeVar struct from kinetic_energetics.jl.
# It uses only 3 segments and reads a single grid point, so it runs in seconds.
# Every time you index the struct, it PRINTS which file + local step it read from,
# so you can see the "global time -> (file, step)" translation happen.
#
# Run interactively so the variables stick around to poke at:
#   cd scripts/
#   julia --project=../ -i analysis_scripts/explore_segtimevar.jl
# then at the julia> prompt try things like:
#   u[1, 1, 1, 5]        # one time step
#   u[1, 1, 1, 1:10]     # a span of steps
#   seg[200], loc[200]   # look at the translation tables directly

using NCDatasets
using CairoMakie

data_dir = "/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_flat_baseforcing_zerostart/"
segments = 1:3          # <-- just three files, keep it small

# ----------------------------------------------------------------------------------
# The struct: just a bundle of 4 things travelling together.
struct SegTimeVar
    datasets::Vector{NCDataset}   # the open files
    varname::String               # which variable, e.g. "u"
    seg::Vector{Int}              # global step -> which file (index into `datasets`)
    loc::Vector{Int}              # global step -> local step inside that file
end

# This is the ONLY clever part. `u[x,y,z,t]` secretly calls getindex(u, x,y,z,t).
# We print the translation so you can watch it, then read that one slab from disk.
function Base.getindex(s::SegTimeVar, xr, yr, zr, t::Integer)
    file  = s.seg[t]
    local_step = s.loc[t]
    println("  $(s.varname)[..., global step $t]  ->  file #$file (seg$(segments[file])), local step $local_step")
    return s.datasets[file][s.varname][xr, yr, zr, local_step]
end

# many steps: just call the single-step version for each and stack along time (dim 4)
Base.getindex(s::SegTimeVar, xr, yr, zr, ts::AbstractVector{<:Integer}) =
    cat((s[xr, yr, zr, t] for t in ts)...; dims=4)
# ----------------------------------------------------------------------------------

# Open the 3 files and BUILD the translation tables (this is the bookkeeping the
# struct hides). For every file, for every local step, record (which file, which step).
vel_ds = NCDataset[]
seg    = Int[]
loc    = Int[]
time   = Float64[]

for s in segments
    ds = NCDataset(joinpath(data_dir, "velocities_seg$(s).nc"), "r") 
    push!(vel_ds, ds) #array of all the velocity segment files
    file = length(vel_ds)                 # this file's position in vel_ds
    print(file, "\n")
    t_seg = ds["time"][:]
    for k in 1:length(t_seg)
        push!(seg, file)                  # global step lives in this file...
        push!(loc, k)                     # ...at local step k
        push!(time, t_seg[k])
    end
end

time

Nt = length(time)
println("\nOpened $(length(vel_ds)) files, $Nt total time steps (t = $(time[1]) .. $(time[end]))\n")

# Show the translation table around the first file boundary so the mapping is concrete.
b1 = findfirst(==(2), seg)   # first global step that lives in file #2

println("Translation table near the file #1 -> #2 boundary:")
for t in (b1-2):(b1+1)
    println("  global step $t  ->  file #$(seg[t]), local step $(loc[t]),  time $(time[t])")
end

# Make the struct for u. Now u BEHAVES like one big [x,y,z,t] array spanning all files.
#so now u acts like one big array 
#vel_ds is a 3d array of datafiles for velocity segments 1 through 3, so u holds data for all three segments
#seg is a vector of length time and returns how many time steps are associated with that segment, 
#so if the first 15 seconds, corresponding to the first 150 timesteps are held in segment 1, the first 150 entries of seg == 1
#loc is similar but it holds the time steps in each segment
#so again if the first 15 segments, corresponding to the first 150 timesteps are held in segment 1, col will count up from 1 to 150, then start over when at segment 2

u = SegTimeVar(vel_ds, "u", seg, loc)

println("\n--- now index it and watch the translation ---")
println("read u at global step 5:")
#so val 5 is the 

val5 = u[:, 1, :, 600]
println("  value = $val5\n")

heatmap(val5)

println("read u at global step $(b1) (just across into file #2):")
valb = u[1, 1, 1, b1]
println("  value = $valb\n")

println("read u over steps $(b1-1):$(b1+1) (a span that crosses the file boundary):")
span = u[1, 1, 1, (b1-1):(b1+1)]
println("  values = $(vec(span))")

println("\nDone. The files are still open as `vel_ds`; `u`, `seg`, `loc`, `time` are ready to poke at.")
