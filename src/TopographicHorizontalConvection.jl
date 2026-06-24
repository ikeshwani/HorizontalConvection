module TopographicHorizontalConvection
    include("simulation.jl")
    #include("simulation_test.jl")
    include("SIM_GPU_TEST.jl")
    include("analysis/diagnostics.jl")
    include("analysis/streamfunction.jl")
    include("analysis/gmix.jl")
    include("analysis/energy_fluxes.jl")
    include("analysis/energetics.jl")
    include("analysis/regions.jl")
    include("analysis/io_utils.jl")
    include("grid_testing.jl")
    include("RANS_energies.jl")
end

#print the julia version in the scripts so we know which version is loaded

#start from the one that is working and make it more complex?

#ask for an interactive gpu node line by line do everything in the batch script (except module load julia)
#then line by line run your code 